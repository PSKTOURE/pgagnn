import os
import signal
import tempfile

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel


def save_checkpoint(path: str, payload: dict):
    """Atomically save a training checkpoint and keep a .bak of the previous one."""
    checkpoint_dir = os.path.dirname(path) or "."
    checkpoint_name = os.path.basename(path)
    backup_path = f"{path}.bak"

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{checkpoint_name}.tmp.",
        dir=checkpoint_dir,
    )
    os.close(fd)

    try:
        torch.save(payload, tmp_path)
        with open(tmp_path, "rb") as tmp_file:
            os.fsync(tmp_file.fileno())

        if os.path.exists(path):
            os.replace(path, backup_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _normalize_rng_state(state) -> torch.ByteTensor | None:
    """Convert serialized RNG state payloads to CPU ByteTensor format."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state(ckpt: dict):
    if "rng_state" in ckpt:
        rng_state = _normalize_rng_state(ckpt["rng_state"])
        if rng_state is not None:
            torch.set_rng_state(rng_state)

    cuda_rng_state = ckpt.get("cuda_rng_state")
    if torch.cuda.is_available() and cuda_rng_state is not None:
        if isinstance(cuda_rng_state, (list, tuple)):
            torch.cuda.set_rng_state_all(
                [_normalize_rng_state(state) for state in cuda_rng_state if state is not None]
            )
        else:
            cuda_rng_state = _normalize_rng_state(cuda_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

    if "np_rng_state" in ckpt:
        np.random.set_state(ckpt["np_rng_state"])


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: str = "cpu",
    ema_model: AveragedModel | None = None,
    swa_model: AveragedModel | None = None,
    swa_scheduler=None,
) -> dict:
    """Load common training state from a checkpoint and return its payload."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if ema_model is not None and ckpt.get("ema_state_dict") is not None:
        ema_model.load_state_dict(ckpt["ema_state_dict"])

    if swa_model is not None and ckpt.get("swa_state_dict") is not None:
        swa_model.load_state_dict(ckpt["swa_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if swa_scheduler is not None and ckpt.get("swa_scheduler_state_dict") is not None:
        swa_scheduler.load_state_dict(ckpt["swa_scheduler_state_dict"])

    _restore_rng_state(ckpt)
    return ckpt


def parse_signals(signal_list: str | None):
    """Parse comma-separated signal names into signal numbers."""
    if not signal_list:
        return None
    resolved = []
    for raw_name in signal_list.split(","):
        name = raw_name.strip().upper()
        if not name:
            continue
        attr = f"SIG{name}"
        sig = getattr(signal, attr, None)
        if sig is None:
            raise ValueError(f"Unsupported signal name: {name}")
        resolved.append(sig)
    return resolved


class StopSignalHandler:
    """Handle SLURM preemption signals for graceful checkpointing."""

    def __init__(self):
        self.stop_requested = False
        self.last_signal = None
        self._save_fn = None

    def install(self, save_fn, signals_to_handle):
        self._save_fn = save_fn
        for sig in signals_to_handle:
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame):
        self.stop_requested = True
        try:
            self.last_signal = signal.Signals(signum).name
        except ValueError:
            self.last_signal = str(signum)
        if self._save_fn is not None:
            try:
                self._save_fn(interrupted=True, signal_name=self.last_signal)
            except (RuntimeError, OSError, ValueError, TypeError) as exc:
                print(f"Failed to save checkpoint on signal {self.last_signal}: {exc}")
