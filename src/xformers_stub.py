# Copyright (c) 2024
"""Stub module for xformers when running in environments or on GPUs where xformers cannot be installed.

GATr natively falls back to PyTorch's torch.nn.functional.scaled_dot_product_attention (SDPA)
when attention masks are standard PyTorch Tensors or None, and FORCE_XFORMERS is False.
This stub allows GATr to be imported and executed without requiring xformers.
"""

import os
import sys
from types import ModuleType


def _install_stub() -> None:
    """Install minimal stub modules for xformers into sys.modules."""
    xformers = ModuleType("xformers")
    ops = ModuleType("xformers.ops")
    fmha = ModuleType("xformers.ops.fmha")
    attn_bias = ModuleType("xformers.ops.fmha.attn_bias")

    class AttentionBias:
        """Stub for xformers.ops.AttentionBias."""

    class BlockDiagonalMask(AttentionBias):
        """Stub for xformers.ops.fmha.BlockDiagonalMask."""

    def _memory_efficient_attention(*args, **kwargs):
        raise NotImplementedError(
            "xformers is not installed in this environment. "
            "GATr only supports standard Tensor attention masks via PyTorch native SDPA "
            "when xformers is unavailable."
        )

    ops.AttentionBias = AttentionBias
    ops.memory_efficient_attention = _memory_efficient_attention
    ops.fmha = fmha
    fmha.BlockDiagonalMask = BlockDiagonalMask
    fmha.attn_bias = attn_bias
    attn_bias.BlockDiagonalMask = BlockDiagonalMask
    xformers.ops = ops

    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops
    sys.modules["xformers.ops.fmha"] = fmha
    sys.modules["xformers.ops.fmha.attn_bias"] = attn_bias


def ensure_xformers_stub() -> bool:
    """Ensure xformers is importable even on systems/GPUs where it is not installed.

    If DISABLE_XFORMERS=1 is set in the environment, or if importing xformers fails,
    registers minimal stub modules in sys.modules.

    Returns:
        bool: True if the stub was installed, False if real xformers is active.
    """
    if os.environ.get("DISABLE_XFORMERS", "0").lower() in ("1", "true", "yes"):
        _install_stub()
        return True

    try:
        import xformers
        import xformers.ops

        if not hasattr(xformers.ops, "memory_efficient_attention"):
            _install_stub()
            return True
        return False
    except (ImportError, Exception):  # noqa: BLE001
        _install_stub()
        return True
