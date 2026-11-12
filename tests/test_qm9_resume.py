"""Completed-epoch restart equivalence, including shuffled batches and EMA selection."""
import os
import random
import signal

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from experiments.qm9 import train_qm9 as qm9
from src.checkpointing import StopSignalHandler


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize('workers', [0, 1])
def test_qm9_signal_resume_matches_uninterrupted_training(tmp_path, monkeypatch, workers):
    val_loader, test_loader = object(), object()
    progress = {'epoch': 0, 'interrupt': None, 'order': [], 'test_calls': 0}
    handler = None
    checkpoint = None

    def train_epoch(model, loader, criterion, optimizer, *args, ema_model=None, **kwargs):
        progress['epoch'] += 1
        model.train()
        loss_sum = 0
        order = []
        for batch_index, (x, y) in enumerate(loader):
            order.extend(x[:, 0].tolist())
            optimizer.zero_grad(set_to_none=True)
            scale = 1 + (torch.rand(()).item() + random.random() + np.random.rand()) / 10
            loss = criterion(model(x), y) * scale
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            loss_sum += loss.item()
            if progress['epoch'] == progress['interrupt'] and batch_index == 0:
                os.kill(os.getpid(), signal.SIGUSR1)
                assert handler.stop_requested
                # The signal must not overwrite the preceding complete epoch.
                previous = torch.load(checkpoint, weights_only=False)
                assert previous['epoch'] == progress['epoch'] - 1
                assert len(previous['losses']['train_loss']) == previous['epoch']
        progress['order'].append(order)
        return loss_sum / len(loader)

    def evaluate(model, loader, *args, **kwargs):
        model.eval()
        if loader is test_loader:
            progress['test_calls'] += 1
            return 0.25
        assert loader is val_loader
        # The best EMA model is before the restart, so it must survive resume.
        return 0.1 if progress['epoch'] == 1 else 1.0

    monkeypatch.setattr(qm9, 'train_one_epoch', train_epoch)
    monkeypatch.setattr(qm9, 'evaluate', evaluate)

    def make_inputs(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        model = nn.Sequential(nn.Linear(1, 4), nn.SiLU(), nn.Dropout(.25), nn.Linear(4, 1)).double()
        x = torch.arange(1, 10, dtype=torch.float64).unsqueeze(-1)
        data = TensorDataset(x, x * .3)
        loader = DataLoader(data, batch_size=3,
                            sampler=RandomSampler(data, generator=torch.Generator().manual_seed(42)),
                            generator=torch.Generator().manual_seed(43), num_workers=workers,
                            persistent_workers=workers > 0,
                            multiprocessing_context='spawn' if workers else None)
        return model, loader

    def run(directory, model, loader, resume=None):
        nonlocal handler, checkpoint
        directory.mkdir(exist_ok=True)
        checkpoint = directory / 'last_model.pt'
        handler = StopSignalHandler()
        return qm9.train_model(model, loader, val_loader, test_loader, nn.L1Loss(),
                               target_idx=0, device=torch.device('cpu'), num_epochs=4,
                               warmup_epochs=1, target_mean=0., target_mad=1., lr=5e-4,
                               weight_decay=5e-7, patience=10, ema_decay=.9,
                               checkpoint_path=str(directory / 'best_model.pth'),
                               last_checkpoint_path=str(checkpoint), resume_from=resume,
                               stop_handler=handler, signals_to_handle=[signal.SIGUSR1], no_tqdm=True)

    old_signal = signal.getsignal(signal.SIGUSR1)
    try:
        full_model, loader = make_inputs(234)
        full_results = run(tmp_path / 'full', full_model, loader)
        full_order = progress['order'][:]
        full_ckpt = torch.load(checkpoint, weights_only=False)
        assert progress['test_calls'] == 1

        progress.update(epoch=0, interrupt=2, order=[], test_calls=0)
        split_model, loader = make_inputs(234)
        stopped = run(tmp_path / 'split', split_model, loader)
        assert stopped['test_loss'] is None
        assert progress['test_calls'] == 0
        stopped_ckpt = torch.load(checkpoint, weights_only=False)
        assert stopped_ckpt['epoch'] == 2
        assert len(stopped_ckpt['losses']['train_loss']) == 2
        assert stopped_ckpt['best_state']['epoch'] == 1
        resume = str(checkpoint)

        # A fresh process would construct a new model/loader before loading.
        resumed_model, loader = make_inputs(999)
        progress['interrupt'] = None
        resumed_results = run(tmp_path / 'split', resumed_model, loader, resume)
        resumed_ckpt = torch.load(checkpoint, weights_only=False)
        assert progress['test_calls'] == 1
        assert progress['order'] == full_order
        assert full_results == resumed_results
        for key in ['epoch', 'model_state_dict', 'ema_state_dict', 'optimizer_state_dict',
                    'scheduler_state_dict', 'best_state', 'epochs_without_improve',
                    'train_sampler_state', 'rng_state',
                    'np_rng_state', 'python_rng_state', 'losses']:
            assert_tree_equal(full_ckpt[key], resumed_ckpt[key])
        assert_tree_equal(full_model.state_dict(), resumed_model.state_dict())
        assert_tree_equal(resumed_model.state_dict(), resumed_ckpt['best_state']['model'])
        assert any(not torch.equal(value, resumed_ckpt['model_state_dict'][key])
                   for key, value in resumed_ckpt['best_state']['model'].items())
    finally:
        signal.signal(signal.SIGUSR1, old_signal)
