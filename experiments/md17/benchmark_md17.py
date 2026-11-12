"""Time complete training/validation epochs without evaluating the test set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.md17.validate_md17 import make_model, sha256
from experiments.md17.train_md17_energy_based import load_dataset, train_one_epoch, evaluate
from src.checkpointing import save_checkpoint
from src.utils import seed_everything


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=2)
    a = p.parse_args()
    if a.epochs < 2:
        p.error('Use at least two epochs to separate startup from steady throughput')
    a.output.mkdir(parents=True, exist_ok=False)
    config_path = root / 'experiments/md17/configs/aspirin.json'
    c = json.loads(config_path.read_text())
    (a.output / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    seed_everything(c['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this throughput measurement')
    device = torch.device('cuda:0')
    start = time.monotonic()
    for name in ['index_train_01.csv', 'index_test_01.csv']:
        if not (a.data_dir / 'raw/rmd17/splits' / name).is_file():
            raise RuntimeError(f'Missing official split: {name}')
    train, val, _test, mu, sigma = load_dataset(
        c['dataset'], c['cutoff'], c['batch_size'], c['num_workers'], c['seed'],
        c['split_idx'], str(a.data_dir))
    # Creating the test subset is part of the loader; never iterate over it here.
    frozen = json.loads((root / 'runs/md17/aspirin_baseline/2026-11-11/split_indices.json').read_text())
    assert list(train.dataset.indices()) == frozen['train']
    assert list(val.dataset.indices()) == frozen['validation']
    model = make_model(c, device)
    assert sum(p.numel() for p in model.parameters()) == 2291688
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'],
                                 weight_decay=c['weight_decay'], amsgrad=True)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(c['ema_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=c['swa_start_epoch'], eta_min=1e-6)
    sources = [*root.glob('src/*.py'), root / 'experiments/md17/train_md17_energy_based.py',
               root / 'experiments/md17/validate_md17.py', Path(__file__), config_path]
    report = dict(config=c, pilot_epochs=a.epochs, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda_build=torch.version.cuda, node=os.uname().nodename,
        train_frames=len(train.dataset), validation_frames=len(val.dataset),
        train_batches=len(train), validation_batches=len(val), test_set_evaluated=False,
        source_sha256={str(f.relative_to(root)): sha256(f) for f in sources},
        dataset_setup_seconds=time.monotonic()-start, epochs=[])
    def save():
        (a.output / 'timing.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    best = float('inf')
    for epoch in range(1, a.epochs + 1):
        torch.cuda.synchronize()
        begin = time.monotonic()
        losses = train_one_epoch('pgagnn', model, train, optimizer, device, mu, sigma,
            c['force_weight'], c['energy_weight'], sdpa_mode=c['sdpa_mode'], ema_model=ema)
        torch.cuda.synchronize()
        trained = time.monotonic()
        metrics = evaluate('pgagnn', ema, val, device, mu, sigma,
            c['force_weight'], c['energy_weight'], sdpa_mode=c['sdpa_mode'])
        torch.cuda.synchronize()
        validated = time.monotonic()
        scheduler.step()
        improved = metrics['force_mae_mev_A'] < best
        if improved:
            best = metrics['force_mae_mev_A']
            torch.save(dict(model=ema.module.state_dict(), epoch=epoch), a.output / 'best_model.pth')
        save_checkpoint(str(a.output / 'pilot_state.pt'), dict(epoch=epoch,
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            ema_state_dict=ema.state_dict(), scheduler_state_dict=scheduler.state_dict()))
        row = dict(epoch=epoch, train_seconds=trained-begin, validation_seconds=validated-trained,
            checkpoint_seconds=time.monotonic()-validated, total_seconds=time.monotonic()-begin,
            train_loss=losses[0], validation=metrics, best_checkpoint_written=improved,
            learning_rate=optimizer.param_groups[0]['lr'],
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
        report['epochs'].append(row)
        report['estimated_3500_epoch_hours'] = row['total_seconds'] * c['epochs'] / 3600
        save()
        print(json.dumps(row), flush=True)
    report['elapsed_seconds'] = time.monotonic()-start
    report['estimate_scope'] = ('Linear extrapolation of last measured epoch; excludes queueing, '
        'interruptions, final testing and extra SWA validation from epoch 3000.')
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
