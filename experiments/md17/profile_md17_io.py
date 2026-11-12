"""Compare home and node-local dataset access, separating loader and step time."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.md17.validate_md17 import make_model
from experiments.md17.train_md17_energy_based import load_dataset, train_one_epoch
from src.checkpointing import save_checkpoint
from src.utils import seed_everything


class TimedBatches:
    def __init__(self, loader, count):
        self.loader, self.count = loader, count
        self.wait, self.step = [], []

    def __iter__(self):
        start = time.monotonic()
        iterator = iter(self.loader)
        self.iterator_startup = time.monotonic() - start
        for _ in range(self.count):
            start = time.monotonic()
            batch = next(iterator)
            self.wait.append(time.monotonic() - start)
            start = time.monotonic()
            yield batch
            torch.cuda.synchronize()
            self.step.append(time.monotonic() - start)
        start = time.monotonic()
        del iterator
        self.iterator_shutdown = time.monotonic() - start


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', type=int, default=20)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    c = json.loads((root / 'experiments/md17/configs/aspirin.json').read_text())
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    report = dict(config=c, gpu=torch.cuda.get_device_name(), batches=a.batches,
                  test_set_evaluated=False, trials=[])
    def save():
        (a.output / 'io_profile.json').write_text(json.dumps(report, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='pgagnn-io-', dir=os.environ.get('TMPDIR', '/tmp')) as tmp:
        local = Path(tmp) / 'md17'
        start = time.monotonic()
        shutil.copytree(a.data_dir, local)
        report['staging_seconds'] = time.monotonic() - start
        report['node_local_path'] = str(local)
        for label, data in [('home_first', a.data_dir), ('node_local', local), ('home_repeat', a.data_dir)]:
            seed_everything(c['seed'])
            start = time.monotonic()
            train, _, _, mu, sigma = load_dataset(c['dataset'], c['cutoff'], c['batch_size'],
                c['num_workers'], c['seed'], c['split_idx'], str(data))
            setup_seconds = time.monotonic() - start
            model = make_model(c, torch.device('cuda'))
            opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'], amsgrad=True)
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(c['ema_decay']))
            timed = TimedBatches(train, min(a.batches, len(train)))
            torch.cuda.synchronize()
            start = time.monotonic()
            losses = train_one_epoch('pgagnn', model, timed, opt, torch.device('cuda'), mu, sigma,
                c['force_weight'], c['energy_weight'], sdpa_mode=c['sdpa_mode'], ema_model=ema)
            elapsed = time.monotonic() - start
            payload = dict(model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                           ema_state_dict=ema.state_dict())
            checkpoint_times = {}
            for location, folder in [('home', a.output), ('node_local', Path(tmp))]:
                target = folder / 'io_test_state.pt'
                start = time.monotonic()
                save_checkpoint(str(target), payload)
                checkpoint_times[location] = time.monotonic() - start
                target.unlink()
            row = dict(location=label, dataset_setup_seconds=setup_seconds,
                total_training_seconds=elapsed, loader_startup_seconds=timed.iterator_startup,
                loader_shutdown_seconds=timed.iterator_shutdown, batch_wait_seconds=timed.wait,
                training_step_seconds=timed.step,
                steady_batch_wait_mean_seconds=sum(timed.wait[2:])/len(timed.wait[2:]),
                steady_step_mean_seconds=sum(timed.step[2:])/len(timed.step[2:]),
                checkpoint_seconds=checkpoint_times, train_loss=losses[0])
            report['trials'].append(row)
            save()
            print(json.dumps(row), flush=True)
            del model, opt, ema, payload, train
            torch.cuda.empty_cache()
    report['note'] = ('Step time includes host-to-device transfer, embedding, energy/force prediction, '
                      'force-loss backward, optimizer and EMA; GPU synchronized after every batch. '
                      'Only training frames used. Temporary staged data removed.')
    save()


if __name__ == '__main__':
    main()
