"""Freeze an MD17 baseline and check a few training batches, without a full run.

Reads the raw NPZ directly using PyG MD17's float32 conversion, avoiding dataset
processing or writes to the input directory. Only training frames are evaluated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import RadiusGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.md17.train_md17_energy_based import (
    GAModel, EnergyPredictor, embed_inputs, get_model, predict_energy_and_forces,
    batch_energy_mse_sum_and_count, batch_force_mse_sum_and_count,
)
from src.utils import seed_everything

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'Unavailable (source may have been transferred without .git)'


def make_model(c, device):
    base = get_model(
        c['model_name'], in_mvc=1, out_mvc=c['out_mvc'], hidden_mvc=c['hidden_mvc'],
        in_sc=c['embed_dim'], out_sc=c['out_sc'], hidden_sc=c['hidden_sc'],
        edge_mvc=c['edge_mvc'], edge_sc=c['edge_sc'], num_layers=c['num_layers'],
        num_heads=c['num_heads'], dropout_prob=c['dropout'], activation=c['activation'],
        attention_type=c['attention_type'], message=c['message'],
    )
    base.use_sparse = c['use_sparse']
    return GAModel(base, EnergyPredictor(c['out_mvc'], c['out_sc']),
                   sc_dim=c['embed_dim'], edge_rbf_dim=c['num_rbf'],
                   edge_sc_hidden=c['edge_sc'], cutoff=c['cutoff'],
                   smooth_cutoff=c.get('smooth_cutoff', False),
                   radial_basis=c.get('radial_basis', 'gaussian')).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'experiments/md17/configs/aspirin.json')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--reference-dir', type=Path,
                   help='Check regenerated weights and batch against a saved local baseline')
    p.add_argument('--disable-dropout', action='store_true',
                   help='Diagnostic: disable dropout while retaining force-loss derivatives')
    a = p.parse_args()
    if a.steps < 1:
        p.error('--steps must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    c = json.loads(a.config.read_text())
    (a.output / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(c['seed'])
    torch.set_num_threads(1)

    files = set()
    for folder in ['src', 'experiments', 'geometric-algebra-transformer', 'egnn', 'Steerable-E3-GNN']:
        files.update(p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts)
    files.update((ROOT / 'src/data').glob('*.pt'))
    files.update(ROOT / name for name in ['pyproject.toml', 'uv.lock'])
    files.add(a.config.resolve())
    manifest = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}
    (a.output / 'source_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with tarfile.open(a.output / 'source.tar.gz', 'w:gz') as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    (a.output / 'pip_freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    (a.output / 'git_status.txt').write_text(git_output('status', '--short') + '\n')
    args = []
    for key, value in c.items():
        if isinstance(value, bool):
            if value:
                args.append('--' + key)
        else:
            args.extend(['--' + key, str(value)])
    (a.output / 'training_command.txt').write_text(shlex.join([
        '.venv/bin/python', 'experiments/md17/train_md17_energy_based.py', *args,
        '--data_dir', str(a.data_dir.resolve()), '--runs_dir', 'runs/md17/aspirin',
    ]) + '\n')

    raw = a.data_dir / 'raw/rmd17/npz_data/rmd17_aspirin.npz'
    split_dir = a.data_dir / 'raw/rmd17/splits'
    train_file = split_dir / f"index_train_{c['split_idx']:02d}.csv"
    test_file = split_dir / f"index_test_{c['split_idx']:02d}.csv"
    indices = np.loadtxt(train_file, dtype=int)
    np.random.default_rng(c['seed']).shuffle(indices)
    train, val = indices[:-50], indices[-50:]
    test = np.loadtxt(test_file, dtype=int)
    assert len(set(train) & set(val)) == 0
    assert len(set(indices) & set(test)) == 0
    splits = dict(train=train.tolist(), validation=val.tolist(), test=test.tolist())
    (a.output / 'split_indices.json').write_text(json.dumps(splits, indent=2) + '\n')
    raw_data = np.load(raw)
    z = torch.from_numpy(raw_data['nuclear_charges']).long()
    # Same conversion and reduction order as MD17 processing / load_dataset.
    energies = torch.from_numpy(raw_data['energies']).float()
    forces = torch.from_numpy(raw_data['forces']).float()
    mu = (energies[train] / z.numel()).mean()
    fs = forces[train].reshape(-1, 3)
    sigma = torch.sqrt((fs ** 2).sum() / (3 * fs.shape[0])).clamp_min(1e-8)
    coords = torch.from_numpy(raw_data['coords']).float()
    frames = train[:c['batch_size']]
    radius = RadiusGraph(r=c['cutoff'])
    batch = Batch.from_data_list([radius(Data(pos=coords[i].clone(), z=z.clone(),
            energy=energies[i].clone(), force=forces[i].clone())) for i in frames])
    torch.save(batch, a.output / 'training_batch.pt')
    report = dict(config=c, base_commit=git_output('rev-parse', 'HEAD'),
        torch=torch.__version__, cuda_build=torch.version.cuda,
        inputs_sha256={str(p): sha256(p) for p in [raw, train_file, test_file]},
        split_counts={k: len(v) for k, v in splits.items()},
        training_frame_indices=frames.tolist(), mu_E=mu.item(), sigma_F=sigma.item(),
        units='kcal/mol and kcal/mol/angstrom',
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        steps=a.steps, dropout_disabled=a.disable_dropout, test_set_evaluated=False)
    def save():
        (a.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    if not torch.cuda.is_available():
        report['status'] = 'GPU unavailable; baseline frozen, GPU checks not run'
        save()
        raise RuntimeError(report['status'])
    device = torch.device('cuda:0')
    report['gpu'] = torch.cuda.get_device_name(device)
    report['compute_capability'] = list(torch.cuda.get_device_capability(device))
    seed_everything(c['seed'])
    model = make_model(c, device)
    report['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(initial, a.output / 'initial_state.pt')
    if a.reference_dir is not None:
        reference = torch.load(a.reference_dir / 'initial_state.pt', map_location='cpu', weights_only=True)
        reference_batch = torch.load(a.reference_dir / 'training_batch.pt', map_location='cpu', weights_only=False)
        # Preserve comparison with the original eight-frame check when using a
        # larger current batch; the additional frames are validated below too.
        compared_batch = Batch.from_data_list(batch.to_data_list()[:reference_batch.num_graphs])
        report['local_reference'] = dict(
            initial_weights_bitwise_equal=all(torch.equal(initial[k], reference[k]) for k in initial),
            batch_prefix_bitwise_equal=all(torch.equal(compared_batch[k], reference_batch[k]) for k in
                                   ['pos', 'z', 'energy', 'force', 'edge_index', 'batch']),
        )
        assert all(report['local_reference'].values()), 'Local reference weights/batch differ'
    g = embed_inputs(batch.clone(), torch.device('cpu'), c['use_sparse'])
    report['edges'] = batch.edge_index.shape[1]
    report['edge_multivector_max_abs'] = g.edge_attr_mv.abs().max().item()
    assert report['edge_multivector_max_abs'] > 0

    def predict(b):
        return predict_energy_and_forces('pgagnn', model, b.clone(), device, mu, sigma,
                use_sparse=c['use_sparse'], sdpa_mode=c['sdpa_mode'])

    def output(b):
        r = predict(b)
        return r[0].detach().cpu(), r[3].detach().cpu()

    def max_error(x, y):
        return (x - y).abs().max().item()

    def symmetry():
        model.eval()
        e, f = output(batch)
        report_errors = {}
        angle = torch.tensor(0.71)
        co, si = angle.cos(), angle.sin()
        rotation = torch.tensor([[co, -si, 0], [si, co, 0], [0, 0, 1.]])
        for name, matrix, shift in [
            ('translation', torch.eye(3), torch.tensor([1.1, -0.6, 0.9])),
            ('rotation', rotation, torch.zeros(3)),
            ('reflection', torch.diag(torch.tensor([-1., 1., 1.])), torch.zeros(3)),
        ]:
            b = batch.clone()
            b.pos = b.pos @ matrix.T + shift
            et, ft = output(b)
            expected_force = f @ matrix.T
            report_errors[name] = dict(energy_max_abs=max_error(et, e),
                force_max_abs=max_error(ft, expected_force),
                passed=bool(torch.allclose(et, e, atol=2e-5, rtol=2e-4) and
                            torch.allclose(ft, expected_force, atol=2e-5, rtol=2e-4)))
        e2, f2 = output(batch)
        report_errors['repeat_evaluation'] = dict(energy_max_abs=max_error(e2, e),
                force_max_abs=max_error(f2, f), bitwise_equal=bool(torch.equal(e2, e) and torch.equal(f2, f)))
        return report_errors

    started = time.monotonic()
    try:
        model.eval()
        torch.save(dict(zip(['energy_normalized', 'forces_normalized'], output(batch))),
                   a.output / 'initial_predictions.pt')
        report['initial_symmetry'] = symmetry()
        save()
        print('Initial symmetry:', report['initial_symmetry'], flush=True)
        sequences, final_states = [], []
        for repeat in range(2):
            model.load_state_dict(initial)
            seed_everything(c['seed'])
            model.train()
            if a.disable_dropout:
                # Only the outer flag controls create_graph in force prediction.
                # Child modules stay in eval mode, disabling all dropout variants.
                model.eval()
                model.training = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'],
                          weight_decay=c['weight_decay'], amsgrad=True)
            losses = []
            for step in range(a.steps):
                optimizer.zero_grad(set_to_none=True)
                r = predict(batch)
                force_sum, atoms = batch_force_mse_sum_and_count(r[3], r[5], r[6], sigma)
                energy_sum, graphs = batch_energy_mse_sum_and_count(r[0], r[2], r[6].sum(1), mu, sigma)
                loss = c['force_weight'] * force_sum / (3 * atoms) + c['energy_weight'] * energy_sum / graphs
                assert torch.isfinite(loss), 'Nonfinite loss'
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'Nonfinite gradient'
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameter'
                losses.append(dict(loss=loss.item(), gradient_norm=norm.item()))
                print(f'repeat {repeat+1}, step {step+1}: {losses[-1]}', flush=True)
            sequences.append(losses)
            final_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        report['training_sequences'] = sequences
        report['repeat_training'] = dict(
            losses_equal=sequences[0] == sequences[1],
            parameters_bitwise_equal=all(torch.equal(final_states[0][k], final_states[1][k]) for k in initial),
            parameter_max_abs=max(max_error(final_states[0][k], final_states[1][k]) for k in initial))
        report['after_steps_symmetry'] = symmetry()
        torch.save(dict(zip(['energy_normalized', 'forces_normalized'], output(batch))),
                   a.output / 'after_steps_predictions.pt')
        torch.save(final_states[-1], a.output / 'after_steps_state.pt')
        report['all_symmetry_checks_passed'] = all(
            v['passed'] for phase in ['initial_symmetry', 'after_steps_symmetry']
            for v in report[phase].values() if 'passed' in v)
        report['status'] = 'checks completed; inspect individual outcomes'
    except Exception as exc:
        report['status'] = 'GPU validation failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        torch.cuda.synchronize()
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_allocated_MiB'] = torch.cuda.max_memory_allocated() / 1024**2
        report['peak_reserved_MiB'] = torch.cuda.max_memory_reserved() / 1024**2
        save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
