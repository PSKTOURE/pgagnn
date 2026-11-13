"""Evaluate a validation-selected aspirin checkpoint using its frozen training source."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    source = args.study_dir / 'source'
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import RadiusGraph
    from experiments.md17.validate_md17 import make_model
    from experiments.md17.train_md17_energy_based import evaluate
    from src.utils import seed_everything

    assert torch.cuda.is_available(), 'GPU required'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    names = ['gaussian_c9', 'gaussian_c11', 'bessel_c9', 'bessel_c11']
    selected = {}
    for name in names:
        state = torch.load(args.study_dir / name / 'pgagnn/selected_model.pth',
                           map_location='cpu', weights_only=False)
        selected[name] = {k: v for k, v in state.items() if k != 'model'}
    winner = min(names, key=lambda name: selected[name]['val_force_mae_mev_A'])
    print('Validation selection:', json.dumps(selected), flush=True)
    print('Winner:', winner, flush=True)
    run = args.study_dir / winner
    config = json.loads((run / 'pgagnn/config.json').read_text())
    split_path = run / 'preflight/split_indices.json'
    splits = json.loads(split_path.read_text())
    assert len(splits['train']) == 950 and len(splits['validation']) == 50
    assert len(splits['test']) == 1000
    for x, y in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(splits[x]) & set(splits[y])
    official_test = args.data_dir / f"raw/rmd17/splits/index_test_{config['split_idx']:02d}.csv"
    assert np.array_equal(np.loadtxt(official_test, dtype=int), splits['test'])
    raw_path = args.data_dir / 'raw/rmd17/npz_data/rmd17_aspirin.npz'
    preflight = json.loads((run / 'preflight/validation.json').read_text())
    with np.load(raw_path) as raw:
        z = torch.from_numpy(raw['nuclear_charges']).long()
        energies = torch.from_numpy(raw['energies']).float()
        forces = torch.from_numpy(raw['forces']).float()
        coords = torch.from_numpy(raw['coords']).float()
    train = splits['train']
    mu = (energies[train] / z.numel()).mean()
    train_forces = forces[train].reshape(-1, 3)
    sigma = torch.sqrt((train_forces ** 2).sum() / (3 * len(train_forces))).clamp_min(1e-8)
    assert mu.item() == preflight['mu_E']
    assert sigma.item() == preflight['sigma_F']
    radius = RadiusGraph(r=config['cutoff'])
    def loader(indices):
        frames = [radius(Data(pos=coords[i].clone(), z=z.clone(),
                  energy=energies[i].clone(), force=forces[i].clone())) for i in indices]
        return DataLoader(frames, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    seed_everything(config['seed'])
    device = torch.device('cuda:0')
    model = make_model(config, device)
    checkpoint = run / 'pgagnn/selected_model.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    def measure(indices):
        return evaluate(config['model_name'], model, loader(indices), device, mu, sigma,
                        config['force_weight'], config['energy_weight'],
                        sdpa_mode=config['sdpa_mode'])
    validation = measure(splits['validation'])
    print('Validation reproduction:', json.dumps(validation), flush=True)
    assert abs(validation['force_mae_mev_A'] - selected[winner]['val_force_mae_mev_A']) < 0.02
    result = dict(recipe=winner, selection_criterion='minimum selected-checkpoint validation force MAE',
                  candidates=selected, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  epoch=state['epoch'], validation=validation, mu_E=mu.item(), sigma_F=sigma.item(),
                  test_count=len(splits['test']), split_sha256=sha256(split_path),
                  frozen_source=str(source), trainer_sha256=sha256(source / 'experiments/md17/train_md17_energy_based.py'),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0), config=config)
    result['test'] = measure(splits['test'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('TEST RESULT:', json.dumps(result['test']), flush=True)
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
