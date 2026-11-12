"""Short real-data GPU compatibility check for the CRIANN QM9 environment."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph

from experiments.qm9.train_qm9 import (
    GraphRegressor, ModelWrapper, embed_inputs, get_model, get_target_readout_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--largest', action='store_true', help='Use the largest molecules for a memory stress check.')

    args = parser.parse_args()
    assert torch.cuda.is_available(), 'This check requires a Slurm GPU allocation'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    dataset = QM9(args.data_dir, transform=RadiusGraph(r=5.0))
    if args.largest:
        counts = dataset.slices['pos'][1:] - dataset.slices['pos'][:-1]
        selected = dataset[counts.topk(min(args.batch_size, len(dataset))).indices]
    else:
        selected = dataset[:args.batch_size]
    batch = next(iter(DataLoader(selected, batch_size=args.batch_size)))
    reflected = batch.clone()
    reflection = torch.diag(torch.tensor([-1., 1., 1.], device=device))
    reflected.pos = batch.pos @ reflection.cpu().T + torch.tensor([.7, -.4, 1.2])
    report = {
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch', 'numpy', 'scipy', 'torch-geometric', 'torch-scatter',
                      'torch-cluster', 'einops', 'tensorboard']},
        'torch_path': torch.__file__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(),
        'device_memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
        'batch_size': batch.num_graphs, 'largest_molecules': args.largest, 'dataset_size': len(dataset),
        'nodes': batch.num_nodes, 'geometry_encoding': 'centered_euclidean_dual',
        'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ['src/primitives.py', 'src/pgagnn.py',
                                       'experiments/qm9/train_qm9.py']},
        'targets': {},
    }
    for target in ['U0', 'alpha', 'mu']:
        torch.manual_seed(42)
        mode, pool, late = get_target_readout_config(target)
        base = get_model('pgagnn', in_mvc=1, out_mvc=16, hidden_mvc=16,
                         in_sc=64, out_sc=32, hidden_sc=128, edge_mvc=1, edge_sc=128,
                         num_layers=3, num_heads=8, attention_type='gatr_sparse')
        base.use_sparse = True
        model = ModelWrapper('pgagnn', base,
                             GraphRegressor(16, 32, pool_type=pool, late_pooling=late,
                                            readout_mode=mode)).to(device)
        original = embed_inputs(batch, 'trivector', device, True)
        mirror = embed_inputs(reflected, 'trivector', device, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-7)
        torch.cuda.reset_peak_memory_stats()
        model.train()
        timings = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(original)
            loss = prediction.square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        model.eval()
        with torch.no_grad():
            a, b = model(original), model(mirror)
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
        report['targets'][target] = {
            'readout': mode, 'parameters': sum(p.numel() for p in model.parameters()),
            'reflection_max_abs': (a - b).abs().max().item(),
            'step_seconds': timings, 'finite_backward': True,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        }
        print(target, report['targets'][target], flush=True)
        del model, optimizer, base, original, mirror, prediction, loss, a, b
        torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS:', output, flush=True)


if __name__ == '__main__':
    main()
