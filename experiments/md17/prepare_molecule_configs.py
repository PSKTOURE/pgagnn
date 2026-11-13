"""Freeze molecule-specific Bessel cutoffs from the optimization split only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MOLECULES = ['azobenzene', 'paracetamol', 'naphthalene', 'salicylic acid',
             'toluene', 'uracil', 'benzene', 'ethanol', 'malonaldehyde']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--reference-config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    c = json.loads(args.reference_config.read_text())
    indices = np.loadtxt(args.data_dir / f"raw/rmd17/splits/index_train_{c['split_idx']:02d}.csv", dtype=int)
    assert len(indices) == 1000
    np.random.default_rng(c['seed']).shuffle(indices)
    train = indices[:-50]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name in MOLECULES:
        file_name = name.replace(' ', '_') if name != 'salicylic acid' else 'salicylic'
        raw = args.data_dir / f'raw/rmd17/npz_data/rmd17_{file_name}.npz'
        with np.load(raw) as data:
            pos = data['coords'][train]
            atoms = len(data['nuclear_charges'])
        distances = np.linalg.norm(pos[:, :, None] - pos[:, None, :], axis=-1)
        maximum = float(distances.max())
        cutoff = float(math.ceil(maximum))
        slug = name.replace(' ', '_')
        config = {**c, 'dataset': f'revised {name}', 'cutoff': cutoff}
        (args.output_dir / f'{slug}.json').write_text(json.dumps(config, indent=2) + '\n')
        reports[slug] = dict(dataset=config['dataset'], atoms=atoms, max_training_distance=maximum,
                             cutoff=cutoff, optimization_frames=len(train),
                             raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        print(slug, 'atoms=', atoms, 'training_diameter=', maximum, 'cutoff=', cutoff, flush=True)
    (args.output_dir / 'cutoff_selection.json').write_text(json.dumps(dict(
        rule='ceil(maximum interatomic distance across 950 optimization configurations), angstrom',
        seed=c['seed'], split=c['split_idx'], train_indices=train.tolist(),
        molecules=reports), indent=2) + '\n')


if __name__ == '__main__':
    main()
