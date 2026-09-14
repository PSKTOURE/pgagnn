# Projective Geometric Algebra Graph Neural Networks (PGA-GNN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)](tests/)

Official PyTorch implementation for the paper:  
**"Projective Geometric Algebra Graph Neural Networks for Equivariant Physical Modeling"**.

---

## Overview

**PGA-GNN** is an exact $E(3)$-equivariant graph neural network built on 3D **Projective Geometric Algebra** (PGA, $\mathbb{R}_{3,0,1}$). By representing geometry in a 16-dimensional graded multivector space, PGA-GNN provides a unified algebraic framework for:
- Points, rigid motions (rotations and translations), velocities, forces, and oriented lines/planes.
- **Factorized Equivariant Linear Layers** with linear channel scaling.
- **Dual-Norm & Geometric Attention Mechanisms** capturing invariant distances and angular features without metric collapse.
- Exact $E(3)$ rotational, reflectional, and translational equivariance.

---

## Repository Structure

```
├── src/                         # Core PGA-GNN library
│   ├── __init__.py              # Public package exports
│   ├── pgagnn.py                # PGA-GNN architecture and message passing layers
│   ├── ggnn.py                  # GGNN vector/scalar transformer baseline
│   ├── layers.py                # Equivariant linear, norm, gating, and basis layers
│   ├── attention.py             # Geometric, dual-norm, similarity, and GATr attention
│   ├── attn_utils.py            # Attention tensor normalization & helper functions
│   ├── primitives.py            # PGA geometric/outer/inner products, duals, embeddings
│   ├── checkpointing.py         # Atomic checkpointing & preemption handlers
│   ├── utils.py                 # Seeds, basis loading, and algebra masks
│   └── data/                    # Precomputed PGA algebra multiplication tables
│       ├── geometric_product.pt
│       └── outer_product.pt
├── experiments/                 # Benchmark experiment suites
│   ├── nbody/                   # N-body gravitational & spring systems
│   │   ├── train_nbody.py       # N-body training and evaluation script
│   │   ├── dataset.py           # Dense and sparse N-body dataset loaders
│   │   ├── simulator.py         # Physics simulator for dataset generation
│   │   ├── run_nbody_training.sh# Multi-seed training runner script
│   │   ├── run_nbody_ablation.sh# Component & layer ablation runner
│   │   ├── plot_model_comparison.py # Model comparison loss plotting
│   │   └── ablation_plot.py     # Ablation analysis plotting
│   ├── md17/                    # MD17 & revised MD17 (rMD17) molecular dynamics
│   │   ├── train_md17_energy_based.py # Energy/force prediction training script
│   │   ├── md17_data.py         # rMD17 dataset loader with fallback mirrors
│   │   ├── run_md17_training.sh # MD17 training runner across all molecules
│   │   └── train_md17.sl        # SLURM cluster execution template
│   └── qm9/                     # QM9 molecular property prediction
│       ├── train_qm9.py         # QM9 training script
│       ├── run_qm9_training.sh  # QM9 runner across target properties
│       └── train_qm9.sl         # SLURM cluster execution template
├── tests/                       # Automated unit & equivariance test suite
│   ├── test_equivariance.py     # E(3) rotation, reflection, translation checks
│   ├── test_pga_gnn_equivariance.py # E(3) full pipeline equivariance checks
│   ├── test_primitives.py       # PGA algebraic identities & embeddings
│   ├── test_layers.py           # Equivariant linear, norm, and gating layers
│   ├── test_models.py           # End-to-end model forward & backward tests
│   └── test_attention.py        # Multi-head attention mechanisms
├── scripts/                     # Environment setup and helper utilities
│   └── setup_submodules.py      # Links baseline submodules to Python environment
├── flops_counter.py             # Model computational complexity & FLOPs benchmark
├── collect_data.py              # Results aggregation script
├── pyproject.toml               # Package dependencies & metadata
├── requirements.txt             # Pip dependencies
└── LICENSE                      # MIT License
```

---

## Installation

### 1. Clone the Repository (with submodules)

```bash
git clone --recursive <REPO_URL>
cd ga-gnn
```

> **Note**: If you cloned without `--recursive`, initialize submodules with:
> ```bash
> git submodule update --init --recursive
> ```

### 2. Set Up Virtual Environment

#### Option 1: Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) provides fast, deterministic environment resolution:

```bash
# Create and sync virtual environment
uv sync
source .venv/bin/activate

# Link baseline submodules (GATr, EGNN, SEGNN)
python scripts/setup_submodules.py
```

#### Option 2: Using standard `pip`

```bash
# Create and activate Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Link baseline submodules (GATr, EGNN, SEGNN)
python scripts/setup_submodules.py
```

---

## Quick Verification

### 1. Run Automated Test Suite
Run the full test suite :
```bash
pytest -v tests\
```

### 2. Run FLOPs Benchmark
Compare the inference FLOPs across models (PGA-GNN dense, PGA-GNN sparse, GATr, GGNN, EGNN, SEGNN):
```bash
python flops_counter.py
```

---

## Reproducing Paper Experiments

### 1. N-Body Physical Systems

#### Generate Datasets
Simulate charged / spring N-body physical trajectories:
```bash
# Generate standard gravitational / charged trajectories
python experiments/nbody/simulator.py
```

#### Train Models
Train across sample sizes ($0.1\%$ to $100\%$ data regimes):
```bash
# Train on standard N-body system
bash experiments/nbody/run_nbody_training.sh 0 3 0 false

# Train on spring-coupled N-body system (sparse mode)
bash experiments/nbody/run_nbody_training.sh 0 3 0 true
```

#### Run Ablation Studies
```bash
# Architectural component ablations (geometric product, edge attributes, scalar streams)
bash experiments/nbody/run_nbody_ablation.sh 0 3 0 false components

# Layer depth scaling ablations
bash experiments/nbody/run_nbody_ablation.sh 0 3 0 false layers
```

#### Plot Results
```bash
python experiments/nbody/plot_model_comparison.py
python experiments/nbody/ablation_plot.py
```

---

### 2. MD17 / Revised MD17 (rMD17) Molecular Dynamics

Train energy-conserving force and energy prediction models on revised MD17 molecules (aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic acid, toluene, uracil):

```bash
# Train PGA-GNN on a specific molecule (e.g., revised aspirin)
python experiments/md17/train_md17_energy_based.py \
    --model_name pgagnn \
    --dataset "revised aspirin" \
    --batch_size 32 \
    --epochs 3500 \
    --lr 0.001 \
    --swa_start_epoch 3000 \
    --swa_lr 0.00005 \
    --num_layers 6 \
    --hidden_mvc 64 \
    --hidden_sc 128 \
    --cutoff 5.0 \
    --use_sparse \
    --gpu 0

# Run all 10 rMD17 molecules sequentially
bash experiments/md17/run_md17_training.sh 0 10 0
```

---

### 3. QM9 Molecular Property Prediction

Train models to predict quantum chemical properties ($U_0$, $\mu$, $\alpha$, $\text{HOMO}$, $\text{LUMO}$, etc.):

```bash
# Train on a single target property (e.g., U0 internal energy at 0K)
python experiments/qm9/train_qm9.py \
    --model_name pgagnn \
    --target U0 \
    --batch_size 128 \
    --epochs 1000 \
    --lr 0.0005 \
    --num_layers 6 \
    --in_sc 64 \
    --hidden_mvc 64 \
    --hidden_sc 128 \
    --cutoff 5.0 \
    --use_sparse \
    --gpu 0

# Run via training script
bash experiments/qm9/run_qm9_training.sh pgagnn U0 0
```

---

### 4. Results Aggregation

Collect and aggregate metrics across completed runs into a clean CSV summary:
```bash
python collect_data.py --data_dir runs/md17/all --output collected_md17_results.csv
```

---

## Python API Quickstart

PGA-GNN can be used directly as a modular PyTorch library:

```python
import torch
from src import PGA_GNN, embed_point, embed_scalar, embed_translation, extract_point

# 1. Instantiate PGA-GNN
model = PGA_GNN(
    in_mvc=1,        # Input multivector channels
    out_mvc=1,       # Output multivector channels
    hidden_mvc=32,   # Hidden multivector width
    in_sc=2,         # Input scalar channels
    out_sc=2,        # Output scalar channels
    hidden_sc=64,    # Hidden scalar width
    num_layers=3,    # Number of message passing layers
    num_heads=4,     # Attention heads
    attention_type="gatr",
)

# 2. Prepare physical geometric inputs
batch_size, num_nodes = 4, 10
positions = torch.randn(batch_size, num_nodes, 3)     # 3D points
velocities = torch.randn(batch_size, num_nodes, 3)    # 3D translation vectors
masses = torch.rand(batch_size, num_nodes, 1) + 0.1   # Invariant scalars

# 3. Embed into 16-D PGA multivectors
pos_mv = embed_point(positions)
vel_mv = embed_translation(velocities)
mass_mv = embed_scalar(masses)
input_mv = (pos_mv + vel_mv + mass_mv).unsqueeze(2)   # (B, N, 1, 16)
scalars = masses                                      # (B, N, 1)
adj = torch.ones(batch_size, num_nodes, num_nodes)

# 4. Forward pass (strictly E(3)-equivariant)
out_mv, out_sc = model(mv=input_mv, sc=scalars, adj=adj)

# 5. Extract predicted 3D coordinates
pred_positions = extract_point(out_mv[:, :, 0, :])     # (B, N, 3)
```

---

## Citation

```bibtex
@article{pgagnn2026,
  title   = {Projective Geometric Algebra Graph Neural Networks for Equivariant Physical Modeling},
  author  = {Anonymous Authors},
  journal = {Under Review},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
