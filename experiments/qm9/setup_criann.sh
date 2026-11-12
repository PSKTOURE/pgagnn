#!/bin/bash
# Reuse CRIANN's shared PyTorch/CUDA stack; install only a small QM9 overlay.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
QM9_MODULE=aidl/pytorch/2.6.0-py3.12-cuda12.6
if ! type module >/dev/null 2>&1; then
    # Noninteractive SSH sessions do not initialize CRIANN's Lmod function.
    set +u
    source /opt/cray/pe/lmod/lmod/init/bash
    set -u
fi
module use /soft/Modules/RH9_modules/aidl
module purge
module load "$QM9_MODULE"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1
ulimit -c 0
UV="${UV:-$HOME/.local/bin/uv}"
"$UV" venv --allow-existing --system-site-packages --python /soft/AIDL/conda_envs/pytorch2602/bin/python "$PROJECT_DIR/.venv"
printf '%s\n' "$QM9_MODULE" > "$PROJECT_DIR/.venv/criann-module"
"$UV" pip install --python "$PROJECT_DIR/.venv/bin/python" --no-cache --no-deps \
    --only-binary :all: -r "$PROJECT_DIR/experiments/qm9/requirements-criann.txt"
"$PROJECT_DIR/.venv/bin/python" -c 'import sys,importlib.metadata as m; print(sys.version); print({n:m.version(n) for n in ["torch","numpy","scipy","torch-geometric","torch-scatter","torch-cluster","einops","tensorboard"]})'
echo "Environment ready: $PROJECT_DIR/.venv (shared module: $QM9_MODULE)"
