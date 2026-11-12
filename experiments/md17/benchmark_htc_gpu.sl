#!/bin/bash
#SBATCH --job-name=md17_timing
#SBATCH --output=slurm/md17_timing-%j.out
#SBATCH --error=slurm/md17_timing-%j.err
#SBATCH --partition=htc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-mpl-${SLURM_JOB_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi
srun .venv/bin/python -u experiments/md17/benchmark_md17.py \
    --data-dir "$PWD/datasets/md17" \
    --output "$PWD/runs/md17/aspirin_timing/${SLURMD_NODENAME}-${SLURM_JOB_ID}" \
    --epochs 2
