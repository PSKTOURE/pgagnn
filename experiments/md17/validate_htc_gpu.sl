#!/bin/bash
#SBATCH --job-name=md17_gpu_check
#SBATCH --output=slurm/md17_gpu_check-%j.out
#SBATCH --error=slurm/md17_gpu_check-%j.err
#SBATCH --partition=htc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00

# Submit from the project root after creating slurm/ and installing .venv.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-mpl-${SLURM_JOB_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

nvidia-smi
srun .venv/bin/python -u experiments/md17/validate_md17.py \
    --data-dir "$PWD/datasets/md17" \
    --reference-dir "$PWD/runs/md17/aspirin_baseline/2026-11-11" \
    --output "$PWD/runs/md17/aspirin_baseline/${SLURMD_NODENAME}-${SLURM_JOB_ID}" "$@"
