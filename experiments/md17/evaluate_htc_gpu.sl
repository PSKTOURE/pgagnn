#!/bin/bash
#SBATCH --job-name=aspirin_test
#SBATCH --output=slurm/aspirin_test-%j.out
#SBATCH --error=slurm/aspirin_test-%j.err
#SBATCH --partition=htc_gpu
#SBATCH --nodelist=caegpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:45:00
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-eval-${SLURM_JOB_ID}"
srun "$PROJECT_DIR/.venv/bin/python" -u "$PROJECT_DIR/experiments/md17/evaluate_saved_md17.py" \
    --study-dir "$PROJECT_DIR/runs/md17/aspirin_radial" \
    --data-dir "$PROJECT_DIR/datasets/md17" \
    --output "$PROJECT_DIR/runs/md17/aspirin_radial/test_evaluation.json"
