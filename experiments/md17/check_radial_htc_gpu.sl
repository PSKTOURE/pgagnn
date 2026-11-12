#!/bin/bash
#SBATCH --job-name=aspirin_radial_check
#SBATCH --output=slurm/aspirin_radial_check-%A_%a.out
#SBATCH --error=slurm/aspirin_radial_check-%A_%a.err
#SBATCH --partition=htc_gpu
#SBATCH --nodelist=caegpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00
#SBATCH --array=0-3%4

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
STUDY_DIR="$PROJECT_DIR/runs/md17/aspirin_radial"
EXPERIMENTS=(gaussian_c9 gaussian_c11 bessel_c9 bessel_c11)
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
cd "$STUDY_DIR/source"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-mpl-${SLURM_JOB_ID}"
srun "$PROJECT_DIR/.venv/bin/python" -u experiments/md17/validate_md17.py \
    --config "experiments/md17/configs/aspirin_radial/$EXPERIMENT.json" \
    --data-dir "$PROJECT_DIR/datasets/md17" \
    --output "$STUDY_DIR/$EXPERIMENT/preflight"
