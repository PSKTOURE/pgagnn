#!/bin/bash
#SBATCH --job-name=md17_io_profile
#SBATCH --output=slurm/md17_io_profile-%j.out
#SBATCH --error=slurm/md17_io_profile-%j.err
#SBATCH --partition=htc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-mpl-${SLURM_JOB_ID}"
df -h /tmp
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,clocks.sm --format=csv -l 1 \
    > "slurm/md17_io_profile-${SLURM_JOB_ID}-gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
srun .venv/bin/python -u experiments/md17/profile_md17_io.py \
    --data-dir "$PWD/datasets/md17" \
    --output "$PWD/runs/md17/aspirin_io/${SLURMD_NODENAME}-${SLURM_JOB_ID}"
