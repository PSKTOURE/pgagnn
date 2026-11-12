#!/bin/bash
#SBATCH --job-name=aspirin_radial
#SBATCH --output=slurm/aspirin_radial-%A_%a.out
#SBATCH --error=slurm/aspirin_radial-%A_%a.err
#SBATCH --partition=htc_gpu
#SBATCH --nodelist=caegpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3%4
#SBATCH --requeue
#SBATCH --signal=USR1@300

# Submit from ~/pgagnn after freezing source into the study directory.
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
STUDY_DIR="$PROJECT_DIR/runs/md17/aspirin_radial"
EXPERIMENTS=(gaussian_c9 gaussian_c11 bessel_c9 bessel_c11)
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
cd "$STUDY_DIR/source"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-mpl-${SLURM_JOB_ID}"
nvidia-smi
echo "Experiment=$EXPERIMENT Job=$SLURM_JOB_ID Source=$PWD"
RESUME=()
CHECKPOINT="$STUDY_DIR/$EXPERIMENT/pgagnn/last_model.pt"
if [[ -f "$CHECKPOINT" ]]; then
    RESUME=(--resume_from "$CHECKPOINT")
fi
set +e
srun "$PROJECT_DIR/.venv/bin/python" -u experiments/md17/train_md17_energy_based.py \
    --config "experiments/md17/configs/aspirin_radial/$EXPERIMENT.json" \
    --data_dir "$PROJECT_DIR/datasets/md17" \
    --runs_dir "$STUDY_DIR/$EXPERIMENT" \
    --no_time_suffix --no_tqdm --no_tensorboard --handle_signals TERM,USR1 \
    "${RESUME[@]}"
TRAIN_EXIT=$?
set -e
if [[ "$TRAIN_EXIT" -eq 3 ]]; then
    echo "Checkpoint saved; requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$TRAIN_EXIT"
