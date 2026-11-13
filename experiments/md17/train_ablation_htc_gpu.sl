#!/bin/bash
#SBATCH --job-name=aspirin_ablation
#SBATCH --output=slurm/aspirin_ablation-%A_%a.out
#SBATCH --error=slurm/aspirin_ablation-%A_%a.err
#SBATCH --partition=htc_gpu
#SBATCH --nodelist=caegpu2
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@300

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"
STUDY_DIR="$PROJECT_DIR/runs/md17/aspirin_ablation"
VARIANTS=(no_gp no_geometric_edges no_gp_no_geometric_edges)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
cd "$STUDY_DIR/source"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-e3-${SLURM_JOB_ID}"
CONFIG="experiments/md17/configs/aspirin_ablation/$VARIANT.json"
PREFLIGHT="$STUDY_DIR/$VARIANT/preflight"
if [[ ! -f "$PREFLIGHT/validation.json" ]]; then
    srun "$PROJECT_DIR/.venv/bin/python" -u experiments/md17/validate_md17.py \
        --config "$CONFIG" --data-dir "$PROJECT_DIR/datasets/md17" --output "$PREFLIGHT"
fi
"$PROJECT_DIR/.venv/bin/python" - "$PREFLIGHT/validation.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['all_symmetry_checks_passed'], 'GPU symmetry checks failed'
assert report['repeat_training']['losses_equal'], 'GPU losses not repeatable'
assert report['repeat_training']['parameters_bitwise_equal'], 'GPU parameters not repeatable'
print('GPU force-training and symmetry preflight passed.', flush=True)
PY
RESUME=()
CHECKPOINT="$STUDY_DIR/$VARIANT/pgagnn/last_model.pt"
if [[ -f "$CHECKPOINT" ]]; then
    RESUME=(--resume_from "$CHECKPOINT")
fi
set +e
srun "$PROJECT_DIR/.venv/bin/python" -u experiments/md17/train_md17_energy_based.py \
    --config "$CONFIG" --data_dir "$PROJECT_DIR/datasets/md17" \
    --runs_dir "$STUDY_DIR/$VARIANT" --no_time_suffix --no_tqdm --no_tensorboard \
    --handle_signals TERM,USR1 "${RESUME[@]}"
TRAIN_EXIT=$?
set -e
if [[ "$TRAIN_EXIT" -eq 3 ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$TRAIN_EXIT"
