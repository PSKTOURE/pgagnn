#!/bin/bash
#SBATCH --job-name=md17_bessel_e3
#SBATCH --output=slurm/md17_bessel_e3-%A_%a.out
#SBATCH --error=slurm/md17_bessel_e3-%A_%a.err
#SBATCH --partition=htc_gpu
#SBATCH --nodelist=caegpu
#SBATCH --array=0-8%4
#SBATCH --open-mode=append
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
STUDY_DIR="$PROJECT_DIR/runs/md17/molecules_e3"
MOLECULES=(azobenzene paracetamol naphthalene salicylic_acid toluene uracil benzene ethanol malonaldehyde)
MOLECULE="${MOLECULES[$SLURM_ARRAY_TASK_ID]}"
cd "$STUDY_DIR/source"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pgagnn-e3-${SLURM_JOB_ID}"
CONFIG="experiments/md17/configs/molecules_e3/$MOLECULE.json"
PREFLIGHT="$STUDY_DIR/$MOLECULE/preflight"
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
CHECKPOINT="$STUDY_DIR/$MOLECULE/pgagnn/last_model.pt"
if [[ -f "$CHECKPOINT" ]]; then
    RESUME=(--resume_from "$CHECKPOINT")
fi
set +e
srun "$PROJECT_DIR/.venv/bin/python" -u experiments/md17/train_md17_energy_based.py \
    --config "$CONFIG" --data_dir "$PROJECT_DIR/datasets/md17" \
    --runs_dir "$STUDY_DIR/$MOLECULE" --no_time_suffix --no_tqdm --no_tensorboard \
    --handle_signals TERM,USR1 "${RESUME[@]}"
TRAIN_EXIT=$?
set -e
if [[ "$TRAIN_EXIT" -eq 3 ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$TRAIN_EXIT"
