#!/bin/bash
#SBATCH -J qm9_train
#SBATCH --output=slurm/qm9/slurm.%x.%j.out
#SBATCH --error=slurm/qm9/slurm.%x.%j.err
#SBATCH --open-mode=append
#SBATCH --partition=hpda_mig
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=23:50:00
#SBATCH --signal=USR1@1000
#SBATCH --requeue

set -euo pipefail

PROJECT_DIR="${QM9_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL_NAME="${1:-pgagnn}"
RUN_NAME="${2:-qm9_e3}"
TARGET="${3:-U0}"
NUM_LAYERS="${4:-3}"
TARGETS=(mu alpha homo lumo gap r2 zpve U0 U H G Cv)
if [[ "$TARGET" == all ]]; then
    TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Submit target=all with --array=0-11}"
    [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX < ${#TARGETS[@]} )) || exit 2
    TARGET="${TARGETS[$TASK_INDEX]}"
fi
[[ " ${TARGETS[*]} " == *" $TARGET "* ]] || { echo "Unknown QM9 target: $TARGET" >&2; exit 2; }
[[ "$MODEL_NAME" == pgagnn || "$MODEL_NAME" == gatr ]] || exit 2
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid run name" >&2; exit 2; }

PYTHON="${QM9_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Python environment missing: $PYTHON; set QM9_PYTHON or create .venv." >&2; exit 1; }
DATA_DIR="${QM9_DATA_DIR:-$HOME/datasets/qm9}"
[[ -d "$DATA_DIR/processed" ]] || { echo "QM9 processed data missing: $DATA_DIR" >&2; exit 1; }
RUN_DIR="$PROJECT_DIR/runs/qm9/$RUN_NAME"
MODEL_RUN_DIR="$RUN_DIR/$TARGET/$MODEL_NAME"
SOURCE_DIR="$RUN_DIR/source"
mkdir -p "$RUN_DIR" "$MODEL_RUN_DIR"

# Freeze once per experiment, shared by array tasks and all requeues.
# An interrupted copy is never exposed as the completed snapshot.
(
    flock -x 9
    if [[ ! -d "$SOURCE_DIR" ]]; then
        SNAPSHOT_TMP=$(mktemp -d "$RUN_DIR/.source.XXXXXX")
        trap 'rm -rf "$SNAPSHOT_TMP"' EXIT
        mkdir -p "$SNAPSHOT_TMP/experiments"
        rsync -a --exclude=.git --exclude=__pycache__ "$PROJECT_DIR/src/" "$SNAPSHOT_TMP/src/"
        rsync -a --exclude=__pycache__ "$PROJECT_DIR/experiments/qm9/" "$SNAPSHOT_TMP/experiments/qm9/"
        if [[ -d "$PROJECT_DIR/geometric-algebra-transformer" ]]; then
            rsync -a --exclude=.git --exclude=.venv --exclude=__pycache__ \
                "$PROJECT_DIR/geometric-algebra-transformer/" "$SNAPSHOT_TMP/geometric-algebra-transformer/"
        fi
        cp "$PROJECT_DIR/pyproject.toml" "$PROJECT_DIR/uv.lock" "$SNAPSHOT_TMP/"
        mv "$SNAPSHOT_TMP" "$SOURCE_DIR"
        trap - EXIT
    fi
) 9>"$RUN_DIR/.source.lock"

LOCAL_WORK_DIR="${QM9_SCRATCH_ROOT:-/dlocal/run}/${SLURM_JOB_ID:?}/${SLURM_ARRAY_TASK_ID:-single}"
LOCAL_SOURCE="$LOCAL_WORK_DIR/source"
LOCAL_DATA="$LOCAL_WORK_DIR/qm9"
mkdir -p "$LOCAL_SOURCE" "$LOCAL_DATA"
rsync -a --delete "$SOURCE_DIR/" "$LOCAL_SOURCE/"
rsync -a "$DATA_DIR/" "$LOCAL_DATA/"

module purge
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LOCAL_SOURCE:$LOCAL_SOURCE/geometric-algebra-transformer"
cd "$LOCAL_SOURCE"

ATTENTION=gatr
EXTRA_ARGS=()
if [[ "$MODEL_NAME" == pgagnn ]]; then
    ATTENTION=gatr_sparse
    EXTRA_ARGS+=(--use_sparse)
fi
if [[ -s "$MODEL_RUN_DIR/last_model.pt" ]]; then
    EXTRA_ARGS+=(--resume_from "$MODEL_RUN_DIR/last_model.pt")
elif [[ -s "$MODEL_RUN_DIR/last_model.pt.bak" ]]; then
    # Keep the checkpoint in the model's directory so resolve_run_directory
    # continues writing to the same run after a backup recovery.
    EXTRA_ARGS+=(--resume_from "$MODEL_RUN_DIR/last_model.pt.bak")
fi

echo "Job=$SLURM_JOB_ID task=${SLURM_ARRAY_TASK_ID:-single} restart=${SLURM_RESTART_COUNT:-0}"
echo "Model=$MODEL_NAME target=$TARGET source=$SOURCE_DIR results=$MODEL_RUN_DIR"
echo "Python=$PYTHON visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

# Checkpoints and best weights go directly to persistent storage. Code and data
# use node-local scratch; no final rsync is needed to make a saved epoch durable.
STATUS=0
srun "$PYTHON" -u experiments/qm9/train_qm9.py \
    --model_name "$MODEL_NAME" --target "$TARGET" \
    --batch_size "${QM9_BATCH_SIZE:-128}" --epochs "${QM9_EPOCHS:-1000}" \
    --lr 0.0005 --weight_decay 5e-7 --warmup_epochs 15 --ema_decay 0.999 \
    --seed 42 --num_layers "$NUM_LAYERS" --num_heads 8 \
    --in_sc 64 --in_mvc 1 --hidden_mvc 16 --hidden_sc 128 \
    --out_mvc 16 --out_sc 32 --edge_mvc 1 --edge_sc 128 \
    --dropout 0.1 --cutoff 5.0 --attention_type "$ATTENTION" \
    --num_workers "${QM9_NUM_WORKERS:-4}" --checkpoint_interval 1 \
    --runs_dir "$RUN_DIR" --data_dir "$LOCAL_DATA" \
    --handle_signals TERM,USR1 --no_time_suffix --no_tqdm \
    "${EXTRA_ARGS[@]}" || STATUS=$?

if [[ "$STATUS" == 3 ]]; then
    [[ -s "$MODEL_RUN_DIR/last_model.pt" ]] || { echo "No saved checkpoint; refusing requeue." >&2; exit 1; }
    JOB_TO_REQUEUE="$SLURM_JOB_ID"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        JOB_TO_REQUEUE="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    echo "Completed-epoch checkpoint saved; requeueing $JOB_TO_REQUEUE"
    scontrol requeue "$JOB_TO_REQUEUE"
    exit 0
fi
exit "$STATUS"
