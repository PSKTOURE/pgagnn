#!/bin/bash
#SBATCH -J "md17_train"
#SBATCH --output slurm/md17/slurm.%x.%j.out
#SBATCH --error slurm/md17/slurm.%x.%j.err
#SBATCH --partition hpda_mig
#SBATCH --gres=gpu:a100_1g.10gb
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --time 23:50:00
#SBATCH --signal=USR1@1000
#SBATCH --requeue

set -euo pipefail
trap 'echo "Error on line $LINENO"; exit 1' ERR

module purge

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$PROJECT_DIR/.venv/bin/activate"

LOCAL_WORK_DIR="/dlocal/run/${SLURM_JOB_ID}"
PERSISTENT_DATA_DIR="${MD17_DATA_DIR:-$HOME/datasets/md17}"

RUN_NAME=${RUN_NAME:-all}
MODEL_NAME=${1:-${MODEL_NAME:-pgagnn}}
DATASET=${2:-${DATASET:-aspirin}}
NUM_LAYERS=${3:-${NUM_LAYERS:-10}}

DATASET="revised ${DATASET}"
DATASET_DIR_NAME="$(echo "$DATASET" | tr ' ' '_')"

use_sparse=""
attention_type="gatr"
if [[ "$MODEL_NAME" =~ ^(pgagnn|ggnn|egnn|segnn)$ ]]; then
    NUM_LAYERS=6
    use_sparse="--use_sparse"
    attention_type="gatr_sparse"
fi

REMOTE_RUN_DIR="$PROJECT_DIR/runs/md17/${RUN_NAME}"
REMOTE_MODEL_RUN_DIR="$REMOTE_RUN_DIR/${DATASET_DIR_NAME}/${MODEL_NAME}"

LOCAL_PROJECT_DIR="$LOCAL_WORK_DIR/ga-gnn"
LOCAL_RUN_DIR="$LOCAL_PROJECT_DIR/runs/md17/${RUN_NAME}"
LOCAL_MODEL_RUN_DIR="$LOCAL_RUN_DIR/${DATASET_DIR_NAME}/${MODEL_NAME}"

DATA_DIR="$PERSISTENT_DATA_DIR"

echo "JOB_ID=$SLURM_JOB_ID"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "LOCAL_WORK_DIR=$LOCAL_WORK_DIR"
echo "LOCAL_PROJECT_DIR=$LOCAL_PROJECT_DIR"
echo "PERSISTENT_DATA_DIR=$PERSISTENT_DATA_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "MODEL_NAME=$MODEL_NAME RUN_NAME=$RUN_NAME DATASET=$DATASET NUM_LAYERS=$NUM_LAYERS"

mkdir -p slurm/md17
mkdir -p "$LOCAL_PROJECT_DIR"
mkdir -p "$LOCAL_RUN_DIR"
mkdir -p "$REMOTE_RUN_DIR"
mkdir -p "$LOCAL_MODEL_RUN_DIR"
mkdir -p "$REMOTE_MODEL_RUN_DIR"
mkdir -p "$(dirname "$DATA_DIR")"

rsync -az \
	--exclude='runs/' \
	--exclude='datasets/' \
	--exclude='__pycache__/' \
	--exclude='.git/' \
	--exclude='.venv/' \
	"$PROJECT_DIR/" "$LOCAL_PROJECT_DIR/"

RESUME_ARG=""
REMOTE_LAST_CHECKPOINT="$REMOTE_MODEL_RUN_DIR/last_model.pt"
REMOTE_LAST_CHECKPOINT_BACKUP="$REMOTE_MODEL_RUN_DIR/last_model.pt.bak"

if [ -f "$REMOTE_LAST_CHECKPOINT" ]; then
	echo "Resuming from checkpoint: $REMOTE_LAST_CHECKPOINT"
	cp "$REMOTE_LAST_CHECKPOINT" "$LOCAL_MODEL_RUN_DIR/"
	RESUME_ARG="--resume_from $LOCAL_MODEL_RUN_DIR/last_model.pt"
elif [ -f "$REMOTE_LAST_CHECKPOINT_BACKUP" ]; then
	echo "Primary checkpoint missing; resuming from backup: $REMOTE_LAST_CHECKPOINT_BACKUP"
	cp "$REMOTE_LAST_CHECKPOINT_BACKUP" "$LOCAL_MODEL_RUN_DIR/last_model.pt"
	RESUME_ARG="--resume_from $LOCAL_MODEL_RUN_DIR/last_model.pt"
else
	echo "No resume checkpoint found; starting fresh."
fi

sync_back() {
	echo "Syncing results back..."
	rsync -az "$LOCAL_RUN_DIR/" "$REMOTE_RUN_DIR/" || echo "WARNING: rsync sync_back failed"
}

trap sync_back EXIT

cd "$LOCAL_PROJECT_DIR"
export PYTHONPATH="$LOCAL_PROJECT_DIR:$LOCAL_PROJECT_DIR/geometric-algebra-transformer:$LOCAL_PROJECT_DIR/egnn:$LOCAL_PROJECT_DIR/Steerable-E3-GNN/models:${PYTHONPATH:-}"

SRUN_EXIT_CODE=0

srun python -u experiments/md17/train_md17_energy_based.py \
	--model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
	--batch_size 32 \
	--epochs 3500 \
	--lr 0.001 \
    --swa_start_epoch 3000 \
	--swa_lr 0.00005 \
	--num_layers "$NUM_LAYERS" \
	--num_heads 8 \
	--edge_mvc 1 \
	--edge_sc 256 \
	--hidden_mvc 16 \
    --hidden_sc 128 \
    --cutoff 5.0 \
    --ema_decay 0.999 \
	--attention_type "$attention_type" \
	--runs_dir "$LOCAL_RUN_DIR/$DATASET_DIR_NAME" \
	--data_dir "$DATA_DIR" \
	--handle_signals TERM,USR1 \
	--no_time_suffix \
	--no_tqdm \
    $use_sparse \
	$RESUME_ARG || SRUN_EXIT_CODE=$?

if [ "$SRUN_EXIT_CODE" -eq 3 ]; then
	echo "Caught exit code 3 from Python. Time limit reached. Syncing checkpoints before requeue..."
	sync_back
	trap - EXIT
	scontrol requeue "$SLURM_JOB_ID" || echo "WARNING: scontrol requeue failed"
	exit 0
elif [ "$SRUN_EXIT_CODE" -ne 0 ]; then
	echo "Training failed with unexpected exit code: $SRUN_EXIT_CODE"
	exit "$SRUN_EXIT_CODE"
else
	echo "Training completed successfully."
	exit 0
fi
