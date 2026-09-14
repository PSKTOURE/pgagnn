#!/bin/bash

# Script to train models on QM9 dataset with different targets

set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/geometric-algebra-transformer:$ROOT_DIR/egnn:$ROOT_DIR/Steerable-E3-GNN/models:${PYTHONPATH:-}"

MODEL_NAME=${1:-"pgagnn"}
TARGET=${2:-"U0"}
GPU_ID=${3:-0}

attention_type="gatr"
sparse_flag=""
num_layers=10
if [ "$MODEL_NAME" == "pgagnn" ]; then
	sparse_flag="--use_sparse"
	attention_type="gatr_sparse"
	num_layers=3
fi

echo "Training $MODEL_NAME for target: $TARGET..."
python train_qm9.py \
	--model_name "$MODEL_NAME" \
	--target "$TARGET" \
	--batch_size 128 \
	--epochs 1000 \
	--lr 0.0005 \
	--num_layers "$num_layers" \
	--in_sc 64 \
	--in_mvc 1 \
	--hidden_mvc 16 \
	--hidden_sc 128 \
	--cutoff 5.0 \
	--runs_dir ../../runs/qm9/qm9_run_02 \
	--attention_type "$attention_type" \
	--no_time_suffix \
	--no_tqdm \
	$sparse_flag \
	--gpu "$GPU_ID"

echo "Finished training $MODEL_NAME for target: $TARGET."


