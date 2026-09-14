#!/bin/bash
# Script to train models on MD17 dataset with different molecular dynamics targets
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/geometric-algebra-transformer:$ROOT_DIR/egnn:$ROOT_DIR/Steerable-E3-GNN/models:${PYTHONPATH:-}"


MODEL_NAME=("pgagnn" "gatrn" "egnn" "segnn")
DATASETS=(
	"revised aspirin"
	"revised azobenzene"
	"revised benzene"
	"revised ethanol"
	"revised malonaldehyde"
	"revised naphthalene"
	"revised paracetamol"
	"revised salicylic acid"
	"revised toluene"
	"revised uracil"
)


START=${1:-0}
END=${2:-10}
GPU_ID=${3:-0}

train_model() {
	local model_name=$1
	local dataset=$2
	local use_sparse=""
	local num_layers=10
	local attention_type="gatr"
	local dataset_name=$(echo "$dataset" | tr ' ' '_')
	if [[ "$model_name" =~ ^(pgagnn|ggnn|egnn|segnn)$ ]]; then
		use_sparse="--use_sparse"
		attention_type="gatr_sparse"
		num_layers=6
	fi
	echo "Training $model_name on dataset: $dataset (GPU_ID=${GPU_ID})..."
	python train_md17_energy_based.py \
		--model_name "$model_name" \
		--dataset "$dataset" \
		--attention_type "$attention_type" \
		--edge_mvc 1 \
		--edge_sc 256 \
		--batch_size 32 \
		--epochs 3500 \
		--lr 0.001 \
		--swa_start_epoch 3000 \
		--swa_lr 0.00005 \
		--num_layers "$num_layers" \
		--num_rbf 64 \
		--hidden_mvc 64 \
		--hidden_sc 128 \
		--cutoff 5.0 \
		--ema_decay 0.999 \
		--runs_dir ../../runs/md17/all/$dataset_name \
		--no_tqdm \
		$use_sparse \
		--gpu "$GPU_ID"

	echo "Training completed for $model_name on $dataset."
}

for idx in "${!DATASETS[@]}"; do
	if [[ $idx -ge $START && $idx -lt $END ]]; then
		dataset="${DATASETS[$idx]}"
		for model in "${MODEL_NAME[@]}"; do
			train_model "$model" "$dataset"
		done
	fi
done

echo "All training runs completed."
