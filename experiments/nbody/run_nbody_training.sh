#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/geometric-algebra-transformer:$ROOT_DIR/egnn:$ROOT_DIR/Steerable-E3-GNN/models:${PYTHONPATH:-}"

START_RUN=${1:-0}
END_RUN=${2:-3}
GPU_ID=${3:-0}
SPRING=${4:-false}

SAMPLE_SIZES=(0.001 0.005 0.01 0.05 0.1 0.5 1.0)
BATCH_SIZES=(64 64 64 64 64 64 64)

LR=(
	0.0003
	0.0003
	0.0003
	0.0003
	0.0003
	0.0003
	0.0003
)

FINAL_LR=(
	0.000003
	0.000003
	0.000003
	0.000003
	0.000003
	0.000003
	0.000003
)

# Models to train
MODELS=(
	pgagnn
	# segnn
	# egnn
	# gatr
	# ggnn
)

# Model-specific settings
declare -Ar NUM_LAYERS=(
	[gatr]=10
	[segnn]=4
	[egnn]=4
	[pgagnn]=2
	[ggnn]=3
)

train_model() {
	local model_name=$1
	local seed=$(( $2 + 42 ))
	local idx=$3

	local sample_size=${SAMPLE_SIZES[$idx]}
	local batch_size=${BATCH_SIZES[$idx]}
	local lr=${LR[$idx]}
	local final_lr=${FINAL_LR[$idx]}

	local num_layers=${NUM_LAYERS[$model_name]:-3}
	local attention_type="gatr"
	local hidden_mvc=16
	local sparse_flag=""
	local dir="nbody"
	local seed_folder=$2
	local reg_scale=0.01

	if [[ "$SPRING" == true ]] &&
		[[ "$model_name" =~ ^(pgagnn|egnn|segnn|ggnn)$ ]]; then
		sparse_flag="--use_sparse"
		attention_type="gatr_sparse"
		dir="nbody_spring"
	fi

	if [[ "$model_name" =~ ^(pgagnn|ggnn)$ ]]; then
		hidden_mvc=64
		reg_scale=0.0
	fi

	local run_dir="../../runs/${dir}/${seed_folder}"
	mkdir -p "$run_dir"

	args=(
		--model_name "$model_name"
		--batch_size "$batch_size"
		--num_steps 50000
		--eval_every 500
		--early_stopping_patience 10
		--in_mvc 1
		--learning_rate "$lr"
		--lr_final "$final_lr"
		--num_layers "$num_layers"
		--num_heads 8
		--hidden_mvc "$hidden_mvc"
		--hidden_sc 128
		--scheduler_type cosine_warmup
		--reg_scale "$reg_scale"
		--dropout 0.1
		--activation gelu
		--attention_type "$attention_type"
		--subsample "$sample_size"
		--use_edge_attr
		--seed "$seed"
		--message geometric
		--no_tqdm
		--runs_dir "$run_dir"
		--gpu "$GPU_ID"
	)

	[[ -n "$sparse_flag" ]] && args+=(--use_sparse)
	[[ "$SPRING" == "true" ]] && args+=(--use_spring)

	python train_nbody.py "${args[@]}" 2>&1

}

run_model() {
	local seed=$1
	local model_name=$2
	echo "========== ${model_name^^} =========="
	for idx in "${!SAMPLE_SIZES[@]}"; do
		train_model "$model_name" "$seed" "$idx"
	done
}

for seed in $(seq "$START_RUN" $((END_RUN - 1))); do
	echo "# Starting Seed $seed"
	for model_name in "${MODELS[@]}"; do
		run_model "$seed" "$model_name"
	done
done

echo "========================================"
echo "All training runs completed!"
echo "========================================"
