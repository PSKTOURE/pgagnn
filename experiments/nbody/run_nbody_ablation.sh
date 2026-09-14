#!/bin/bash

# Script to run ordered N-body ablations for GA-GNN.

set -euo pipefail

# Navigate to the correct directory.
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/geometric-algebra-transformer:$ROOT_DIR/egnn:$ROOT_DIR/Steerable-E3-GNN/models:${PYTHONPATH:-}"

START_RUN=${1:-0}
END_RUN=${2:-3}
GPU_ID=${3:-0}
SPRING=${4:-false}
ABLATION_MODE=${5:-components}
LAYER_BASELINE_INDEX=${6:-0}

SAMPLE_SIZES=(0.001 0.01 0.1)
BATCH_SIZES=(64 64 64)
LR=(0.0003 0.0003 0.0003)
FINAL_LR=(0.000003 0.000003 0.000003)

ABLATION_NAMES=(
	"baseline"
	"geometric_product_no_edge_attr"
	"no_geometric_product_with_edge_attr"
	"no_geometric_product_no_edge_attr"
	"no_scalar"
)
ABLATION_MESSAGES=(
	"geometric"
	"geometric"
	"linear"
	"linear"
	"geometric"
)
ABLATION_NO_SCALAR=(
	false
	false
	false
	false
	true
)
ABLATION_EDGE_ATTR=(
	true
	false
	true
	false
	true
)
LAYERS=(1 3 4 5 6)

declare -Ar NUM_LAYERS=(
	[gatr]=10
	[segnn]=4
	[egnn]=4
	[pgagnn]=2
)

if [ "${#SAMPLE_SIZES[@]}" -ne "${#BATCH_SIZES[@]}" ]; then
	echo "Error: SAMPLE_SIZES and BATCH_SIZES must have the same length."
	exit 1
fi

if [[ "$ABLATION_MODE" != "components" && "$ABLATION_MODE" != "layers" ]]; then
	echo "Error: ABLATION_MODE must be 'components' or 'layers'."
	echo "Usage: $0 [START_RUN] [END_RUN] [GPU_ID] [SPRING] [ABLATION_MODE] [LAYER_BASELINE_INDEX]"
	exit 1
fi

if [ "$LAYER_BASELINE_INDEX" -lt 0 ] || [ "$LAYER_BASELINE_INDEX" -ge "${#ABLATION_NAMES[@]}" ]; then
	echo "Error: LAYER_BASELINE_INDEX must be between 0 and $(( ${#ABLATION_NAMES[@]} - 1 ))."
	exit 1
fi

train_model() {
	local model_name=$1
	local run_seed=$(( $2 + 42 ))
	local sample_index=$3
	local ablation_index=$4
	local num_layers=${5:-${NUM_LAYERS[$model_name]:-2}}
	local attention_type="gatr"
	local sparse_flag=""
	local no_scalar_flag=""
	local edge_attr_flag=""

	local ablation_name=${ABLATION_NAMES[$ablation_index]}
	local message=${ABLATION_MESSAGES[$ablation_index]}

	if [ "$ABLATION_MODE" = "layers" ]; then
		ablation_name="layers_${num_layers}"
	fi

	if [[ "$SPRING" = true ]] && [[ "$model_name" =~ ^(pgagnn|egnn|segnn|ggnn)$ ]]; then
		sparse_flag="--use_sparse"
		attention_type="gatr_sparse"
	fi

	if [ "${ABLATION_NO_SCALAR[$ablation_index]}" = true ]; then
		no_scalar_flag="--no_scalar"
	fi
    
	if [ "${ABLATION_EDGE_ATTR[$ablation_index]}" = true ]  || [ "$ABLATION_MODE" = "layers" ]; then
		edge_attr_flag="--use_edge_attr"
	fi

	local run_dir="../../runs/nbody/ablations/$ablation_name/$run_seed"
	mkdir -p "$run_dir"

	python train_nbody.py \
		--model_name "$model_name" \
		--batch_size "${BATCH_SIZES[$sample_index]}" \
		--num_steps 50000 \
		--eval_every 300 \
		--early_stopping_patience 10 \
		--learning_rate "${LR[$sample_index]}" \
		--lr_final "${FINAL_LR[$sample_index]}" \
		--num_layers "$num_layers" \
		--num_heads 8 \
		--hidden_mvc 64 \
		--hidden_sc 128 \
		--scheduler_type cosine_warmup \
		--reg_scale 0.0 \
		--dropout 0.1 \
		--activation gelu \
		--no_tqdm \
		--attention_type "$attention_type" \
		--subsample "${SAMPLE_SIZES[$sample_index]}" \
		--seed "$run_seed" \
		--message "$message" \
		--runs_dir "$run_dir" \
		--gpu "$GPU_ID" \
		$no_scalar_flag \
		$sparse_flag \
		$edge_attr_flag
}

run_models() {
	local run_seed=$1
	local model_name=$2

	echo "========== Training ${model_name^^} ablations (${ABLATION_MODE}) =========="

	if [ "$ABLATION_MODE" = "components" ]; then
		for ablation_index in "${!ABLATION_NAMES[@]}"; do
			echo "---- Ablation: ${ABLATION_NAMES[$ablation_index]} ----"
			for sample_index in "${!SAMPLE_SIZES[@]}"; do
				train_model "$model_name" "$run_seed" "$sample_index" "$ablation_index"
			done
		done
	else
		echo "---- Layer baseline config: ${ABLATION_NAMES[$LAYER_BASELINE_INDEX]} ----"
		for num_layers in "${LAYERS[@]}"; do
			echo "---- Num Layers: ${num_layers} ----"
			for sample_index in "${!SAMPLE_SIZES[@]}"; do
				train_model "$model_name" "$run_seed" "$sample_index" "$LAYER_BASELINE_INDEX" "$num_layers"
			done
		done
	fi
}

for n in $(seq "$START_RUN" $((END_RUN - 1))); do
	echo "========================================"
	echo "Starting training run $n"
	echo "========================================"

	for model_name in pgagnn; do
		run_models "$n" "$model_name"
	done
done

echo "========================================"
echo "All training runs completed!"
echo "========================================"
