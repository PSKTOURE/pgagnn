#!/bin/bash
# Usage: bash experiments/qm9/submit_criann.sh [run_name] [all|target] [max_parallel]
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
RUN_NAME="${1:-qm9_e3}"
TARGET="${2:-all}"
MAX_PARALLEL="${3:-4}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid run name" >&2; exit 2; }
[[ " all mu alpha homo lumo gap r2 zpve U0 U H G Cv " == *" $TARGET "* ]] || { echo "Unknown target: $TARGET" >&2; exit 2; }
[[ -x "${QM9_PYTHON:-$PROJECT_DIR/.venv/bin/python}" ]] || { echo "Set QM9_PYTHON or create the project .venv before submission." >&2; exit 1; }
[[ -d "${QM9_DATA_DIR:-$HOME/datasets/qm9}/processed" ]] || { echo "Set QM9_DATA_DIR to the QM9 dataset before submission." >&2; exit 1; }
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "max_parallel must be positive" >&2; exit 2; }
mkdir -p "$PROJECT_DIR/slurm/qm9"
cd "$PROJECT_DIR"
OPTIONS=()
if [[ "$TARGET" == all ]]; then
    OPTIONS+=(--array="0-11%$MAX_PARALLEL")
fi
sbatch "${OPTIONS[@]}" experiments/qm9/train_qm9.sl pgagnn "$RUN_NAME" "$TARGET" 3
