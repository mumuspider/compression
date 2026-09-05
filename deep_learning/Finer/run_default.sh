#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME="${BENCHMARK_RUN_NAME:-paper_runs_20260327_main}"
RUN_ROOT="${BENCHMARK_RESULTS_ROOT:-/home/wenchang/WangLL/HSI-Compression-benchmark/methods/results/${RUN_NAME}}"
export BENCHMARK_RESULTS_ROOT="$RUN_ROOT"

DATASET="${1:-paviau}"
GPU_ID="${2:-0}"
SEED="${3:-1}"
NUM_ITERS="${4:-10000}"

resolve_dataset() {
  local dataset="$1"
  case "$dataset" in
    paviau)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/PaviaU/PaviaU.mat"
      ;;
    urban)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/urban/Urban_R162.mat"
      ;;
    longkou)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat"
      ;;
    indian_pines|indiapine)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local dataset="$1"
  resolve_dataset "$dataset"
  echo "============================================================"
  echo "Model: Finer"
  echo "Dataset: $dataset"
  echo "Image path: $IMG_PATH"
  echo "GPU: $GPU_ID"
  echo "Seed: $SEED"
  echo "Iterations: $NUM_ITERS"
  echo "Results root: $RUN_ROOT"
  echo "============================================================"

  python main.py \
    --model_type Finer \
    --img_path "$IMG_PATH" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --num_iters "$NUM_ITERS" \
    --learning_rate 2e-4 \
    --hidden_layers 1 \
    --hidden_features 100 \
    --axis 0
}

if [[ "$DATASET" == "all" ]]; then
  for dataset in indian_pines paviau urban longkou; do
    run_one "$dataset"
  done
else
  run_one "$DATASET"
fi
