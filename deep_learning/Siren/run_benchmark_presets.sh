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
      HIDDEN_FEATURES=555
      TARGET_BPPPB=0.55
      ;;
    longkou)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat"
      HIDDEN_FEATURES=315
      TARGET_BPPPB=0.10
      ;;
    hanchuan)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat"
      HIDDEN_FEATURES=100
      TARGET_BPPPB=0.06
      ;;
    urban)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/urban/Urban_R162.mat"
      HIDDEN_FEATURES=362
      TARGET_BPPPB=0.40
      ;;
    indian_pines|indiapine)
      IMG_PATH="/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat"
      HIDDEN_FEATURES=179
      TARGET_BPPPB=0.50
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
  echo "Model: Siren"
  echo "Dataset: $dataset"
  echo "Image path: $IMG_PATH"
  echo "Target bpppb: $TARGET_BPPPB"
  echo "GPU: $GPU_ID"
  echo "Seed: $SEED"
  echo "Iterations: $NUM_ITERS"
  echo "Results root: $RUN_ROOT"
  echo "Hidden layers: 1"
  echo "Hidden features: $HIDDEN_FEATURES"
  echo "============================================================"

  python main.py \
    --model_type Siren \
    --img_path "$IMG_PATH" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --num_iters "$NUM_ITERS" \
    --learning_rate 2e-4 \
    --hidden_layers 1 \
    --hidden_features "$HIDDEN_FEATURES" \
    --axis 0
}

if [[ "$DATASET" == "all" ]]; then
  for dataset in indian_pines paviau urban longkou hanchuan; do
    run_one "$dataset"
  done
else
  run_one "$DATASET"
fi
