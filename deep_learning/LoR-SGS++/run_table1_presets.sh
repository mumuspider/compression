#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME="${BENCHMARK_RUN_NAME:-paper_runs_20260327_main}"
RUN_ROOT="${BENCHMARK_RESULTS_ROOT:-/home/wenchang/WangLL/HSI-Compression-benchmark/methods/results/${RUN_NAME}}"
export BENCHMARK_RESULTS_ROOT="$RUN_ROOT"

DATASET="${1:-paviau}"
ITERATIONS="${2:-10000}"
ENV_NAME="${3:-LineR}"
SEED="${4:-5}"

resolve_dataset() {
  local dataset="$1"
  local dataset_key="${dataset,,}"
  case "$dataset_key" in
    paviau)
      RANK=12
      NUM_POINTS=14500
      CODEBOOK_SIZE=64
      ;;
    indian_pines|indiapine)
      RANK=12
      NUM_POINTS=5265
      CODEBOOK_SIZE=64
      ;;
    longkou)
      RANK=12
      NUM_POINTS=14500
      CODEBOOK_SIZE=64
      ;;
    hanchuan)
      RANK=12
      NUM_POINTS=46300
      CODEBOOK_SIZE=64
      ;;
    urban)
      RANK=12
      NUM_POINTS=14500
      CODEBOOK_SIZE=64
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
  echo "Model: LoR-SGS+"
  echo "Dataset: $dataset"
  echo "Rank(k): $RANK"
  echo "Num points (N): $NUM_POINTS"
  echo "Codebook size (B, metadata): $CODEBOOK_SIZE"
  echo "Iterations: $ITERATIONS"
  echo "Seed: $SEED"
  echo "Results root: $RUN_ROOT"
  echo "Assumption: NMF initialization already exists in HSI/init"
  echo "============================================================"

  conda run -n "$ENV_NAME" python main.py --mode paper --dataset "$dataset" --num_points "$NUM_POINTS" --iterations "$ITERATIONS" --seed "$SEED"
}

if [[ "$DATASET" == "all" ]]; then
  for dataset in indian_pines paviau urban longkou hanchuan; do
    run_one "$dataset"
  done
else
  run_one "$DATASET"
fi
