#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME="${BENCHMARK_RUN_NAME:-paper_runs_20260327_main}"
RUN_ROOT="${BENCHMARK_RESULTS_ROOT:-/home/wenchang/WangLL/HSI-Compression-benchmark/methods/results/${RUN_NAME}}"
export BENCHMARK_RESULTS_ROOT="$RUN_ROOT"

DATASET="${1:-paviau}"
MODE="${2:-paper}"
NUM_POINTS="${3:-14500}"
ITERATIONS="${4:-10000}"
# SEED="${5:-5}"

conda run -n LineR python ablation.py \
  --mode "$MODE" \
  --dataset "$DATASET" \
  --num_points "$NUM_POINTS" \
  --iterations "$ITERATIONS" \
  --seed 5
