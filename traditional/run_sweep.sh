#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME="${BENCHMARK_RUN_NAME:-paper_runs_20260327_main}"
RUN_ROOT="${BENCHMARK_RESULTS_ROOT:-/home/wenchang/WangLL/HSI-Compression-benchmark/methods/results/${RUN_NAME}}"
export BENCHMARK_RESULTS_ROOT="$RUN_ROOT"

DATASET="${1:-all}"
METHOD="${2:-all}"
DEVICE="${3:-cpu}"

JPEG_LEVELS="${JPEG_LEVELS:-10,20,30,40,50,60,70,80,90,95}"
JPEG2K_RATIOS="${JPEG2K_RATIOS:-5,8,10,12,15,20,30,50,75,100}"
PCA_DCT_DIMS="${PCA_DCT_DIMS:-6,8,10,12,14}"
PCA_DCT_KEEP_COEFFS="${PCA_DCT_KEEP_COEFFS:-4,8}"

resolve_image_path() {
  local dataset_key="${1,,}"
  case "$dataset_key" in
    indian_pines|indiapine|indianpine)
      echo "/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat"
      ;;
    paviau)
      echo "/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/PaviaU/PaviaU.mat"
      ;;
    longkou)
      echo "/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat"
      ;;
    hanchuan)
      echo "/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat"
      ;;
    urban)
      echo "/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/urban/Urban_R162.mat"
      ;;
    *)
      echo "Unsupported dataset: $1" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local method="$2"
  local image_path
  image_path="$(resolve_image_path "$dataset")"

  echo "============================================================"
  echo "Traditional method: $method"
  echo "Dataset: $dataset"
  echo "Image path: $image_path"
  echo "Results root: $RUN_ROOT/traditional"
  echo "============================================================"

  case "$method" in
    jpeg)
      python jpeg.py --image_path "$image_path" --device "$DEVICE" --quality_levels "$JPEG_LEVELS"
      ;;
    jpeg2k)
      python jpeg2k.py --image_path "$image_path" --device "$DEVICE" --compression_ratios "$JPEG2K_RATIOS"
      ;;
    pca_dct)
      echo "PCA-DCT dims: $PCA_DCT_DIMS"
      echo "PCA-DCT keep_coeffs: $PCA_DCT_KEEP_COEFFS"
      python pca_dct.py --image_path "$image_path" --pca_dims "$PCA_DCT_DIMS" --keep_coeffs "$PCA_DCT_KEEP_COEFFS"
      ;;
    *)
      echo "Unsupported method: $method" >&2
      exit 1
      ;;
  esac
}

resolve_datasets() {
  local dataset_arg="${1,,}"
  if [[ "$dataset_arg" == "all" ]]; then
    echo "indian_pines paviau urban longkou hanchuan"
  else
    echo "$dataset_arg"
  fi
}

resolve_methods() {
  local method_arg="${1,,}"
  if [[ "$method_arg" == "all" ]]; then
    echo "jpeg jpeg2k pca_dct"
  else
    echo "$method_arg"
  fi
}

for dataset in $(resolve_datasets "$DATASET"); do
  for method in $(resolve_methods "$METHOD"); do
    run_one "$dataset" "$method"
  done
done

python interpolate_results.py --run_root "$RUN_ROOT" --dataset "$DATASET" --method "$METHOD"
