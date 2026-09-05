import argparse
import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path("/home/wenchang/WangLL/HSI-Compression-benchmark/methods/results/paper_runs_20260327_main")

DATASET_CONFIGS = {
    "indian_pines": {
        "image_stem": "Indian_pines",
        "target_bpp": 0.50,
        "aliases": {"indian_pines", "indiapine", "indianpine"},
    },
    "paviau": {
        "image_stem": "PaviaU",
        "target_bpp": 0.55,
        "aliases": {"paviau"},
    },
    "longkou": {
        "image_stem": "WHU_Hi_LongKou",
        "target_bpp": 0.10,
        "aliases": {"longkou"},
    },
    "hanchuan": {
        "image_stem": "WHU_Hi_HanChuan",
        "target_bpp": 0.40,
        "aliases": {"hanchuan", "hanchuan"},
    },
    "urban": {
        "image_stem": "Urban_R162",
        "target_bpp": 0.40,
        "aliases": {"urban"},
    },
}

METHODS = ("jpeg", "jpeg2k", "pca_dct")
METRIC_KEYS = (
    "psnr",
    "ssim",
    "sam",
    "compression_ratio",
    "model_size_kb",
    "encode_time",
    "decode_time",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Interpolate traditional codec results to target bpppb.")
    parser.add_argument("--run_root", type=str, default=None, help="Benchmark run root. Defaults to BENCHMARK_RESULTS_ROOT or the current paper run root.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset alias or 'all'.")
    parser.add_argument("--method", type=str, default="all", choices=("all", *METHODS), help="Traditional method or 'all'.")
    parser.add_argument("--target_bpp", type=float, default=None, help="Optional manual target bpppb override for single-dataset runs.")
    return parser.parse_args()


def resolve_run_root(run_root_arg):
    if run_root_arg:
        return Path(run_root_arg).expanduser().resolve()
    env_root = os.environ.get("BENCHMARK_RESULTS_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DEFAULT_RUN_ROOT


def normalize_dataset_key(dataset_name):
    key = dataset_name.lower()
    for canonical_name, cfg in DATASET_CONFIGS.items():
        if key == canonical_name or key in cfg["aliases"]:
            return canonical_name
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def resolve_dataset_list(dataset_arg):
    if dataset_arg.lower() == "all":
        return list(DATASET_CONFIGS.keys())
    return [normalize_dataset_key(dataset_arg)]


def resolve_method_list(method_arg):
    if method_arg == "all":
        return list(METHODS)
    return [method_arg]


def find_latest_results_json(run_root, method, image_stem):
    method_root = run_root / "traditional" / method
    if not method_root.exists():
        raise FileNotFoundError(f"Traditional method output directory not found: {method_root}")
    candidates = sorted(method_root.glob(f"*_{image_stem}/{image_stem}_{method}_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No result JSON found for method={method}, image={image_stem} under {method_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def sort_results_by_bpp(results):
    rows = []
    if isinstance(results, list):
        for idx, item in enumerate(results):
            row = {
                "bpp": float(item.get("bpp", item.get("bpppb_est"))),
                "index": idx,
            }
            for key, value in item.items():
                row[key] = float(value) if isinstance(value, (int, float)) else value
            rows.append(row)
    else:
        for idx, bpp in enumerate(results["bpp"]):
            row = {"bpp": float(bpp), "index": idx}
            for key, values in results.items():
                if idx < len(values):
                    value = values[idx]
                    row[key] = float(value) if isinstance(value, (int, float)) else value
            rows.append(row)
    rows.sort(key=lambda item: item["bpp"])
    deduped = []
    for row in rows:
        if deduped and math.isclose(row["bpp"], deduped[-1]["bpp"], rel_tol=0.0, abs_tol=1e-12):
            deduped[-1] = row
        else:
            deduped.append(row)
    return deduped


def interpolate_rows(sorted_rows, target_bpp):
    if not sorted_rows:
        raise ValueError("No rows available for interpolation.")
    if target_bpp <= sorted_rows[0]["bpp"]:
        return sorted_rows[0], sorted_rows[0], 0.0, "clamp_low"
    if target_bpp >= sorted_rows[-1]["bpp"]:
        return sorted_rows[-1], sorted_rows[-1], 0.0, "clamp_high"
    for low_row, high_row in zip(sorted_rows[:-1], sorted_rows[1:]):
        low_bpp = low_row["bpp"]
        high_bpp = high_row["bpp"]
        if low_bpp <= target_bpp <= high_bpp:
            if math.isclose(low_bpp, high_bpp, rel_tol=0.0, abs_tol=1e-12):
                return low_row, high_row, 0.0, "duplicate_bpp"
            alpha = (target_bpp - low_bpp) / (high_bpp - low_bpp)
            return low_row, high_row, alpha, "linear"
    return sorted_rows[-1], sorted_rows[-1], 0.0, "fallback_last"


def interpolate_value(low_value, high_value, alpha):
    return float(low_value + (high_value - low_value) * alpha)


def build_interpolated_payload(dataset_key, method, target_bpp, result_json):
    with open(result_json, "r") as file_obj:
        payload = json.load(file_obj)
    sorted_rows = sort_results_by_bpp(payload["results"])
    low_row, high_row, alpha, mode = interpolate_rows(sorted_rows, target_bpp)
    interpolated = {
        "dataset": dataset_key,
        "image_name": payload["image_name"],
        "method": method,
        "target_bpp": float(target_bpp),
        "mode": mode,
        "source_json": str(result_json),
        "num_points": len(sorted_rows),
        "low_point": low_row,
        "high_point": high_row,
        "alpha": float(alpha),
        "metrics": {
            "bpp": float(target_bpp if mode == "linear" else low_row["bpp"]),
        },
    }
    for key in METRIC_KEYS:
        low_value = low_row.get(key)
        high_value = high_row.get(key)
        if low_value is None or high_value is None:
            continue
        if mode == "linear":
            interpolated["metrics"][key] = interpolate_value(low_value, high_value, alpha)
        else:
            interpolated["metrics"][key] = float(low_value)
    if isinstance(payload["results"], dict):
        control_keys = [key for key in payload["results"].keys() if key not in {"bpp", *METRIC_KEYS}]
    else:
        control_keys = [key for key in low_row.keys() if key not in {"bpp", "index", *METRIC_KEYS}]
    interpolated["control_keys"] = control_keys
    interpolated["control_values"] = {
        key: {"low": low_row.get(key), "high": high_row.get(key)}
        for key in control_keys
    }
    return interpolated


def main():
    args = parse_args()
    run_root = resolve_run_root(args.run_root)
    datasets = resolve_dataset_list(args.dataset)
    methods = resolve_method_list(args.method)
    interpolated_root = run_root / "traditional" / "interpolated"
    per_case_root = interpolated_root / "per_case"
    per_case_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for dataset_key in datasets:
        dataset_cfg = DATASET_CONFIGS[dataset_key]
        target_bpp = float(args.target_bpp if args.target_bpp is not None and len(datasets) == 1 else dataset_cfg["target_bpp"])
        image_stem = dataset_cfg["image_stem"]
        for method in methods:
            result_json = find_latest_results_json(run_root, method, image_stem)
            interpolated = build_interpolated_payload(dataset_key, method, target_bpp, result_json)
            output_json = per_case_root / f"{dataset_key}_{method}_target_{target_bpp:.2f}.json"
            with open(output_json, "w") as file_obj:
                json.dump(interpolated, file_obj, indent=2)
            summary_rows.append({
                "dataset": dataset_key,
                "method": method,
                "target_bpp": interpolated["target_bpp"],
                "actual_bpp": interpolated["metrics"]["bpp"],
                "psnr": interpolated["metrics"].get("psnr"),
                "ssim": interpolated["metrics"].get("ssim"),
                "sam": interpolated["metrics"].get("sam"),
                "compression_ratio": interpolated["metrics"].get("compression_ratio"),
                "model_size_kb": interpolated["metrics"].get("model_size_kb"),
                "encode_time": interpolated["metrics"].get("encode_time"),
                "decode_time": interpolated["metrics"].get("decode_time"),
                "mode": interpolated["mode"],
                "source_json": interpolated["source_json"],
            })

    summary_json = interpolated_root / "traditional_interpolated_summary.json"
    with open(summary_json, "w") as file_obj:
        json.dump(summary_rows, file_obj, indent=2)

    summary_csv = interpolated_root / "traditional_interpolated_summary.csv"
    fieldnames = [
        "dataset", "method", "target_bpp", "actual_bpp", "psnr", "ssim", "sam",
        "compression_ratio", "model_size_kb", "encode_time", "decode_time", "mode", "source_json",
    ]
    with open(summary_csv, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Interpolated summary saved to: {summary_json}")
    print(f"Interpolated CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
