import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from main import DEFAULT_RESULTS_ROOT, ROOT, load_dataset, save_single_exp_outputs, set_random_seed
from train_compression import train_nd


CASE_SPECS = [
    {
        "case_id": "a0",
        "case_label": "A0",
        "name": "LoR-SGS Baseline",
        "use_coef_refine": False,
        "use_endmember_calib": False,
    },
    {
        "case_id": "a1",
        "case_label": "A1",
        "name": "A0 + Coefficient Refinement",
        "use_coef_refine": True,
        "use_endmember_calib": False,
    },
    {
        "case_id": "a2",
        "case_label": "A2",
        "name": "A0 + Endmember Calibration",
        "use_coef_refine": False,
        "use_endmember_calib": True,
    },
    {
        "case_id": "a3",
        "case_label": "A3",
        "name": "A0 + Both Modules",
        "use_coef_refine": True,
        "use_endmember_calib": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="LoR-SGS+ ablation runner")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "benchmark"])
    parser.add_argument("--dataset", type=str, default="paviau")
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num_points", type=int, default=14500)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--coef_refine_kernel", type=int, default=3)
    parser.add_argument("--calib_rank", type=int, default=2)
    return parser.parse_args()


def build_case_args(base_args, case_spec, img_path):
    return SimpleNamespace(
        mode=base_args.mode,
        dataset=base_args.dataset,
        img_path=img_path,
        iterations=base_args.iterations,
        num_points=base_args.num_points,
        seed=base_args.seed,
        use_coef_refine=case_spec["use_coef_refine"],
        coef_refine_kernel=base_args.coef_refine_kernel,
        use_endmember_calib=case_spec["use_endmember_calib"],
        calib_rank=base_args.calib_rank,
        ablation_case=case_spec["case_id"],
    )


def save_summary(summary_dir: Path, summary_payload: dict):
    summary_json = summary_dir / f"{summary_dir.name}_summary.json"
    with open(summary_json, "w") as file_obj:
        json.dump(summary_payload, file_obj, indent=4, default=str)

    summary_csv = summary_dir / f"{summary_dir.name}_summary.csv"
    fieldnames = [
        "case",
        "name",
        "use_coef_refine",
        "use_endmember_calib",
        "psnr",
        "delta_psnr",
        "ssim",
        "ms_ssim",
        "sam",
        "bpppb",
        "training_time",
        "eval_time",
    ]
    with open(summary_csv, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_payload["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def resolve_ablation_root():
    env_root_value = os.environ.get("BENCHMARK_RESULTS_ROOT", "").strip()
    env_root = Path(env_root_value) if env_root_value else None
    if env_root is not None:
        return env_root / "LoR-SGS+" / "ablation"
    return DEFAULT_RESULTS_ROOT / "LoR-SGS+" / "ablation"


def main():
    args = parse_args()
    set_random_seed(args.seed)
    E, I, config = load_dataset(args.dataset, args.mode, args.img_path)
    img_path = str(config["mat_path"]) if args.img_path is None else args.img_path
    image_name = Path(img_path).stem if args.mode != "paper" else args.dataset
    data_name = "HSI" if args.mode == "paper" else "benchmark"

    print("\n=== LoR-SGS+ Ablation Configuration ===")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Image path: {img_path}")
    print(f"Rank: {config['rank']}")
    print(f"Iterations: {args.iterations}")
    print(f"Number of Gaussian Points: {args.num_points}")
    print(f"Seed: {args.seed}")
    print(f"Coefficient refinement kernel: {args.coef_refine_kernel}")
    print(f"Endmember calibration rank: {args.calib_rank}")
    print(f"Shape: {I.shape}")
    print()

    gt = torch.tensor(I)
    gt = gt.view(-1, gt.size(0), gt.size(1), gt.size(2)).permute(0, 3, 1, 2).contiguous()
    gt = torch.clamp(gt, 0, 1)

    baseline_psnr = None
    rows = []
    case_artifacts = {}

    ablation_root = resolve_ablation_root()
    summary_dir = ablation_root / (
        f"{image_name}_LoR-SGS+_ablation_summary_P{args.num_points}_it{args.iterations}_R{config['rank']}"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    for case_spec in CASE_SPECS:
        case_id = case_spec["case_id"]
        exp_name = f"{image_name}_LoR-SGS+_ablation_{case_id}_P{args.num_points}_it{args.iterations}_R{config['rank']}_S{args.seed}"
        logdir = ablation_root / exp_name
        logdir.mkdir(parents=True, exist_ok=True)

        # Reset RNG state before each case so every ablation branch is directly
        # comparable to an independently launched run with the same seed.
        set_random_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("=" * 80)
        print(f"Running {case_spec['case_label']} - {case_spec['name']}")
        print(f"  use_coef_refine={case_spec['use_coef_refine']}")
        print(f"  use_endmember_calib={case_spec['use_endmember_calib']}")
        print(f"  output={logdir}")
        print("=" * 80)

        metrics, training_logs, checkpoint_paths = train_nd(
            gt,
            endmember=E,
            iterations=args.iterations,
            num_points=args.num_points,
            model_name=f"GaussianImage_Cholesky_nd_plus_{case_id}",
            image_name=image_name,
            data_name=data_name,
            use_coef_refine=case_spec["use_coef_refine"],
            coef_refine_kernel=args.coef_refine_kernel,
            use_endmember_calib=case_spec["use_endmember_calib"],
            calib_rank=args.calib_rank,
        )
        metrics["shape"] = tuple(I.shape)

        case_args = build_case_args(args, case_spec, img_path)
        save_single_exp_outputs(logdir, exp_name, case_args, config, metrics, checkpoint_paths, training_logs)

        if baseline_psnr is None:
            baseline_psnr = metrics["psnr"]
        delta_psnr = metrics["psnr"] - baseline_psnr
        row = {
            "case": case_spec["case_label"],
            "name": case_spec["name"],
            "use_coef_refine": case_spec["use_coef_refine"],
            "use_endmember_calib": case_spec["use_endmember_calib"],
            "psnr": metrics["psnr"],
            "delta_psnr": delta_psnr,
            "ssim": metrics["ssim"],
            "ms_ssim": metrics["ms_ssim"],
            "sam": metrics["sam"],
            "bpppb": metrics["bpppb"],
            "training_time": metrics["training_time"],
            "eval_time": metrics["eval_time"],
        }
        rows.append(row)
        case_artifacts[case_spec["case_label"]] = {
            "logdir": str(logdir),
            "best_checkpoint": str(checkpoint_paths["best"]),
            "latest_checkpoint": str(checkpoint_paths["latest"]),
            "results_json": str(logdir / f"{exp_name}_results_Q8.json"),
        }

    summary_payload = {
        "dataset": args.dataset,
        "mode": args.mode,
        "img_path": img_path,
        "iterations": args.iterations,
        "num_points": args.num_points,
        "seed": args.seed,
        "coef_refine_kernel": args.coef_refine_kernel,
        "calib_rank": args.calib_rank,
        "rows": rows,
        "artifacts": case_artifacts,
    }
    save_summary(summary_dir, summary_payload)

    print("\n=== Ablation Summary ===")
    for row in rows:
        print(
            f"{row['case']}: PSNR={row['psnr']:.4f}, "
            f"ΔPSNR={row['delta_psnr']:+.4f}, "
            f"bpppb={row['bpppb']:.4f}, "
            f"coef_refine={'on' if row['use_coef_refine'] else 'off'}, "
            f"endmember_calib={'on' if row['use_endmember_calib'] else 'off'}"
        )
    print(f"\nSummary saved to: {summary_dir}")


if __name__ == "__main__":
    main()
