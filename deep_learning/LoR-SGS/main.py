import argparse
import json
import os
import re
import random
from pathlib import Path
import numpy as np
import scipy.io
import torch
from train_compression import train_nd
ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = Path('/home/wenchang/WangLL/HSI-Compression-benchmark/methods/results/paper_runs_20260327_main')
PAPER_DATASETS = {
    "paviau": {
        "mat_path": ROOT / 'HSI/data/PaviaU.mat',
        "mat_key": 'paviaU',
        "rank": 12,
        "init_key": 'PaviaU',
        "normalize_mode": 'h_w_c',
        "postprocess": 'crop_last_340',
    },
    "urban": {
        "mat_path": ROOT / 'HSI/data/Urban_R162.mat',
        "mat_key": 'Y',
        "rank": 12,
        "init_key": 'Urban',
        "normalize_mode": 'c_hw_square',
        "postprocess": None,
    },
    "indian_pines": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat'),
        "mat_key": 'indian_pines',
        "rank": 12,
        "init_key": 'Indian_pines',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "pavia": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Pavia/Pavia.mat'),
        "mat_key": 'pavia',
        "rank": 12,
        "init_key": 'Pavia',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "salinas": {
        "mat_path": ROOT / 'HSI/data/Salinas.mat',
        "mat_key": 'salinas',
        "rank": 12,
        "init_key": 'Salinas',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "loukia": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Loukia/Loukia.mat'),
        "mat_key": 'ori_data',
        "rank": 12,
        "init_key": 'Loukia',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "cuprite": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/cuprite/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat'),
        "mat_key": 'X',
        "rank": 12,
        "init_key": 'Cuprite',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "brain": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/brain/04203_Calibrated_filtered.mat'),
        "mat_key": 'data',
        "rank": 12,
        "init_key": 'Brain',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "longkou": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat'),
        "mat_key": 'WHU_Hi_LongKou',
        "rank": 12,
        "init_key": 'LongKou',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "hanchuan": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat'),
        "mat_key": 'WHU_Hi_HanChuan',
        "rank": 12,
        "init_key": 'HanChuan',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "jasperridge": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/jasperridge/jasperRidge2_R198.mat'),
        "mat_key": 'Y',
        "rank": 10,
        "init_key": 'JR',
        "normalize_mode": 'c_hw_square',
        "postprocess": None,
    },
}
BENCHMARK_DATASETS = {
    "indian_pines": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat'),
        "mat_key": 'indian_pines',
        "rank": 12,
        "init_key": 'Indian_pines',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "paviau": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/PaviaU/PaviaU.mat'),
        "mat_key": 'paviaU',
        "rank": 12,
        "init_key": 'PaviaU',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "pavia": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Pavia/Pavia.mat'),
        "mat_key": 'pavia',
        "rank": 12,
        "init_key": 'Pavia',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
    "salinas": {
        "mat_path": Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Salina/Salinas.mat'),
        "mat_key": 'salinas',
        "rank": 12,
        "init_key": 'Salinas',
        "normalize_mode": 'h_w_c',
        "postprocess": None,
    },
}
def normalize_hsi(cube, mode):
    cube = cube.astype(float)
    cube = np.clip(cube, 0, None)
    if mode == 'c_hw_square':
        for i in range(cube.shape[0]):
            max_val = np.max(cube[i, :])
            if max_val > 0:
                cube[i, :] /= max_val
        side = int(np.sqrt(cube.shape[1]))
        cube = cube.reshape(cube.shape[0], side, side).transpose(1, 2, 0)
        return cube
    if mode == 'h_w_c':
        for i in range(cube.shape[2]):
            max_val = np.max(cube[:, :, i])
            if max_val > 0:
                cube[:, :, i] /= max_val
        return cube
    raise ValueError(f'Unknown normalize mode: {mode}')
def apply_postprocess(cube, postprocess):
    if postprocess == 'crop_last_340':
        return cube[-340:, :, :]
    return cube
def get_dataset_configs(mode):
    return PAPER_DATASETS if mode == 'paper' else BENCHMARK_DATASETS
def resolve_endmember_path(init_key: str, rank: int):
    init_dir = ROOT / 'HSI' / 'init'
    candidates = [
        init_dir / f'{init_key}_endmember_rank_{rank}.npy',
        init_dir / f'{init_key.lower()}_endmember_rank_{rank}.npy',
        init_dir / f'{init_key}_endmember_rank_{rank}_NMF.npy',
        init_dir / f'{init_key.lower()}_endmember_rank_{rank}_NMF.npy',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, rank
    fallback_matches = []
    for prefix in (init_key, init_key.lower()):
        fallback_matches.extend(sorted(init_dir.glob(f'{prefix}_endmember_rank_*.npy')))
    if fallback_matches:
        unique_matches = []
        seen = set()
        for match in fallback_matches:
            resolved = match.resolve()
            if resolved not in seen:
                unique_matches.append(match)
                seen.add(resolved)
        if len(unique_matches) == 1:
            match = unique_matches[0]
            rank_match = re.search(r'_rank_(\d+)', match.name)
            resolved_rank = int(rank_match.group(1)) if rank_match else rank
            print(
                f"[warn] Missing {init_key} rank={rank} initialization, "
                f"falling back to {match.name} (rank={resolved_rank})."
            )
            return match, resolved_rank
    raise FileNotFoundError(
        'Missing endmember initialization. Expected one of: ' + ', '.join(str(path) for path in candidates)
    )
def load_dataset(name, mode, img_path_override=None):
    dataset_key = name.lower()
    dataset_configs = get_dataset_configs(mode)
    if dataset_key not in dataset_configs:
        valid = ', '.join(sorted(dataset_configs.keys()))
        raise ValueError(f'Unknown dataset name for mode={mode}: {name}. Valid: {valid}')
    config = dict(dataset_configs[dataset_key])
    mat_path = Path(img_path_override) if img_path_override else config['mat_path']
    if not mat_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {mat_path}')
    endmember_path, resolved_rank = resolve_endmember_path(config['init_key'], config['rank'])
    E = np.load(endmember_path).astype(np.float32)
    config['rank'] = resolved_rank
    I = scipy.io.loadmat(mat_path)[config['mat_key']]
    I = normalize_hsi(I, config['normalize_mode'])
    I = apply_postprocess(I, config.get('postprocess'))
    config['mat_path'] = mat_path
    return E, I, config


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(args, config):
    results_root = os.environ.get("BENCHMARK_RESULTS_ROOT", "").strip()
    results_subdir = os.environ.get("BENCHMARK_RESULTS_SUBDIR", "").strip()
    if args.mode == 'paper':
        image_name = args.dataset
        exp_name = f"{image_name}_LoR-SGS_paper_P{args.num_points}_it{args.iterations}_R{config['rank']}_S{args.seed}"
    else:
        image_name = Path(args.img_path).stem if args.img_path else args.dataset
        exp_name = f"{image_name}_LoR-SGS_P{args.num_points}_it{args.iterations}_R{config['rank']}_S{args.seed}"
    if results_root:
        logdir = Path(results_root) / 'LoR-SGS' / 'single_exp' / exp_name
    elif results_subdir:
        logdir = ROOT / 'results' / results_subdir / 'single_exp' / exp_name
    else:
        logdir = DEFAULT_RESULTS_ROOT / 'LoR-SGS' / 'single_exp' / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    return logdir, exp_name, image_name
def ensure_symlink(src: Path, dst: Path):
    src = src.resolve()
    if not src.exists():
        return
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    os.symlink(src, dst)
def save_single_exp_outputs(logdir, exp_name, args, config, metrics, checkpoint_paths, training_logs):
    best_ckpt = checkpoint_paths['best']
    latest_ckpt = checkpoint_paths['latest']
    best_artifact = logdir / f'{exp_name}_best.pth.tar'
    latest_artifact = logdir / f'{exp_name}_latest.pth.tar'
    ensure_symlink(best_ckpt, best_artifact)
    ensure_symlink(latest_ckpt, latest_artifact)
    checkpoint_size_bytes = best_ckpt.stat().st_size if best_ckpt.exists() else 0
    img_numel = metrics['shape'][0] * metrics['shape'][1] * metrics['shape'][2]
    results = {
        'results': {
            'fp': None,
            'quantized': {
                'quant_method': 'BuiltInQuant',
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'ms_ssim': metrics['ms_ssim'],
                'sam': metrics['sam'],
                'bpppb': metrics['bpppb'],
                'bit_components': metrics.get('bit_components'),
                'best_bpppb': metrics.get('best_bpppb'),
                'best_bit_components': metrics.get('best_bit_components'),
                'pth_bpppb': (checkpoint_size_bytes * 8 / img_numel) if checkpoint_size_bytes else None,
                'compression_ratio': (os.path.getsize(args.img_path) / checkpoint_size_bytes) if checkpoint_size_bytes and args.img_path else None,
                'model_size_kb': checkpoint_size_bytes / 1024 if checkpoint_size_bytes else None,
                'encode_time': metrics['training_time'],
                'decode_time': metrics['eval_time'],
                'eval_fps': metrics['eval_fps'],
                'peak_gpu_mem_mib': metrics.get('peak_gpu_mem_mib'),
            },
        },
        'args': vars(args),
        'mode': args.mode,
        'dataset': {
            **config,
            'mat_path': str(config['mat_path']),
        },
        'artifacts': {
            'best_checkpoint': str(best_ckpt),
            'latest_checkpoint': str(latest_ckpt),
        },
    }
    with open(logdir / f'{exp_name}_results_Q8.json', 'w') as file_obj:
        json.dump(results, file_obj, indent=4, default=str)
    with open(logdir / f'{exp_name}_training_logs.json', 'w') as file_obj:
        json.dump(training_logs, file_obj, indent=4)
    curve_src = checkpoint_paths.get('curve')
    curve_dst = logdir / f'{exp_name}_training_curves.png'
    if curve_src is not None:
        ensure_symlink(curve_src, curve_dst)
    train_txt_src = checkpoint_paths['train_txt']
    train_txt_dst = logdir / f'{exp_name}_train.txt'
    ensure_symlink(train_txt_src, train_txt_dst)
def main():
    parser = argparse.ArgumentParser(description='LoR-SGS Hyperspectral Image Compression')
    parser.add_argument('--mode', type=str, default='paper', choices=['paper', 'benchmark'])
    parser.add_argument('--dataset', type=str, default='paviau',
                        help='Dataset key. Valid keys depend on --mode.')
    parser.add_argument('--img_path', type=str, default=None,
                        help='Optional explicit .mat path. In benchmark mode this overrides the default benchmark path for the dataset key.')
    parser.add_argument('--iterations', type=int, default=8000,
                        help='Number of training iterations')
    parser.add_argument('--num_points', type=int, default=600,
                        help='Number of Gaussian points')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducible experiments.')
    args = parser.parse_args()
    set_random_seed(args.seed)
    E, I, config = load_dataset(args.dataset, args.mode, args.img_path)
    if args.img_path is None:
        args.img_path = str(config['mat_path'])
    logdir, exp_name, image_name = setup_logging(args, config)
    print(f"\n=== Training Configuration ===")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Image path: {args.img_path}")
    print(f"Rank: {config['rank']}")
    print(f"Iterations: {args.iterations}")
    print(f"Number of Gaussian Points: {args.num_points}")
    print(f"Seed: {args.seed}")
    print(f"Shape: {I.shape}")
    print(f"Output dir: {logdir}")
    print()
    GT = torch.tensor(I)
    GT = GT.view(-1, GT.size(0), GT.size(1), GT.size(2)).permute(0, 3, 1, 2).contiguous()
    GT = torch.clamp(GT, 0, 1)
    data_name = 'HSI' if args.mode == 'paper' else 'benchmark'
    metrics, training_logs, checkpoint_paths = train_nd(
        GT,
        endmember=E,
        iterations=args.iterations,
        num_points=args.num_points,
        model_name='GaussianImage_Cholesky_nd',
        image_name=image_name,
        data_name=data_name,
    )
    metrics['shape'] = tuple(I.shape)
    save_single_exp_outputs(logdir, exp_name, args, config, metrics, checkpoint_paths, training_logs)
if __name__ == '__main__':
    main()
