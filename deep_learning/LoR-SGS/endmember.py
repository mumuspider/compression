import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.decomposition import NMF


ROOT = Path(__file__).resolve().parent

PAPER_DATASETS = {
    'paviau': {
        'mat_path': ROOT / 'HSI/data/PaviaU.mat',
        'mat_key': 'paviaU',
        'normalize_mode': 'h_w_c',
        'init_key': 'PaviaU',
        'postprocess': 'crop_last_340',
    },
    'urban': {
        'mat_path': ROOT / 'HSI/data/Urban_R162.mat',
        'mat_key': 'Y',
        'normalize_mode': 'c_hw_square',
        'init_key': 'Urban',
        'postprocess': None,
    },
    'indian_pines': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat'),
        'mat_key': 'indian_pines',
        'normalize_mode': 'h_w_c',
        'init_key': 'Indian_pines',
        'postprocess': None,
    },
    'pavia': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Pavia/Pavia.mat'),
        'mat_key': 'pavia',
        'normalize_mode': 'h_w_c',
        'init_key': 'Pavia',
        'postprocess': None,
    },
    'salinas': {
        'mat_path': ROOT / 'HSI/data/Salinas.mat',
        'mat_key': 'salinas',
        'normalize_mode': 'h_w_c',
        'init_key': 'Salinas',
        'postprocess': None,
    },
    'loukia': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Loukia/Loukia.mat'),
        'mat_key': 'ori_data',
        'normalize_mode': 'h_w_c',
        'init_key': 'Loukia',
        'postprocess': None,
    },
    'cuprite': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/cuprite/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat'),
        'mat_key': 'X',
        'normalize_mode': 'h_w_c',
        'init_key': 'Cuprite',
        'postprocess': None,
    },
    'brain': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/brain/04203_Calibrated_filtered.mat'),
        'mat_key': 'data',
        'normalize_mode': 'h_w_c',
        'init_key': 'Brain',
        'postprocess': None,
    },
    'longkou': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat'),
        'mat_key': 'WHU_Hi_LongKou',
        'normalize_mode': 'h_w_c',
        'init_key': 'LongKou',
        'postprocess': None,
    },
    'hanchuan': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat'),
        'mat_key': 'WHU_Hi_HanChuan',
        'normalize_mode': 'h_w_c',
        'init_key': 'HanChuan',
        'postprocess': None,
    },
    'jasperridge': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/jasperridge/jasperRidge2_R198.mat'),
        'mat_key': 'Y',
        'normalize_mode': 'c_hw_square',
        'init_key': 'JR',
        'postprocess': None,
    },
}

BENCHMARK_DATASETS = {
    'indian_pines': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat'),
        'mat_key': 'indian_pines',
        'normalize_mode': 'h_w_c',
        'init_key': 'Indian_pines',
        'postprocess': None,
    },
    'paviau': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/PaviaU/PaviaU.mat'),
        'mat_key': 'paviaU',
        'normalize_mode': 'h_w_c',
        'init_key': 'PaviaU',
        'postprocess': None,
    },
    'pavia': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Pavia/Pavia.mat'),
        'mat_key': 'pavia',
        'normalize_mode': 'h_w_c',
        'init_key': 'Pavia',
        'postprocess': None,
    },
    'salinas': {
        'mat_path': Path('/home/wenchang/WangLL/HSI-Compression-benchmark/dataset/Salina/Salinas.mat'),
        'mat_key': 'salinas',
        'normalize_mode': 'h_w_c',
        'init_key': 'Salinas',
        'postprocess': None,
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
    elif mode == 'h_w_c':
        for i in range(cube.shape[2]):
            max_val = np.max(cube[:, :, i])
            if max_val > 0:
                cube[:, :, i] /= max_val
    else:
        raise ValueError(f'Unknown normalize mode: {mode}')
    return cube


def apply_postprocess(cube, postprocess):
    if postprocess == 'crop_last_340':
        return cube[-340:, :, :]
    return cube


def get_dataset_configs(mode):
    return PAPER_DATASETS if mode == 'paper' else BENCHMARK_DATASETS


def load_dataset(name, mode):
    dataset_key = name.lower()
    dataset_configs = get_dataset_configs(mode)
    if dataset_key not in dataset_configs:
        valid = ', '.join(sorted(dataset_configs.keys()))
        raise ValueError(f'Unknown dataset for mode={mode}: {name}. Valid: {valid}')

    cfg = dataset_configs[dataset_key]
    data = scipy.io.loadmat(cfg['mat_path'])[cfg['mat_key']]
    data = normalize_hsi(data, cfg['normalize_mode'])
    data = apply_postprocess(data, cfg.get('postprocess'))
    data = np.transpose(data, (2, 0, 1)).reshape(data.shape[2], -1)
    return data, cfg['init_key']


def nmf_initialization(I, rank, init_key):
    print(f'Running NMF initialization on {init_key} with rank={rank}')
    nmf = NMF(rank, init='random', random_state=42, max_iter=12000)
    endmember = nmf.fit_transform(I).T
    abundance = nmf.components_.T

    init_dir = ROOT / 'HSI' / 'init'
    os.makedirs(init_dir, exist_ok=True)
    np.save(init_dir / f'{init_key}_endmember_rank_{rank}_NMF.npy', endmember)
    np.save(init_dir / f'{init_key}_abundance_rank_{rank}_NMF.npy', abundance)
    print(f'Saved NMF initialization results for {init_key}.')


def main():
    parser = argparse.ArgumentParser(description='NMF initialization for hyperspectral endmembers.')
    parser.add_argument('--mode', type=str, default='paper', choices=['paper', 'benchmark'])
    parser.add_argument('--dataset', type=str, required=True, help='Dataset key. Valid keys depend on --mode.')
    parser.add_argument('--rank', type=int, default=12, help='Number of endmembers (rank)')
    args = parser.parse_args()

    I, init_key = load_dataset(args.dataset, args.mode)
    nmf_initialization(I, args.rank, init_key)


if __name__ == '__main__':
    main()
