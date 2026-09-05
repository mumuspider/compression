import os
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import torch
from pytorch_msssim import SSIM

################## dataset ##################
def _load_hsi_array(file_path):
    img_data = scipy.io.loadmat(file_path)

    three_dim_arrays = [
        v for k, v in img_data.items()
        if not k.startswith('__') and isinstance(v, np.ndarray) and len(v.shape) == 3
    ]
    if three_dim_arrays:
        return three_dim_arrays[0].astype(np.float32)

    if {'Y', 'nRow', 'nCol'}.issubset(img_data.keys()):
        spectra = img_data['Y']
        height = int(np.asarray(img_data['nRow']).reshape(-1)[0])
        width = int(np.asarray(img_data['nCol']).reshape(-1)[0])
        if spectra.ndim == 2 and spectra.shape[1] == height * width:
            return spectra.reshape(spectra.shape[0], height, width).transpose(1, 2, 0).astype(np.float32)

    if 'Y' in img_data and isinstance(img_data['Y'], np.ndarray) and img_data['Y'].ndim == 2:
        spectra = img_data['Y']
        side = int(round(np.sqrt(spectra.shape[1])))
        if side * side == spectra.shape[1]:
            return spectra.reshape(spectra.shape[0], side, side).transpose(1, 2, 0).astype(np.float32)

    raise ValueError(f'Unsupported HSI .mat layout: {file_path}')


def load_hsi_data(file_path, device='cuda', dtype=torch.float32):
    """
    Load hyperspectral image data from a .mat file, normalize, and move to the specified device.

    Args:
        file_path (str): Path to the .mat data file.
        device (str): Device name ('cuda' or 'cpu').
        dtype (torch.dtype): Data type for the tensor.

    Returns:
        torch.Tensor: Hyperspectral image tensor of shape [C, H, W].
    """
    img = _load_hsi_array(file_path)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)  # Convert to [C, H, W]

    # Normalize to [0, 1]
    denom = (img.max() - img.min()).clamp_min(1e-12)
    img = (img - img.min()) / denom

    # Move to specified device and dtype
    img = img.to(device, dtype)

    return img


def resolve_output_root(output_dir=None):
    """Resolve the output root for traditional methods."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    benchmark_results_root = os.environ.get("BENCHMARK_RESULTS_ROOT", "").strip()
    if benchmark_results_root:
        return Path(benchmark_results_root).expanduser().resolve() / "traditional"
    return Path(__file__).resolve().parent / "output"


def create_task_output_dir(output_dir, method_name, hsi_path):
    """Create a timestamped task directory under the method-specific output root."""
    image_name = Path(hsi_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = resolve_output_root(output_dir) / method_name / f"{timestamp}_{image_name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


################## metrics ##################
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1.0 / mse).item()

def compute_ssim(img1, img2):
    ssim_calculator = SSIM(data_range=1.0, channel=img1.shape[0])
    return ssim_calculator(img1.unsqueeze(0), img2.unsqueeze(0)).item()

def compute_sam(img1, img2):
    # C, H, W -> H*W, C
    img1_flat = img1.reshape(img1.shape[0], -1).permute(1, 0)
    img2_flat = img2.reshape(img2.shape[0], -1).permute(1, 0)
    
    cos_sim = torch.nn.functional.cosine_similarity(img1_flat, img2_flat, dim=1)
    
    sam = torch.mean(torch.acos(cos_sim)).item() 
    
    return sam
