import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.fftpack import dct, idct
from scipy.ndimage import uniform_filter


def resolve_output_root(output_dir=None):
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    benchmark_results_root = os.environ.get("BENCHMARK_RESULTS_ROOT", "").strip()
    if benchmark_results_root:
        return Path(benchmark_results_root).expanduser().resolve() / "traditional"
    return Path(__file__).resolve().parent / "output"


def create_task_output_dir(output_dir, method_name, hsi_path):
    image_name = Path(hsi_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = resolve_output_root(output_dir) / method_name / f"{timestamp}_{image_name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def load_hsi_cube(file_path):
    """Load H x W x C hyperspectral data from a .mat file and normalize it to [0, 1]."""
    img_data = sio.loadmat(file_path)

    three_dim_arrays = [
        value for key, value in img_data.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3
    ]
    if three_dim_arrays:
        cube = three_dim_arrays[0].astype(np.float32)
    elif {"Y", "nRow", "nCol"}.issubset(img_data.keys()):
        spectra = img_data["Y"].astype(np.float32)
        height = int(np.asarray(img_data["nRow"]).reshape(-1)[0])
        width = int(np.asarray(img_data["nCol"]).reshape(-1)[0])
        cube = spectra.reshape(spectra.shape[0], height, width).transpose(1, 2, 0)
    else:
        raise ValueError(f"Unsupported .mat layout: {file_path}")

    cube = np.clip(cube, 0.0, None)
    value_range = float(cube.max() - cube.min())
    if value_range <= 0:
        return np.zeros_like(cube, dtype=np.float32)
    cube = (cube - cube.min()) / value_range
    return cube.astype(np.float32)


def compute_psnr(reference, reconstructed):
    mse = np.mean((reference - reconstructed) ** 2, dtype=np.float64)
    mse = max(mse, 1e-12)
    return float(10.0 * np.log10(1.0 / mse))


def compute_sam(reference, reconstructed):
    ref = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    rec = reconstructed.reshape(-1, reconstructed.shape[-1]).astype(np.float64)
    numerator = np.sum(ref * rec, axis=1)
    denominator = np.linalg.norm(ref, axis=1) * np.linalg.norm(rec, axis=1)
    denominator = np.clip(denominator, 1e-12, None)
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return float(np.mean(np.arccos(cosine)))

def compute_ssim_band(reference_band, reconstructed_band):
    reference_band = reference_band.astype(np.float64)
    reconstructed_band = reconstructed_band.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = uniform_filter(reference_band, size=7)
    mu_y = uniform_filter(reconstructed_band, size=7)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = uniform_filter(reference_band * reference_band, size=7) - mu_x_sq
    sigma_y_sq = uniform_filter(reconstructed_band * reconstructed_band, size=7) - mu_y_sq
    sigma_xy = uniform_filter(reference_band * reconstructed_band, size=7) - mu_xy
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    denominator = np.clip(denominator, 1e-12, None)
    return float(np.mean(numerator / denominator))


def compute_ssim(reference, reconstructed):
    scores = [compute_ssim_band(reference[:, :, idx], reconstructed[:, :, idx]) for idx in range(reference.shape[-1])]
    return float(np.mean(scores))


def zigzag_indices(n):
    order = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                j = s - i
                if 0 <= j < n:
                    order.append((i, j))
        else:
            for j in range(min(s, n - 1), max(-1, s - n), -1):
                i = s - j
                if 0 <= i < n:
                    order.append((i, j))
    return order


def apply_2d_dct(block):
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def apply_2d_idct(block):
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def blockwise_dct_compress(component_image, block_size=8, keep_coeffs=16):
    """Apply block-wise DCT and keep the first k zigzag coefficients in each block."""
    height, width = component_image.shape
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size
    padded = np.pad(component_image, ((0, pad_h), (0, pad_w)), mode="constant")

    reconstructed = np.zeros_like(padded, dtype=np.float32)
    zigzag = zigzag_indices(block_size)
    keep_coeffs = min(keep_coeffs, block_size * block_size)
    num_blocks = 0

    for row in range(0, padded.shape[0], block_size):
        for col in range(0, padded.shape[1], block_size):
            block = padded[row:row + block_size, col:col + block_size]
            coeff = apply_2d_dct(block)
            coeff_kept = np.zeros_like(coeff)
            for idx in range(keep_coeffs):
                r, c = zigzag[idx]
                coeff_kept[r, c] = coeff[r, c]
            reconstructed[row:row + block_size, col:col + block_size] = apply_2d_idct(coeff_kept)
            num_blocks += 1

    return reconstructed[:height, :width].astype(np.float32), num_blocks


def pca_fit_transform(pixels, pca_dim):
    mean = pixels.mean(axis=0, keepdims=True)
    centered = pixels - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:pca_dim].T.astype(np.float32)
    scores = centered @ components
    return mean.astype(np.float32), components, scores.astype(np.float32)


def pca_inverse_transform(scores, mean, components):
    return scores @ components.T + mean


def estimate_compressed_size_bytes(height, width, channels, pca_dim, keep_coeffs, num_blocks, coeff_dtype=np.float32):
    coeff_itemsize = np.dtype(coeff_dtype).itemsize
    retained_coeffs = pca_dim * num_blocks * keep_coeffs
    pca_overhead = channels * pca_dim + channels  # PCA basis + mean vector
    total_values = retained_coeffs + pca_overhead
    return int(total_values * coeff_itemsize)


def run_pca_dct(cube, pca_dim, keep_coeffs, block_size=8):
    height, width, channels = cube.shape
    pixels = cube.reshape(-1, channels)

    encode_start = time.time()
    mean, components, scores = pca_fit_transform(pixels, pca_dim)
    score_cube = scores.reshape(height, width, pca_dim)

    reconstructed_scores = np.zeros_like(score_cube, dtype=np.float32)
    total_blocks = 0
    for comp_idx in range(pca_dim):
        reconstructed_component, num_blocks = blockwise_dct_compress(
            score_cube[:, :, comp_idx],
            block_size=block_size,
            keep_coeffs=keep_coeffs,
        )
        reconstructed_scores[:, :, comp_idx] = reconstructed_component
        total_blocks = num_blocks
    encode_time = time.time() - encode_start

    decode_start = time.time()
    reconstructed_pixels = pca_inverse_transform(
        reconstructed_scores.reshape(-1, pca_dim),
        mean,
        components,
    )
    reconstructed_cube = reconstructed_pixels.reshape(height, width, channels)
    reconstructed_cube = np.clip(reconstructed_cube, 0.0, 1.0).astype(np.float32)
    decode_time = time.time() - decode_start

    compressed_size_bytes = estimate_compressed_size_bytes(
        height=height,
        width=width,
        channels=channels,
        pca_dim=pca_dim,
        keep_coeffs=keep_coeffs,
        num_blocks=total_blocks,
    )
    original_size_bytes = cube.size * cube.dtype.itemsize
    compression_ratio = float(original_size_bytes / max(compressed_size_bytes, 1))
    bpppb_est = float((compressed_size_bytes * 8) / cube.size)

    metrics = {
        "psnr": compute_psnr(cube, reconstructed_cube),
        "ssim": compute_ssim(cube, reconstructed_cube),
        "sam": compute_sam(cube, reconstructed_cube),
        "compression_ratio": compression_ratio,
        "bpppb_est": bpppb_est,
        "encode_time": float(encode_time),
        "decode_time": float(decode_time),
        "compressed_size_kb": compressed_size_bytes / 1024.0,
        "original_size_kb": original_size_bytes / 1024.0,
    }
    return reconstructed_cube, metrics


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="PCA-DCT hyperspectral image compression baseline")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the .mat hyperspectral image")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory root")
    parser.add_argument("--pca_dims", type=str, default="10,15,20", help="Comma-separated PCA dimensions")
    parser.add_argument("--keep_coeffs", type=str, default="4,8,16,32", help="Comma-separated kept DCT coefficients")
    parser.add_argument("--block_size", type=int, default=8, help="Block size for DCT")
    parser.add_argument("--save_mat", action="store_true", help="Save reconstructed cubes as .mat files")
    args = parser.parse_args()

    cube = load_hsi_cube(args.image_path)
    image_name = Path(args.image_path).stem
    task_dir = create_task_output_dir(args.output_dir, "pca_dct", args.image_path)

    pca_dims = parse_int_list(args.pca_dims)
    keep_coeffs_list = parse_int_list(args.keep_coeffs)
    results = []

    for pca_dim in pca_dims:
        for keep_coeffs in keep_coeffs_list:
            if pca_dim <= 0 or pca_dim > cube.shape[-1]:
                raise ValueError(f"Invalid PCA dimension {pca_dim} for cube with {cube.shape[-1]} bands.")
            if keep_coeffs <= 0 or keep_coeffs > args.block_size * args.block_size:
                raise ValueError(f"Invalid keep_coeffs {keep_coeffs} for block size {args.block_size}.")

            print("=" * 80)
            print(f"PCA-DCT | image={image_name} | pca_dim={pca_dim} | keep_coeffs={keep_coeffs}")
            reconstructed_cube, metrics = run_pca_dct(
                cube,
                pca_dim=pca_dim,
                keep_coeffs=keep_coeffs,
                block_size=args.block_size,
            )
            result = {
                "pca_dim": pca_dim,
                "keep_coeffs": keep_coeffs,
                **metrics,
            }
            print(
                f"PSNR={metrics['psnr']:.4f} dB, "
                f"SAM={metrics['sam']:.4f}, "
                f"CR={metrics['compression_ratio']:.4f}, "
                f"bpppb_est={metrics['bpppb_est']:.4f}"
            )
            results.append(result)

            if args.save_mat:
                rec_path = task_dir / f"{image_name}_pcadct_d{pca_dim}_k{keep_coeffs}.mat"
                sio.savemat(rec_path, {"data": reconstructed_cube})

    result_columns = {
        "pca_dim": [row["pca_dim"] for row in results],
        "keep_coeffs": [row["keep_coeffs"] for row in results],
        "bpp": [row["bpppb_est"] for row in results],
        "psnr": [row["psnr"] for row in results],
        "ssim": [row["ssim"] for row in results],
        "sam": [row["sam"] for row in results],
        "compression_ratio": [row["compression_ratio"] for row in results],
        "model_size_kb": [row["compressed_size_kb"] for row in results],
        "encode_time": [row["encode_time"] for row in results],
        "decode_time": [row["decode_time"] for row in results],
    }
    results_json = {
        "image_name": image_name,
        "image_path": str(Path(args.image_path).resolve()),
        "shape": list(cube.shape),
        "block_size": args.block_size,
        "results": result_columns,
        "raw_results": results,
    }
    output_json = task_dir / f"{image_name}_pca_dct_results.json"
    with open(output_json, "w") as file_obj:
        json.dump(results_json, file_obj, indent=2)

    print("=" * 80)
    print(f"Saved PCA-DCT results to: {output_json}")


if __name__ == "__main__":
    main()
