# LoR-SGS+: Lightweight Plug-in Enhancement for LoR-SGS

## Summary

`LoR-SGS+` is a lightweight upgrade of `LoR-SGS` for hyperspectral image compression.

It keeps the original Gaussian splatting backbone and low-rank spectral formulation, and only adds two small plug-in modules:

- `Coefficient Refinement Block`
- `Low-rank Endmember Calibration`

The design goal is to improve reconstruction quality with minimal structural changes and limited extra runtime.

## Method

The data flow is:

```text
Gaussian Splatting
    ->
Coefficient Map
    ->
Coefficient Refinement Block
    ->
Refined Coefficient Map
    ->
Low-rank Endmember Calibration
    ->
HSI Reconstruction
```

### 1. Coefficient Refinement Block

After Gaussian rasterization, the coefficient map is reshaped to `[1, rank, H, W]` and refined by a lightweight residual block:

- `depthwise 3x3 conv`
- `GELU`
- `pointwise 1x1 conv`
- residual add

The last pointwise layer is zero-initialized so the model starts from the original `LoR-SGS` behavior.

### 2. Low-rank Endmember Calibration

The original NMF endmember matrix `E` is kept as the base.

`LoR-SGS+` learns a small low-rank residual:

`E_hat = E + gamma * (U @ V)`

where:

- `U` has shape `[rank, calib_rank]`
- `V` has shape `[calib_rank, C]`
- `gamma` is a learnable scalar initialized to zero

This allows scene-adaptive spectral correction while keeping the parameter overhead small.

## Requirements

```bash
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

## Setup

The estimated coefficient basis matrix file is automatically generated in `HSI/init/`.

You can also use the paper-mode datasets already configured in `main.py`, such as:

- `paviau`
- `urban`
- `salinas`
- `longkou`

## Example Usage

Run the following commands sequentially on `PaviaU`:

```bash
python endmember.py --mode paper --dataset paviau --rank 12
python main.py --mode paper --dataset paviau --num_points 14500 --iterations 10000
```

Disable the two new modules for near-baseline behavior:

```bash
python main.py \
  --mode paper \
  --dataset paviau \
  --num_points 14500 \
  --iterations 10000 \
  --no-use_coef_refine \
  --no-use_endmember_calib
```

Enable the full `LoR-SGS+` setting explicitly:

```bash
python main.py \
  --mode paper \
  --dataset paviau \
  --num_points 14500 \
  --iterations 10000 \
  --use_coef_refine \
  --coef_refine_kernel 3 \
  --use_endmember_calib \
  --calib_rank 2
```

## New CLI Arguments

- `--use_coef_refine` / `--no-use_coef_refine`
- `--coef_refine_kernel`
- `--use_endmember_calib` / `--no-use_endmember_calib`
- `--calib_rank`

## Notes

- `LoR-SGS+` is implemented in a separate folder and does not modify the original `LoR-SGS`.
- Quantization and codec logic are kept unchanged from the current `LoR-SGS` implementation.
- The main implementation changes are concentrated in `gaussianimage_cholesky_unknown.py`.
