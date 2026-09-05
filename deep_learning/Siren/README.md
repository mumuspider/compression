# Siren: Sinusoidal INR Baseline for HSI Compression

## Model Summary

`Siren` is the formal coordinate-based baseline used in the current public paper setup.

Core properties:

- input: normalized 2D coordinates `(x, y)`
- decoder: sinusoidal MLP with periodic activations
- no latent codebook
- no low-rank spectral refinement
- no spectral calibration gate

`LSSIR` is compared against this baseline rather than against any unpublished internal model.

## Example Usage

```bash
python main.py \
  --model_type Siren \
  --img_path /home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat \
  --gpu_id 0 \
  --num_iters 5000 \
  --learning_rate 2e-4 \
  --hidden_layers 3 \
  --hidden_features 256 \
  --use_quant \
  --quant_bits 8 \
  --axis 0
```

## Current Paper Presets

- `PaviaU @ target bpppb≈0.6`: `hidden_layers=6`, `hidden_features=512`
- `Indian Pines @ target bpppb≈1.0`: `hidden_layers=3`, `hidden_features=400`
- `Brain @ target bpppb≈0.5233`: `hidden_layers=5`, `hidden_features=768`

## Brain Shortlist

Current Brain shortlist:

- `hidden_layers=5`
- `hidden_features=768`
- `final_bpppb ≈ 0.5106`
- `PSNR ≈ 44.14 dB`
- `SSIM ≈ 0.9743`
- `SAM ≈ 0.0698`
