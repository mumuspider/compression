# Finer: Periodic INR Baseline for HSI Compression

## 模型定位

`Finer` 来自 `/home/wenchang/WangLL/HSI-Compression-benchmark/INR` 的复制副本。

- 原始 `INR/` 目录不做修改
- 正式 benchmark 中统一使用 `--model_type Finer`
- 当前默认比较中，`Finer` 替代 `NeRF` 进入主表，但 `NeRF/` 目录继续保留在磁盘上

## 当前 benchmark preset

- `Brain @ bpppb≈0.5233`: `hidden_layers=5`, `hidden_features=768`
- `Salinas @ bpppb≈0.75`: `hidden_layers=5`, `hidden_features=640`
- `PaviaU @ bpppb≈0.6`: `hidden_layers=6`, `hidden_features=512`
- `Indian Pines @ bpppb≈1.0`: `hidden_layers=3`, `hidden_features=400`

## 示例命令

```bash
cd /home/wenchang/WangLL/HSI-Compression-benchmark/methods/deep_learning/Finer
conda run -n AE python main.py \
  --model_type Finer \
  --img_path /home/wenchang/WangLL/HSI-Compression-benchmark/dataset/indiapine/Indian_pines.mat \
  --gpu_id 0 \
  --seed 1 \
  --num_iters 5000 \
  --learning_rate 2e-4 \
  --hidden_layers 3 \
  --hidden_features 400 \
  --first_omega 30 \
  --hidden_omega 30 \
  --use_quant \
  --quant_bits 8 \
  --axis 0
```
