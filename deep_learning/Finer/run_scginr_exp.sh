#!/bin/bash

# SCGINR vs SPINR Comparison Experiment on PaviaU
# This script runs 5 experiments to compare SCGINR with SPINR

DATASET_PATH="/home/wenchang/WangLL/dataset/PaviaU/PaviaU.mat"
GPU_ID=0
SEED=1
NUM_ITERS=5000
LR=2e-4
HIDDEN_LAYERS=0
HIDDEN_FEATURES=256
TABLE_DIM=8
QUANT_BITS=8

echo "=========================================="
echo "SCGINR vs SPINR Comparison Experiments"
echo "Dataset: PaviaU"
echo "=========================================="

# Experiment 1: SCGINR with pool_size=3 (default)
echo ""
echo "Experiment 1/5: SCGINR (pool_size=3)"
echo "------------------------------------------"
python main.py \
  --model_type SCGINR \
  --img_path $DATASET_PATH \
  --gpu_id $GPU_ID \
  --seed $SEED \
  --num_iters $NUM_ITERS \
  --learning_rate $LR \
  --hidden_layers $HIDDEN_LAYERS \
  --hidden_features $HIDDEN_FEATURES \
  --table_dim $TABLE_DIM \
  --pool_size 3 \
  --use_quant \
  --quant_bits $QUANT_BITS \
  --skip_keys net \
  --axis 0

# Experiment 2: SCGINR with pool_size=5
echo ""
echo "Experiment 2/5: SCGINR (pool_size=5)"
echo "------------------------------------------"
python main.py \
  --model_type SCGINR \
  --img_path $DATASET_PATH \
  --gpu_id $GPU_ID \
  --seed $SEED \
  --num_iters $NUM_ITERS \
  --learning_rate $LR \
  --hidden_layers $HIDDEN_LAYERS \
  --hidden_features $HIDDEN_FEATURES \
  --table_dim $TABLE_DIM \
  --pool_size 5 \
  --use_quant \
  --quant_bits $QUANT_BITS \
  --skip_keys net \
  --axis 0

# Experiment 3: SCGINR with pool_size=7
echo ""
echo "Experiment 3/5: SCGINR (pool_size=7)"
echo "------------------------------------------"
python main.py \
  --model_type SCGINR \
  --img_path $DATASET_PATH \
  --gpu_id $GPU_ID \
  --seed $SEED \
  --num_iters $NUM_ITERS \
  --learning_rate $LR \
  --hidden_layers $HIDDEN_LAYERS \
  --hidden_features $HIDDEN_FEATURES \
  --table_dim $TABLE_DIM \
  --pool_size 7 \
  --use_quant \
  --quant_bits $QUANT_BITS \
  --skip_keys net \
  --axis 0

# Experiment 4: SCGINR with different learning rate (1.5e-4)
echo ""
echo "Experiment 4/5: SCGINR (pool_size=3, lr=1.5e-4)"
echo "------------------------------------------"
python main.py \
  --model_type SCGINR \
  --img_path $DATASET_PATH \
  --gpu_id $GPU_ID \
  --seed $SEED \
  --num_iters $NUM_ITERS \
  --learning_rate 1.5e-4 \
  --hidden_layers $HIDDEN_LAYERS \
  --hidden_features $HIDDEN_FEATURES \
  --table_dim $TABLE_DIM \
  --pool_size 3 \
  --use_quant \
  --quant_bits $QUANT_BITS \
  --skip_keys net \
  --axis 0

# Experiment 5: SPINR baseline (for comparison)
echo ""
echo "Experiment 5/5: SPINR (baseline)"
echo "------------------------------------------"
python main.py \
  --model_type SPINR \
  --img_path $DATASET_PATH \
  --gpu_id $GPU_ID \
  --seed $SEED \
  --num_iters $NUM_ITERS \
  --learning_rate $LR \
  --hidden_layers $HIDDEN_LAYERS \
  --hidden_features $HIDDEN_FEATURES \
  --table_dim $TABLE_DIM \
  --use_quant \
  --quant_bits $QUANT_BITS \
  --skip_keys net \
  --axis 0

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
