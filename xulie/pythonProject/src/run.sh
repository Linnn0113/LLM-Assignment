#!/bin/bash
echo "=================================================="
echo "       开始运行 Encoder-Decoder (Seq2Seq) 实验"
echo "=================================================="

# 设置随机种子以保证可复现性
SEED=42

# --- 1. 运行基线模型 (Baseline) ---
echo "--- (1/3) 正在运行: 基线模型 (带 PE) ---"
python train.py \
    --experiment_name "baseline_seq2seq" \
    --seed $SEED \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 8 \
    --d_ff 1024 \
    --batch_size 32 \
    --max_len 64 \
    --epochs 13 \
    --lr 3e-4 \
    --grad_clip 1.0

# --- 2. 运行消融实验 (Ablation) ---
echo "--- (2/3) 正在运行: 消融实验 (无 PE) ---"
python train_seq2seq.py \
    --experiment_name "no_pe_seq2seq" \
    --seed $SEED \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 8 \
    --d_ff 1024 \
    --batch_size 32 \
    --max_len 64 \
    --epochs 13 \
    --lr 3e-4 \
    --grad_clip 1.0 \
    --disable_pe # <-- 关键标志：禁用位置编码

# --- 3. 生成对比图表 ---
echo "--- (3/3) 正在生成 Seq2Seq 对比图表 ---"
python plot_comparison.py

echo "=================================================="
echo "        Encoder-Decoder (Seq2Seq) 实验已全部完成！"
echo "=================================================="