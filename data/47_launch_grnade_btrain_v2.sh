#!/bin/bash
# 第 47 步：gRNAde B 层 v2 重训启动脚本（新划分 splits_v2）
# 依赖：data/46 完成后（grnade_btrain_v2/processed.pt + b_split_v2.pt）
# 用法：CUDA_VISIBLE_DEVICES=<gpu> bash data/47_launch_grnade_btrain_v2.sh <gpu>
set -e
cd /inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF
GPU=${1:-0}
export X3DNA=/tmp/x3dna
export ETERNAFOLD=/tmp/eternafold
export PROJECT_PATH=/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde
export DATA_PATH=/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data/grnade_btrain_v2/
export CKPT_DIR=/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/models/gRNAde_btrain_v2
mkdir -p "$CKPT_DIR"

PY=/root/miniconda3/envs/aptamer_eval/bin/python
# 46 号输出 b_split_v2.pt → main.py 需要 das_split.pt 命名（split_type='das'）
cp -f data/grnade_btrain_v2/b_split_v2.pt data/grnade_btrain_v2/das_split.pt

$PY tools/gRNAde/main.py \
    --config tools/gRNAde/configs/btrain_v2.yaml \
    --expt_name grnade_btrain_v2 \
    --no_wandb
