#!/bin/bash
# 消融 v2 重跑（新划分 splits_v2）——参数化启动脚本
# 用法: bash data/run_ablation_v2.sh <消融名: no_ss|no_aug|no_ligand|no_iface> <GPU物理号>
# 流程: S1(no_ss/no_aug 需重训) -> S2 -> 38 号 quick 评估（10 样本/温度，相对诊断口径）
# 产物: models/PACoAptamer/s{1,2}_v2_<name>.pth + survey/metrics_PACo_v2_<name>.json
# 注意: 评估 quick 池仅 30 样本，不能与主表 300 池直接比（池膨胀 +6.7pp 教训），仅作相对排序
set -e
NAME="$1"
GPU="$2"
BASE="/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
PY="/root/miniconda3/envs/aptamer_eval/bin/python"
cd "$BASE"

echo "===== 消融 v2: $NAME on GPU $GPU ====="
date

EXTRA=""
if [ "$NAME" = "no_ss" ]; then EXTRA="--no_ss"; fi
if [ "$NAME" = "no_aug" ]; then EXTRA="--no_aug"; fi

if [ "$NAME" = "no_ss" ] || [ "$NAME" = "no_aug" ]; then
  echo "-- S1 重训 ($NAME) --"
  # 注意：train.py --out 会自动拼 .pth（勿传后缀，否则 .pth.pth）
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/PACoAptamer/train.py --stage s1 --gpu 0 --splits_v2 $EXTRA --out s1_v2_$NAME
fi

INIT="models/PACoAptamer/s1_v2.pth"   # no_ligand/no_iface 复用主 S1
if [ "$NAME" = "no_ss" ] || [ "$NAME" = "no_aug" ]; then
  INIT="models/PACoAptamer/s1_v2_$NAME.pth"
fi

echo "-- S2 微调 ($NAME, init $INIT) --"
if [ "$NAME" = "no_ligand" ]; then EXTRA="--no_ligand"; fi
if [ "$NAME" = "no_iface" ]; then EXTRA="--no_iface"; fi
CUDA_VISIBLE_DEVICES=$GPU $PY tools/PACoAptamer/train.py --stage s2 --gpu 0 --init "$INIT" --s2_data s2_samples_v2.pt $EXTRA --out s2_v2_$NAME

echo "-- 评估 (quick 10/温度, 30 池) --"
EVAL_EXTRA=""
if [ "$NAME" = "no_ss" ]; then EVAL_EXTRA="--no_ss"; fi
CUDA_VISIBLE_DEVICES=$GPU $PY data/38_eval_paco_btest.py --ckpt models/PACoAptamer/s2_v2_$NAME.pth --gpu 0 --splits_v2 --quick 10 --out metrics_PACo_v2_$NAME.json $EVAL_EXTRA

echo "===== 消融 v2 [$NAME] 全部完成 $(date) ====="
