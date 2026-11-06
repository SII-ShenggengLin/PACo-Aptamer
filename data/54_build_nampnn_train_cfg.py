#!/usr/bin/env python3
"""第 54 步：NA-MPNN v2 重训 CSV + 训练配置生成

- train.csv / valid.csv：na_run.py 期望的全部列（structure_path + date + sampling_probability
  + ppm_paths + asmb_lengths_path + 7 个 asmb_*_path），链 = splits_v2 train/val ∩ 52 号预处理成功
- nampnn_train_v2.json：基于官方 design_model.json，仅改路径/TOTAL_STEPS/PREV_CHECKPOINT，
  架构与超参全保持官方（HIDDEN_DIM 128 / 3+3 层 / k=32 / Noam / label smoothing 0.1 / AMP）
输出：data/nampnn_btrain_v2/{train.csv,valid.csv,nampnn_train_v2.json}
"""
import os, json
import pandas as pd

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
PREPROC = os.path.join(DATA_DIR, "nampnn_btrain_v2", "preproc")
CANON = os.path.join(DATA_DIR, "canonical_v2")

ASMB_COLS = ["asmb_interface_masks", "asmb_side_chain_interface_masks",
             "asmb_nearest_protein_side_chain_index", "asmb_base_pair_masks",
             "asmb_base_pair_index", "asmb_canonical_base_pair_masks",
             "asmb_canonical_base_pair_index"]

def build_csv(chains, out_path):
    rows = []
    dropped = []
    for stem in chains:
        lp = os.path.join(PREPROC, "asmb_lengths", f"{stem}.npy")
        pdb = os.path.join(CANON, f"{stem}.pdb")
        if not os.path.exists(lp) or not os.path.exists(pdb):
            dropped.append(stem)
            continue
        row = {
            "structure_path": pdb,
            "date": "2020-01-01",          # 占位（DATE_CUTOFF 2030 全通过；日期信息不参与训练）
            "sampling_probability": 1.0,   # 无聚簇降权（划分已是簇级隔离；等权采样）
            "ppm_paths": "[]",             # 无实验 PPM 数据
            "asmb_lengths_path": lp,
        }
        for c in ASMB_COLS:
            row[c + "_path"] = os.path.join(PREPROC, c, f"{stem}.npy")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return len(df), dropped

def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    n_train, dt = build_csv(sorted(sp["train"]), os.path.join(DATA_DIR, "nampnn_btrain_v2", "train.csv"))
    n_val, dv = build_csv(sorted(sp["val"]), os.path.join(DATA_DIR, "nampnn_btrain_v2", "valid.csv"))
    print(f"train.csv: {n_train} 链（剔除 {len(dt)}）；valid.csv: {n_val} 链（剔除 {len(dv)}）", flush=True)

    params = json.load(open(os.path.join(ROOT, "tools", "NA-MPNN", "design_model.json")))
    params.update({
        "DF_PATH_TRAIN": os.path.join(DATA_DIR, "nampnn_btrain_v2", "train.csv"),
        "DF_PATH_VALID": os.path.join(DATA_DIR, "nampnn_btrain_v2", "valid.csv"),
        "BASE_FOLDER": os.path.join(ROOT, "models", "NA-MPNN_btrain_v2"),
        "PREV_CHECKPOINT": "",            # 从头训练（严格隔离口径）
        "TOTAL_STEPS": 20000,             # ~110 steps/epoch × 180 epoch；按 val 选最优 s_*.pt
        "SAVE_EVERY_N_STEPS": 1000,
        "MAX_NUMBER_OF_PDBS_VALID": 200,  # val 96 全用
    })
    with open(os.path.join(DATA_DIR, "nampnn_btrain_v2", "nampnn_train_v2.json"), "w") as f:
        json.dump(params, f, indent=1)
    print("训练配置写出: nampnn_train_v2.json", flush=True)

if __name__ == "__main__":
    main()
