#!/usr/bin/env python3
"""第 23 步：B 层 gRNAde 重训数据构建（我们的严格划分）

输入：blayer_splits.json（train/val/test 链）+ chain_pdb/（已切链）
输出：data/grnade_btrain/processed.pt（gRNAde 格式）+ b_split.pt（划分索引）

processed.pt 格式（对齐官方）：
  {key: {'sequence', 'id_list', 'coords_list' [(L,27,3)], 'sec_struct_list', ...}}
27 原子 = gRNAde RNA_ATOMS（P/C5'/O5'/C4'/O4'/C3'/O3'/C2'/O2'/C1'/碱基 14 + OP1/OP2）
缺失原子用 1e-5 填充；SS 用 RNAfold 预测（viennarna_fold_env）
"""
import os, json, sys, torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde")
from src.constants import RNA_ATOMS

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CHAIN_DIR = os.path.join(DATA_DIR, "chain_pdb")
OUT_DIR = os.path.join(DATA_DIR, "grnade_btrain")
os.makedirs(OUT_DIR, exist_ok=True)
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
FILL = 1e-5

def extract_coords(pdb_path, seq):
    """提取 (L, 27, 3) 坐标（按 RNA_ATOMS 顺序，缺失填充）"""
    residues = {}
    order = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            key = (line[22:26].strip(), line[26].strip())
            atom = line[12:16].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            if key not in residues:
                residues[key] = {}
                order.append(key)
            residues[key][atom] = (x, y, z)
    if len(order) != len(seq):
        return None
    coords = np.full((len(seq), len(RNA_ATOMS), 3), FILL, dtype=np.float32)
    for i, key in enumerate(order):
        for j, atom_name in enumerate(RNA_ATOMS):
            if atom_name in residues[key]:
                coords[i, j] = residues[key][atom_name]
    return coords

def predict_ss(seq):
    import subprocess, re
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n", capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return m[0] if m else None

def work_one(stem):
    pdb_path = os.path.join(CHAIN_DIR, f"{stem}.pdb")
    if not os.path.exists(pdb_path):
        return (stem, None)
    seq = None
    # 从 PDB 提取序列
    seen = set()
    seq_list = []
    for line in open(pdb_path):
        if line.startswith("ATOM"):
            rn = line[17:20].strip()
            num = line[22:26].strip()
            if (rn, num) not in seen:
                seen.add((rn, num))
                seq_list.append(rn)
    seq = "".join(seq_list)
    if not seq or any(c not in "ACGU" for c in seq):
        return (stem, None)
    coords = extract_coords(pdb_path, seq)
    if coords is None:
        return (stem, None)
    ss = predict_ss(seq)
    if not ss:
        return (stem, None)
    return (stem, {
        "sequence": seq,
        "id_list": [stem],
        "coords_list": [torch.tensor(coords)],
        "sec_struct_list": [ss],
    })

def main():
    blayer = json.load(open(os.path.join(DATA_DIR, "survey", "blayer_splits.json")))
    stems = sorted(blayer.keys())
    print(f"待处理链: {len(stems)}", flush=True)

    data = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (stem, r) in enumerate(ex.map(work_one, stems, chunksize=16)):
            if r:
                data[stem] = r
            if (i + 1) % 500 == 0:
                print(f"进度: {i+1}/{len(stems)} 有效 {len(data)}", flush=True)

    # 划分索引（与 processed.pt 键顺序对齐）
    keys = sorted(data.keys())
    idx = {k: i for i, k in enumerate(keys)}
    train_idx = [idx[k] for k in keys if blayer[k] == "train"]
    val_idx = [idx[k] for k in keys if blayer[k] == "val"]
    test_idx = [idx[k] for k in keys if blayer[k] == "test"]

    torch.save(data, os.path.join(OUT_DIR, "processed.pt"))
    torch.save([train_idx, val_idx, test_idx], os.path.join(OUT_DIR, "b_split.pt"))
    print(f"processed.pt: {len(data)} 链 | train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}", flush=True)
    print(f"写入: {OUT_DIR}/processed.pt + b_split.pt", flush=True)

if __name__ == "__main__":
    main()
