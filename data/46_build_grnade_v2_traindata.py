#!/usr/bin/env python3
"""第 46 步：gRNAde B 层 v2 重训数据构建（新划分 splits_v2）

输入：splits_v2.json（train 5,773 / val 96 / test 477）+ 数据源
数据源优先级（对齐 45 号）：
  1. processed.pt（旧 blayer，27 原子 + 官方 SS，3,910 链）
  2. s2_samples.pt（复合物链 27 原子 + RNAfold SS）
  3. chain_pdb/chain_pdb_v2（biotite 提取 + RNAfold SS）
输出：data/grnade_btrain_v2/processed.pt（gRNAde 格式）+ b_split_v2.pt（train/val/test 索引）
"""
import os, json, sys, re, subprocess, torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/gRNAde")
sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
from src.constants import RNA_ATOMS

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CHAIN_DIR = os.path.join(DATA_DIR, "chain_pdb")
CP2_DIR = os.path.join(DATA_DIR, "chain_pdb_v2")
OUT_DIR = os.path.join(DATA_DIR, "grnade_btrain_v2")
os.makedirs(OUT_DIR, exist_ok=True)
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
FILL = 1e-5

sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
ALL = sorted(set(sp["train"]) | set(sp["val"]) | set(sp["test"]))
print(f"目标链: {len(ALL)}", flush=True)


def predict_ss(seq):
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n",
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return m[0] if m else None


def extract_coords(pdb_path, seq):
    """从 PDB 提取 (L, 27, 3)（按 RNA_ATOMS 顺序，缺失填充）"""
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


def seq_from_pdb(pdb_path):
    seen = set()
    seq_list = []
    for line in open(pdb_path):
        if line.startswith("ATOM"):
            rn = line[17:20].strip()
            num = line[22:26].strip()
            if (rn, num) not in seen:
                seen.add((rn, num))
                seq_list.append(rn)
    s = "".join(seq_list)
    return s if s and all(c in "ACGU" for c in s) else None


# 1. processed.pt（直接复用，含官方 SS）
proc = torch.load(os.path.join(DATA_DIR, "grnade_btrain", "processed.pt"),
                  map_location="cpu", weights_only=False)
data = {}
for stem in ALL:
    if stem in proc:
        d = proc[stem]
        if (d["coords_list"][0].shape[0] == len(d["sequence"])
                and len(d["sec_struct_list"][0]) == len(d["sequence"])):
            data[stem] = d
print(f"processed.pt 复用: {len(data)}", flush=True)

# 2. s2_samples.pt（27 原子 + RNAfold SS）
s2 = torch.load(os.path.join(DATA_DIR, "s2_btrain", "s2_samples.pt"), map_location="cpu")
n2 = 0
for stem in ALL:
    if stem in data or stem not in s2:
        continue
    d = s2[stem]
    seq = d["sequence"]
    coords27 = np.array(d["coords_27"], dtype=np.float32)  # G27 顺序，需对齐 RNA_ATOMS
    if coords27.shape[0] != len(seq) or len(seq) > 512:
        continue
    ss = predict_ss(seq)
    if ss is None or len(ss) != len(seq):
        continue
    # 27 原子顺序：processed.pt 用 RNA_ATOMS，s2 用 G27——顺序已验证一致（P..OP2）
    from PACoAptamer.data_utils import G27
    if len(G27) == len(RNA_ATOMS) and all(a == b for a, b in zip(G27, RNA_ATOMS)):
        coords_t = coords27
    else:
        # 重排：RNA_ATOMS 中每个原子在 G27 中的索引
        idx = [G27.index(a) if a in G27 else -1 for a in RNA_ATOMS]
        coords_t = np.full((len(seq), len(RNA_ATOMS), 3), FILL, dtype=np.float32)
        for j, i in enumerate(idx):
            if i >= 0:
                coords_t[:, j] = coords27[:, i]
    data[stem] = {
        "sequence": seq,
        "id_list": [stem],
        "coords_list": [torch.tensor(coords_t)],
        "sec_struct_list": [ss],
    }
    n2 += 1
print(f"s2_samples 覆盖: +{n2}", flush=True)

# 3. chain_pdb 提取
def work_one(stem):
    for d in (CHAIN_DIR, CP2_DIR):
        pdb_path = os.path.join(d, stem + ".pdb")
        if not os.path.exists(pdb_path):
            continue
        seq = seq_from_pdb(pdb_path)
        if not seq or len(seq) > 512:
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
    return (stem, None)

pending = [s for s in ALL if s not in data]
print(f"需 chain_pdb 提取: {len(pending)}", flush=True)
n3 = 0
# 串行提取（fork 模式下 worker 卡死——ProcessPoolExecutor 与顶层大对象加载不兼容；串行 ~0.7s/条 稳定）
for i, stem in enumerate(pending):
    r = work_one(stem)
    if r and r[1]:
        data[stem] = r[1]
        n3 += 1
    if (i + 1) % 400 == 0:
        print(f"进度: {i+1}/{len(pending)} 有效 {n3}", flush=True)
print(f"chain_pdb 提取: +{n3}", flush=True)

missing = [s for s in ALL if s not in data]
print(f"缺失: {len(missing)}", flush=True)
if missing:
    print("  示例:", missing[:15], flush=True)

keys = sorted(data.keys())
idx = {k: i for i, k in enumerate(keys)}
train_idx = [idx[k] for k in keys if k in sp["train"]]
val_idx = [idx[k] for k in keys if k in sp["val"]]
test_idx = [idx[k] for k in keys if k in sp["test"]]
torch.save(data, os.path.join(OUT_DIR, "processed.pt"))
torch.save([train_idx, val_idx, test_idx], os.path.join(OUT_DIR, "b_split_v2.pt"))
print(f"完成: processed.pt {len(data)} 链 | train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}",
      flush=True)
