#!/usr/bin/env python3
"""第 45 步：S1 v2 数据特征化（新划分 splits_v2 全量）

- 覆盖：splits_v2 的 train(5,773) + val(96) + test(477) 链
- 数据源优先级：
  1. processed.pt（27 原子 + 官方 SS，3,910 链）
  2. s2_samples.pt（复合物链 27 原子）
  3. chain_pdb（biotite 提取骨架坐标 + RNAfold SS 预测，41a 同款 P 原子代表）
- 输出：data/paco_s1_v2.pt（train+val+test 全部，键 {stem: {coords9, ss_pairs, seq}}）
  评估/训练按 splits_v2.json 的 group 切分（load_s1 的 cache 逻辑）
"""
import os, re, json, subprocess, sys
import numpy as np
import torch
import biotite.structure.io.pdb as pdbio

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CHAIN_DIR = os.path.join(DATA, "chain_pdb")
CP2_DIR = os.path.join(DATA, "chain_pdb_v2")
OUT = os.path.join(DATA, "paco_s1_v2.pt")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"

sys.path.insert(0, "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools")
from PACoAptamer.data_utils import to_paco9, ss_to_pairs

G27 = ["P", "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
       "N1", "C2", "O2", "N2", "N3", "C4", "O4", "N4", "C5", "C6", "O6",
       "N6", "N7", "C8", "N9", "OP1", "OP2"]
STD = {"A", "C", "G", "U"}

sp = json.load(open(os.path.join(DATA, "splits_v2", "splits_v2.json")))
ALL = sorted(set(sp["train"]) | set(sp["val"]) | set(sp["test"]))
print(f"目标链: {len(ALL)}", flush=True)

# 1. processed.pt
proc = torch.load(os.path.join(DATA, "grnade_btrain", "processed.pt"),
                  map_location="cpu", weights_only=False)
out = {}
for stem in ALL:
    if stem in proc:
        d = proc[stem]
        seq = d["sequence"]
        coords27 = np.array(d["coords_list"][0], dtype=np.float32)
        ss = d["sec_struct_list"][0] if "sec_struct_list" in d else None
        if coords27.shape[0] == len(seq) and ss and len(ss) == len(seq):
            out[stem] = {"coords9": to_paco9(coords27), "ss_pairs": ss_to_pairs(ss), "seq": seq}
print(f"processed.pt 覆盖: {len(out)}", flush=True)

def predict_ss(seq):
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n",
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    return m[0] if m else None


# 2. s2_samples.pt
s2 = torch.load(os.path.join(DATA, "s2_btrain", "s2_samples.pt"), map_location="cpu")
n2 = 0
for stem in ALL:
    if stem in out or stem not in s2:
        continue
    d = s2[stem]
    seq = d["sequence"]
    coords27 = np.array(d["coords_27"], dtype=np.float32)
    if coords27.shape[0] != len(seq):
        continue
    # s2 无 SS → RNAfold 预测
    ss = predict_ss(seq)
    if ss is None or len(ss) != len(seq):
        continue
    out[stem] = {"coords9": to_paco9(coords27), "ss_pairs": ss_to_pairs(ss), "seq": seq}
    n2 += 1
print(f"s2_samples 覆盖: +{n2}", flush=True)


def extract_pdb_stem(stem):
    """从 chain_pdb/chain_pdb_v2 提取 27 原子骨架 + 序列（P 原子代表残基）"""
    for d in (CHAIN_DIR, CP2_DIR):
        f = os.path.join(d, stem + ".pdb")
        if not os.path.exists(f):
            continue
        arr = pdbio.PDBFile.read(f).get_structure()[0]
        arr = arr[~arr.hetero]
        ap = arr[arr.atom_name == "P"]
        if ap.array_length() == 0:
            return None
        # 残基代表（P 原子序号）→ 27 原子组装
        seq = []
        res_keys = []
        for rn in ap.res_name:
            rn = str(rn).strip().upper()
            b = "X"
            for x in "ACGU":
                if rn == x or rn.endswith(x):
                    b = x
                    break
            seq.append(b)
        s = "".join(seq)
        if len(s) > 512:
            return None
        # 按残基组装 27 原子（缺失位 FILL 1e-5，对齐 processed 语义）
        coords27 = np.full((len(s), 27, 3), 1e-5, dtype=np.float32)
        for i in range(len(s)):
            # 该残基在原始 array 中的行
            mask = (arr.res_id == ap.res_id[i])
            sub = arr[mask]
            for atom in sub.atom_name:
                a = str(atom)
                if a in G27:
                    coords27[i, G27.index(a)] = sub.coord[sub.atom_name == atom][0]
        return s, coords27
    return None


# 3. chain_pdb 提取
n3 = 0
for stem in ALL:
    if stem in out:
        continue
    got = extract_pdb_stem(stem)
    if got is None:
        continue
    s, coords27 = got
    ss = predict_ss(s)
    if ss is None or len(ss) != len(s):
        continue
    out[stem] = {"coords9": to_paco9(coords27), "ss_pairs": ss_to_pairs(ss), "seq": s}
    n3 += 1
print(f"chain_pdb 提取: +{n3}", flush=True)

missing = [s for s in ALL if s not in out]
print(f"缺失: {len(missing)}", flush=True)
if missing:
    print("  示例:", missing[:15], flush=True)
torch.save(out, OUT)
print(f"完成: {len(out)} 链 → {OUT}", flush=True)
