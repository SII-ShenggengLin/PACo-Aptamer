#!/usr/bin/env python3
"""41a：构建统一序列库（新划分阶段 0 第一步）
全池 = chains_pass 6,988 链 + s2 复合物独有链（104）
序列来源：processed.pt（3,910）+ s2_samples.pt（859）+ chain_pdb 提取（其余，biotite）
产出：data/splits_v2/seqs_all.fasta（>stem\nSEQ）+ seqs_all.json + 覆盖统计
"""
import os, json, sys
import numpy as np
import torch
import biotite.structure.io.pdb as pdbio

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CHAIN_DIR = os.path.join(DATA, "chain_pdb")
OUT_DIR = os.path.join(DATA, "splits_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. processed.pt（标准序列口径：A/C/G/U/X）
proc = torch.load(os.path.join(DATA, "grnade_btrain", "processed.pt"),
                  map_location="cpu", weights_only=False)
seqs = {k: proc[k]["sequence"] for k in proc}
print(f"processed.pt: {len(seqs)}", flush=True)

# 2. s2_samples.pt（复合物链）
s2 = torch.load(os.path.join(DATA, "s2_btrain", "s2_samples.pt"), map_location="cpu")
for k, v in s2.items():
    seqs[k] = v["sequence"]   # 复合物独有链在这里补齐；已有键覆盖（保持 processed 口径优先？s2 与 processed 重叠键序列应一致）
print(f"+s2: {len(seqs)}", flush=True)

# 3. chains_pass.json 全集
chains = json.load(open(os.path.join(DATA, "chains_pass.json")))
cp_keys = set(f"{c['pdb']}_{c['chain']}" for c in chains)
print(f"chains_pass 全集: {len(cp_keys)}", flush=True)

# 4. 缺失链从 chain_pdb 提取（biotite）
# chain_pdb 是 BGSU 单链 RNA 文件；RNA 残基名：A/C/G/U 单字母或修饰名（末字符为碱基）
def extract_chain_seq(f):
    # chain_pdb 为 BGSU 单链文件（链已重命名 A）
    # 注意 res_name 是原子级数组 → 取 P 骨架原子作为残基代表（每残基 1 个 P）
    stack = pdbio.PDBFile.read(f).get_structure()
    arr = stack[0]
    arr = arr[~arr.hetero]                      # 去水/配体/离子
    arr_p = arr[arr.atom_name == "P"]           # 残基级代表原子
    if arr_p.array_length() == 0:
        return None
    seq = []
    for rn in arr_p.res_name:
        rn = rn.strip().upper()
        base = "X"
        for b in "ACGU":
            if rn == b or rn.endswith(b):
                base = b
                break
        seq.append(base)
    s = "".join(seq)
    return s if s else None

missing = sorted(cp_keys - set(seqs))
print(f"需从 PDB 提取: {len(missing)}", flush=True)
extracted = 0
for stem in missing:
    f = os.path.join(CHAIN_DIR, stem + ".pdb")
    if not os.path.exists(f):
        continue
    try:
        seq = extract_chain_seq(f)
        if seq is None or len(seq) > 600:   # chains_pass 上限 512；超长=提取错误
            continue
        seqs[stem] = seq
        extracted += 1
    except Exception as e:
        pass   # 失败链留空（不打印刷屏；统计最终缺口）
print(f"提取成功: {extracted}", flush=True)

# 5. 最终库（含复合物独有链）
all_keys = cp_keys | set(s2.keys())
seqs_final = {k: seqs[k] for k in all_keys if k in seqs}
print(f"最终序列库: {len(seqs_final)} 链", flush=True)

# 6. 写 fasta + json
with open(os.path.join(OUT_DIR, "seqs_all.fasta"), "w") as f:
    for k in sorted(seqs_final):
        f.write(f">{k}\n{seqs_final[k]}\n")
json.dump(seqs_final, open(os.path.join(OUT_DIR, "seqs_all.json"), "w"))
lens = [len(v) for v in seqs_final.values()]
print(f"长度: min {min(lens)} / max {max(lens)} / 中位 {sorted(lens)[len(lens)//2]}", flush=True)
print("完成", flush=True)
