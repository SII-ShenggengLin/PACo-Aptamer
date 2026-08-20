#!/usr/bin/env python3
"""用官方 tmscores 全两两矩阵 + 序列预聚类，还原官方 499 簇"""
import os, torch, json
from collections import defaultdict

D = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
AL = f"{D}/alignif/data"

def extract_seq(path):
    """提取残基名序列（去重连续相同残基号）"""
    seq = []
    prev = None
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                resname = line[17:20].strip()
                seqnum = line[22:26].strip()
                if (resname, seqnum) != prev:
                    seq.append(resname)
                    prev = (resname, seqnum)
            else:
                prev = None
    return seq

split = torch.load(f"{AL}/split.pt", map_location='cpu', weights_only=False)
order = split['train'] + split['val'] + split['test']
print(f"benchmark 链: {len(order)}")

# 1. 提取序列
seqs = {}
for name in order:
    p = f"{AL}/pdbs/{name}.pdb"
    if os.path.exists(p):
        seqs[name] = "".join(extract_seq(p))
print(f"提取到序列: {len(seqs)}")

# 2. 精确序列分组（CD-HIT 0.8 的保守下界；先看精确分组的簇数）
seq_groups = defaultdict(list)
for name, s in seqs.items():
    seq_groups[s].append(name)
print(f"精确序列簇: {len(seq_groups)}")

# 3. 代表增量聚类（用 tmscores）
file_map = {}
for f in os.listdir(f"{AL}/tmscores"):
    file_map[f[:-3].lower()] = f[:-3]

reps = [v[0] for v in seq_groups.values()]
cache = {}
clusters = []
missing = 0
for rep in reps:
    key = rep.lower()
    if key not in cache:
        if key not in file_map:
            missing += 1
            continue
        cache[key] = torch.load(f"{AL}/tmscores/{file_map[key]}.pt", map_location='cpu', weights_only=False)
    d = cache[key]
    joined = None
    for ci, (crep, members) in enumerate(clusters):
        tm = d.get(crep)
        if tm is not None and isinstance(tm, float) and tm >= 0.45:
            joined = ci
            break
    if joined is None:
        clusters.append((rep, list(seq_groups["".join(seqs[rep])])))
    else:
        clusters[joined][1].extend(seq_groups["".join(seqs[rep])])

print(f"缺失 TM 代表: {missing}")
print(f"序列预聚类 + 代表增量 簇数: {len(clusters)}")
sizes = sorted(len(m) for _, m in clusters)
print(f"簇大小: min={sizes[0]} max={sizes[-1]} 中位={sizes[len(sizes)//2]}")
print(f">1链簇: {sum(1 for s in sizes if s>1)}  单例: {sum(1 for s in sizes if s==1)}")

# 保存簇（成员列表）
json.dump([m for _, m in clusters],
          open(f"{D}/official_rebuilt_clusters.json", "w"), indent=0)
print("已保存 official_rebuilt_clusters.json")
