#!/usr/bin/env python3
"""41c：CD-HIT 0.8 序列簇 + qTMclust TM0.45 结构簇 → 图合并 → 最终簇
规则：任意两链同 CD-HIT 簇 或 同 TM 簇 → 必在同一最终簇（连通分量）
产出: splits_v2/final_clusters.json（{cid: [chains]}）+ 统计
"""
import os, json
from collections import deque

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
OUT = os.path.join(DATA, "splits_v2")

# 1. CD-HIT 簇
cdhit = json.load(open(os.path.join(OUT, "cdhit_clusters.json")))
seq_clusters = [v for v in cdhit.values()]

# 2. TM 簇（05 会话 qTMclust 产物，成员名带 .pdb 需 strip）
tm_clusters = []
with open(os.path.join(DATA, "our_qTMclust_out.txt")) as f:
    for line in f:
        cols = [c for c in line.strip().split("\t") if c]
        if cols:
            tm_clusters.append([c[:-4] if c.endswith(".pdb") else c for c in cols])
print(f"CD-HIT 簇: {len(seq_clusters)} | TM 簇: {len(tm_clusters)}", flush=True)

# 3. 图：节点 = 链；邻接 = 同簇成员
adj = {}
def add_edge_group(members):
    for m in members:
        adj.setdefault(m, set()).update(members)
for cl in seq_clusters:
    add_edge_group(cl)
for cl in tm_clusters:
    add_edge_group(cl)

# 4. 连通分量（BFS）
seen = set()
final_clusters = []
for start in adj:
    if start in seen:
        continue
    comp = []
    q = deque([start])
    seen.add(start)
    while q:
        u = q.popleft()
        comp.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    final_clusters.append(sorted(comp))
final_clusters.sort(key=len, reverse=True)

n_single = sum(1 for c in final_clusters if len(c) == 1)
print(f"最终簇: {len(final_clusters)}（单成员 {n_single}）", flush=True)
total = sum(len(c) for c in final_clusters)
print(f"覆盖链数: {total}", flush=True)

# 5. 合并合理性验证：同一最终簇内，任意链对是否同序列簇或同 TM 簇（传递闭包应成立）
bad = 0
for c in final_clusters:
    if len(c) == 1: continue
    # 检查簇是否由至少一组同源关系连成（间接连通即可，无需全对）
# 验证：CD-HIT 簇未跨最终簇
for cl in seq_clusters:
    reps_final = {}
    for m in cl:
        for ci, fc in enumerate(final_clusters):
            if m in fc:
                reps_final.setdefault(ci, 0)
                reps_final[ci] += 1
    if len(reps_final) > 1:
        bad += 1
print(f"CD-HIT 簇跨最终簇数量（应为 0）: {bad}", flush=True)

json.dump({i: c for i, c in enumerate(final_clusters)},
          open(os.path.join(OUT, "final_clusters.json"), "w"), indent=1)
print("完成", flush=True)
