#!/usr/bin/env python3
"""qTMclust 结构聚类（TM>0.45）— AlignIF 复现第 4 步
用法: python3 04_cluster.py [最大链数，默认全部；小样本测试用]
"""
import os, sys, subprocess, time

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
CHAIN_DIR = os.path.join(DATA, "chain_pdb")
QTMCLUST = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/tools/USalign-pylelab/qTMclust"

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
files = sorted(os.listdir(CHAIN_DIR))
if limit > 0:
    files = files[:limit]

# 生成 chain list
list_f = os.path.join(DATA, "chain_list.txt")
with open(list_f, "w") as f:
    for name in files:
        f.write(os.path.splitext(name)[0] + "\n")

out_f = os.path.join(DATA, "cluster_out.txt")
t0 = time.time()
cmd = [QTMCLUST, "-dir", CHAIN_DIR + "/", list_f, "-TMcut", "0.45",
       "-ter", "3", "-split", "0", "-o", out_f]
print("运行:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
print(r.stdout[-2000:] if r.stdout else "", flush=True)
if r.stderr:
    print("STDERR:", r.stderr[-2000:], flush=True)
print(f"qTMclust 完成，耗时 {(time.time()-t0)/60:.1f} 分钟")

# 统计簇
clusters = []
with open(out_f) as f:
    for line in f:
        cols = [c for c in line.strip().split("\t") if c]
        if cols:
            clusters.append(cols)
print(f"总簇数: {len(clusters)}")
sizes = sorted(len(c) for c in clusters)
print(f"簇大小: min={sizes[0]}, max={sizes[-1]}, 中位={sizes[len(sizes)//2]}")
print(f"代表链: {[c[0] for c in clusters[:10]]}")
