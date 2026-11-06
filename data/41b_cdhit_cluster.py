#!/usr/bin/env python3
"""41b：CD-HIT 0.8 序列预聚类（新划分阶段 0 第二步）
用法: python 41b_cdhit_cluster.py
产出: splits_v2/cdhit_clusters.json（{rep: [members]}）+ cdhit_out.clstr
"""
import os, json, subprocess

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
OUT = os.path.join(DATA, "splits_v2")
os.makedirs(OUT, exist_ok=True)
FASTA = os.path.join(OUT, "seqs_all.fasta")
PREFIX = os.path.join(OUT, "cdhit_out")

# cd-hit -c 0.8（核酸 4 字母，word size 4；-g 1 严格代表选择）
cmd = ["cd-hit", "-i", FASTA, "-o", PREFIX, "-c", "0.8", "-n", "4", "-g", "1",
       "-T", "8", "-M", "60000", "-d", "0"]
print("运行:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
print(r.stdout[-1500:] if r.stdout else "", flush=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:], flush=True)
    raise SystemExit(f"cd-hit 失败 rc={r.returncode}")

# 解析 .clstr
clusters = []
cur = None
with open(PREFIX + ".clstr") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">Cluster"):
            if cur: clusters.append(cur)
            cur = {"rep": None, "members": []}
        elif line:
            # "0	812nt, >1A9N_Q... *" / "... at 80%"
            if ">" in line:
                name = line.split(">")[1].split("...")[0]
                is_rep = line.rstrip().endswith("*")
                cur["members"].append(name)
                if is_rep:
                    cur["rep"] = name
if cur: clusters.append(cur)
n_single = sum(1 for c in clusters if len(c["members"]) == 1)
print(f"CD-HIT 簇数: {len(clusters)}（单成员簇 {n_single}）", flush=True)
total = sum(len(c["members"]) for c in clusters)
print(f"覆盖链数: {total}", flush=True)

json.dump({c["rep"]: c["members"] for c in clusters},
          open(os.path.join(OUT, "cdhit_clusters.json"), "w"), indent=1)
print("完成", flush=True)
