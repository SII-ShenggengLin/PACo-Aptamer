#!/usr/bin/env python3
"""簇级划分 + 时间隔离验证 — AlignIF 复现第 5 步
AlignIF 划分：训练 399 簇 / 验证 50 簇 / 测试 50 簇（按簇划分，防泄漏）
独立测试集：时间隔离（2024-05-01 → 2025-01-01 新释放，且与训练 TM<0.45）
"""
import json, os, random
from collections import defaultdict

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"

# 1. 读簇（我们管线的 qTMclust 输出；成员名带 .pdb 后缀需去掉）
clusters = []
with open(os.path.join(DATA, "our_qTMclust_out.txt")) as f:
    for line in f:
        cols = [c for c in line.strip().split("\t") if c]
        if cols:
            clusters.append([c[:-4] if c.endswith(".pdb") else c for c in cols])

# 2. 读链信息（pdb/chain/length/resolution）
chains_meta = {f"{c['pdb']}_{c['chain']}": c for c in json.load(open(os.path.join(DATA, "chains_pass.json")))}

# 3. 读释放日期
bgsu = json.load(open("/tmp/bgsu_pdbs.json"))
def release_date(chain_name):
    pdb = chain_name.split("_")[0].lower()
    return bgsu.get(pdb.upper(), {}).get("release_date", "9999-99-99")

# 4. 簇内链的日期标记
for ci, cl in enumerate(clusters):
    dates = [release_date(m) for m in cl]
    cl_dates = dates
    clusters[ci] = (cl, cl_dates)

# 5. 独立测试集 = 2024-05-01→2025-01-01 的簇（AlignIF: 时间隔离 + TM<0.45 双重隔离）
indep = [(cl, dts) for cl, dts in clusters if any("2024-05-01" < d <= "2025-01-01" for d in dts)]
train_pool = [(cl, dts) for cl, dts in clusters if all(d <= "2024-05-01" for d in dts)]
print(f"独立测试窗簇: {len(indep)}（含 {sum(len(c[0]) for c in indep)} 条链）")
print(f"训练窗簇: {len(train_pool)}（含 {sum(len(c[0]) for c in train_pool)} 条链）")

# 6. 训练窗内按簇划分 train/val/test（AlignIF 比例 399/50/50）
random.seed(42)
random.shuffle(train_pool)
n = len(train_pool)
n_val = max(1, int(n * 0.1))
n_test = n_val
train_cl = train_pool[:n - n_val - n_test]
val_cl = train_pool[n - n_val - n_test:n - n_test]
test_cl = train_pool[n - n_test:]
print(f"训练簇: {len(train_cl)} 条链 {sum(len(c[0]) for c in train_cl)}")
print(f"验证簇: {len(val_cl)} 条链 {sum(len(c[0]) for c in val_cl)}")
print(f"测试簇: {len(test_cl)} 条链 {sum(len(c[0]) for c in test_cl)}")

# 7. 保存划分结果
result = {
    "train": sorted(m for cl, _ in train_cl for m in cl),
    "val": sorted(m for cl, _ in val_cl for m in cl),
    "test": sorted(m for cl, _ in test_cl for m in cl),
    "independent_test": sorted(m for cl, _ in indep for m in cl),
    "n_clusters": {"train": len(train_cl), "val": len(val_cl), "test": len(test_cl), "indep": len(indep)},
}
json.dump(result, open(os.path.join(DATA, "splits.json"), "w"), indent=1)
print("splits.json 已保存")
