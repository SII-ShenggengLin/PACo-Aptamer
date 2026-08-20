#!/usr/bin/env python3
"""我们的链集合 vs AlignIF 官方数据对照 — 复现验证第 6 步
对照内容：
1. 链集合交叠（我们过滤出的 vs 官方 split 7,956）
2. 聚类簇数 vs 官方 499
3. 划分逻辑一致性（时间隔离）
"""
import json, os, torch

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
AL = f"{DATA}/alignif"

# 1. 官方链集合
split = torch.load(f"{AL}/data/split.pt", map_location='cpu', weights_only=False)
official = set(split['train']) | set(split['val']) | set(split['test']) | set(split['independent'])
print(f"官方链总数: {len(official)}")

# 2. 我们的链集合
chains = json.load(open(f"{DATA}/chains_pass.json"))
ours = {f"{c['pdb']}_{c['chain']}" for c in chains}
print(f"我们过滤出的链数: {len(ours)}")

# 3. 交叠（大小写不敏感匹配）
o_lower = {x.lower(): x for x in official}
ours_lower = {x.lower() for x in ours}
overlap = ours_lower & set(o_lower.keys())
ours_only = ours_lower - set(o_lower.keys())
official_only = set(o_lower.keys()) - ours_lower
print(f"\n=== 对照 ===")
print(f"我们∩官方: {len(overlap)}")
print(f"我们独有: {len(ours_only)}")
print(f"官方独有: {len(official_only)}")
print(f"覆盖率(官方∩我们/官方): {len(overlap)/len(official)*100:.1f}%")
print(f"精确率(我们∩官方/我们): {len(overlap)/len(ours)*100:.1f}%")

# 4. 差异样本示例
print(f"\n我们独有示例(前10): {sorted(ours_only)[:10]}")
print(f"官方独有示例(前10): {sorted(official_only)[:10]}")

# 5. 官方独有链的时间分布（判断是否主要是测试窗/独立集）
bgsu = json.load(open("/tmp/bgsu_pdbs.json"))
def rd(name):
    pdb = name.split("_")[0].upper()
    return bgsu.get(pdb, {}).get("release_date", "?")
from collections import Counter
c = Counter(rd(x) for x in official_only if isinstance(x, str) and rd(x)[:4] != "?")
print(f"\n官方独有链的释放年份分布: {dict(sorted(c.items()))}")

json.dump({
    "overlap": sorted(overlap), "ours_only": sorted(ours_only),
    "official_only": sorted(official_only),
}, open(f"{DATA}/compare_result.json", "w"), indent=1)
print("\ncompare_result.json 已保存")
