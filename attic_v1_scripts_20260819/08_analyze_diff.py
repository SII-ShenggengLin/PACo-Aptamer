#!/usr/bin/env python3
"""官方独有单链差异分类分析"""
import json, os, importlib.util, sys
from collections import Counter

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
spec = importlib.util.spec_from_file_location('pc', os.path.join(DATA, '02_parse_chains.py'))
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)

res = json.load(open(os.path.join(DATA, "compare_result.json")))
official_only = res['official_only']
single = [x for x in official_only if '-' not in x]

stats = Counter()
examples = {}
for x in single:
    pdb = x.split('_')[0].upper()
    chain = x.split('_')[1]
    p = os.path.join(DATA, "rcsb_pdb", f"{pdb}.pdb")
    if not os.path.exists(p) or os.path.getsize(p) <= 5000:
        stats['文件缺失/占位'] += 1
        examples.setdefault('文件缺失/占位', []).append(x)
        continue
    try:
        r, chains = pc.parse_pdb(p)
    except Exception:
        stats['解析错误'] += 1
        continue
    match = None
    for c in chains:
        if c.lower() == chain.lower():
            match = c
            break
    if match is None:
        stats['链ID不存在'] += 1
        examples.setdefault('链ID不存在', []).append(x)
        continue
    ct = pc.chain_type(chains[match])
    n = len(chains[match])
    if ct == 'rna':
        if r is None or r > 4.0:
            stats['RNA但分辨率>4或无'] += 1
        elif n < 16:
            stats['RNA但长度<16'] += 1
        else:
            stats['RNA且合格(应为匹配错误)'] += 1
            examples.setdefault('RNA且合格', []).append(x)
    else:
        stats[f'非rna({ct})'] += 1
        examples.setdefault(f'非rna({ct})', []).append(x)

print("=== 官方独有单链分类统计 ===")
for k, v in stats.most_common():
    print(f"{k}: {v}")
print("\n=== 各类示例 ===")
for k, v in examples.items():
    print(f"{k}: {v[:5]}")
