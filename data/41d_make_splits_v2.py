#!/usr/bin/env python3
"""41d：新划分 v2（CD-HIT 0.8 + TM0.45 双条件，簇级；用户批准的方案 B）
规则：
- 全池簇按 max_date 分类；巨簇（tRNA，>1000 链）留训练
- 纯老簇（无 2024+ 链）→ 簇级 8:1:1 → train/val/test
- 时间敏感簇（非巨簇）：老链 → test（与 train 双条件 ✓）；2024+ 链 → independent_test
- tRNA 巨簇：老链 → train；2024+ 链 → 弃用（无法干净评估，如实报告）
- zero_shot_targets = (test ∪ indep) 中配体训练期不可见的链（配体隔离标签）
- case_frozen：冻结 2-3 个案例样本（任何评估组外不可用）
- 审计：组大小/交集矩阵/配体验证/弃用清单
产出: splits_v2/splits_v2.json + audit_v2.json
"""
import os, json, random
from collections import Counter

DATA = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
OUT = os.path.join(DATA, "splits_v2")

final_clusters = json.load(open(os.path.join(OUT, "final_clusters.json")))
chains = json.load(open(os.path.join(DATA, "chains_pass.json")))
cands = {c["pdb"]: c for c in json.load(open(os.path.join(DATA, "survey", "candidates.json")))}
bgsu = json.load(open("/tmp/bgsu_pdbs.json"))

def release_date(pdb):
    return bgsu.get(pdb.upper(), {}).get("release_date", "9999-99-99")

def func_ligs(pdb):
    c = cands.get(pdb)
    return c["ligands"].get("other", []) if c else []

TIME_CUTOFF = "2023-12-31"
MEGA_THRESH = 1000   # 巨簇阈值（tRNA 5,373）
random.seed(42)

# 1. 簇分类
clusters = [{"id": cid, "members": mem} for cid, mem in final_clusters.items()]
for cl in clusters:
    cl["dates"] = [release_date(m.split("_")[0]) for m in cl["members"]]
    cl["max_date"] = max(cl["dates"])
    cl["old"] = [m for m, d in zip(cl["members"], cl["dates"]) if d <= TIME_CUTOFF]
    cl["new"] = [m for m, d in zip(cl["members"], cl["dates"]) if d > TIME_CUTOFF]

mega = [cl for cl in clusters if len(cl["members"]) > MEGA_THRESH]
pure_old = [cl for cl in clusters if cl["max_date"] <= TIME_CUTOFF and cl not in mega]
time_sens = [cl for cl in clusters if cl["max_date"] > TIME_CUTOFF and cl not in mega]
print(f"巨簇: {len(mega)}（{sum(len(c['members']) for c in mega)} 链）| 纯老簇: {len(pure_old)}（{sum(len(c['members']) for c in pure_old)}）| 时间敏感簇: {len(time_sens)}", flush=True)

# 2. tRNA 巨簇：老链→train，新链→弃用
train_chains = [m for cl in mega for m in cl["old"]]
discarded = [m for cl in mega for m in cl["new"]]
print(f"巨簇老链→train: {len(train_chains)} | 巨簇新链→弃用: {len(discarded)}", flush=True)

# 3. 纯老簇簇级 8:1:1
random.shuffle(pure_old)
n = len(pure_old)
n_val = max(1, int(n * 0.1))
n_test = n_val
tr_cl = pure_old[:n - n_val - n_test]
va_cl = pure_old[n - n_val - n_test:n - n_test]
te_cl = pure_old[n - n_test:]
def flat(cls): return [m for c in cls for m in c["members"]]
train_chains += flat(tr_cl)
val_chains = flat(va_cl)
test_chains = flat(te_cl)
print(f"纯老簇划分: train {len(tr_cl)}簇 {len(flat(tr_cl))}链 / val {len(va_cl)}簇 {len(flat(va_cl))}链 / test {len(te_cl)}簇 {len(flat(te_cl))}链", flush=True)

# 4. 时间敏感簇：老链→test；新链→indep
test_chains += [m for cl in time_sens for m in cl["old"]]
indep_chains = [m for cl in time_sens for m in cl["new"]]
print(f"时间敏感簇: 老链→test {len([m for cl in time_sens for m in cl['old']])} | 新链→indep {len(indep_chains)}", flush=True)

# 5. 配体隔离 zero_shot（标签叠加在 test ∪ indep）
train_ligs = set(l for m in train_chains for l in func_ligs(m.split("_")[0]))
eval_pool = test_chains + indep_chains
zs_chains = [m for m in eval_pool
             if (ls := func_ligs(m.split("_")[0])) and all(l not in train_ligs for l in ls)]
zs_entries = sorted({m.split("_")[0] for m in zs_chains})
zs_ligs = set(l for e in zs_entries for l in func_ligs(e))
print(f"train 配体种类: {len(train_ligs)} | zero_shot 链: {len(zs_chains)}（条目 {len(zs_entries)}，配体 {len(zs_ligs)} 全不可见）", flush=True)

# 6. 案例冻结：zero_shot 中 2024+ 且适配体尺度（16-200nt）
zsl = {m: next((c for c in chains if c["pdb"] == m.split("_")[0] and c["chain"] == m.split("_")[-1]), {}).get("length", 0) for m in zs_chains}
pref = [m for m in zs_chains if release_date(m.split("_")[0]) > TIME_CUTOFF and 16 <= zsl[m] <= 200]
case_frozen = sorted(pref)[:3] if len(pref) >= 2 else sorted(zs_chains)[:3]
print(f"案例冻结: {case_frozen}", flush=True)

# 7. 汇总写入
splits = {
    "version": "v2-B",
    "rules": "CD-HIT 0.8 + TM0.45 簇级；巨簇(tRNA)留训练；时间敏感簇整簇归评估侧；indep=2024+双条件；zero_shot=配体+双条件",
    "time_cutoff": TIME_CUTOFF,
    "n_clusters": {"train": len(tr_cl) + len(mega), "val": len(va_cl), "test": len(te_cl) + len(time_sens)},
    "train": sorted(train_chains), "val": sorted(val_chains), "test": sorted(test_chains),
    "independent_test": sorted(indep_chains),
    "zero_shot_targets": sorted(zs_chains),
    "case_frozen": case_frozen,
    "discarded": sorted(discarded),
    "cluster_of": {m: cl["id"] for cl in clusters for m in cl["members"]},
}
json.dump(splits, open(os.path.join(OUT, "splits_v2.json"), "w"), indent=1)

# 8. 审计
groups = ["train", "val", "test", "independent_test", "zero_shot_targets"]
gsets = {g: set(splits[g]) for g in groups}
audit = {"group_sizes": {g: len(gsets[g]) for g in groups},
         "discarded": len(discarded)}
audit["pairwise_intersection"] = {f"{g}∩{h}": len(gsets[g] & gsets[h])
                                  for i, g in enumerate(groups) for h in groups[i+1:]}
audit["train_ligands"] = len(train_ligs)
audit["zs_ligands"] = len(zs_ligs)
audit["zs_ligand_overlap_train"] = len(zs_ligs & train_ligs)
audit["case_frozen"] = case_frozen
audit["case_frozen_in_any_group"] = {e: [g for g in groups if e in gsets[g]] for e in case_frozen}
audit["case_frozen_in_train"] = [e for e in case_frozen if e in gsets["train"]]
json.dump(audit, open(os.path.join(OUT, "audit_v2.json"), "w"), indent=1)
print("\n=== 审计 ===")
print(json.dumps(audit, indent=1), flush=True)
print("完成", flush=True)
