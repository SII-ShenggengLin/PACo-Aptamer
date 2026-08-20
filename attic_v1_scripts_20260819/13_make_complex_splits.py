#!/usr/bin/env python3
"""复合物池正式划分（第 13 步）：训练/验证/测试 + 独立测试 + zero-shot 靶标子集

三重防泄漏（对齐 04 文档 10 条）：
1. 结构簇级 holdout：条目按 RNA 链 qTMclust 簇（TM0.45）聚合，按簇整体 8:1:1 划分
   （同一簇内任何条目不得跨集合 —— 簇级隔离）
2. 时间切分：训练/验证/测试只用 ≤2023-12-31 释放的复合物；
   独立测试集 = 2024-01-01 后释放（时间隔离）
3. 配体级 zero-shot：另设 zero-shot 靶标子集，其配体在训练集完全不可见
   （靶标条件化方法的专项评估；主基准不要求配体隔离，保证所有对比方法公平）

验证集保障：簇级 8:1:1 划分，val 预计 ~80 条目（比例 ~10%）。
"""
import json, os
from collections import defaultdict, Counter

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
OUT_DIR = os.path.join(DATA_DIR, "survey")

TIME_CUTOFF = "2023-12-31"   # 训练/验证/测试 时间上界
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def load_clusters():
    """qTMclust 输出 → {chain: cluster_id}（strip .pdb 后缀）"""
    cluster_of_chain = {}
    with open(os.path.join(DATA_DIR, "our_qTMclust_out.txt")) as f:
        for cid, line in enumerate(f):
            for tok in line.strip().split("\t"):
                tok = tok.strip()
                if tok:
                    cluster_of_chain[tok[:-4]] = cid   # strip .pdb
    return cluster_of_chain

def main():
    final = json.load(open(os.path.join(OUT_DIR, "final_complexes.json")))
    chains = json.load(open(os.path.join(DATA_DIR, "chains_pass.json")))
    cluster_of_chain = load_clusters()
    print(f"复合物条目: {len(final)}  链簇映射: {len(cluster_of_chain)} 条链", flush=True)

    # 条目 → 该条目在池内 RNA 链的簇集合
    chains_by_pdb = defaultdict(list)
    for c in chains:
        chains_by_pdb[c["pdb"]].append(c)

    entry_clusters = {}
    for f in final:
        cs = set()
        for c in chains_by_pdb.get(f["pdb"], []):
            if (c["pdb"] + "_" + c["chain"]) in cluster_of_chain:
                cs.add(cluster_of_chain[c["pdb"] + "_" + c["chain"]])
        entry_clusters[f["pdb"]] = sorted(cs)

    # ---- 时间分层 ----
    old = [f for f in final if f["release_date"] and f["release_date"] <= TIME_CUTOFF]
    new = [f for f in final if f["release_date"] and f["release_date"] > TIME_CUTOFF]
    nodate = [f for f in final if not f["release_date"]]
    print(f"≤{TIME_CUTOFF}: {len(old)} | 2024+: {len(new)} | 无日期: {len(nodate)}", flush=True)

    # ---- 簇级 8:1:1（old 部分） ----
    # 聚合：簇 → 条目列表；无簇条目的条目自成"虚拟簇"
    cluster_entries = defaultdict(list)
    for f in old:
        cids = entry_clusters.get(f["pdb"], [])
        key = tuple(cids) if cids else ("novel", f["pdb"])
        cluster_entries[key].append(f)

    cluster_items = sorted(cluster_entries.items(), key=lambda kv: -len(kv[1]))
    total_old = len(old)
    n_clusters = len(cluster_items)
    print(f"结构簇(含虚拟簇): {n_clusters} | old 条目总数: {total_old}", flush=True)

    # 贪心按条目数平衡：val/test 各取 ~10% 条目。
    # 巨簇（>15 条目，常见折叠家族如核糖体/SAM 开关）留训练集——避免 val/test 被单一折叠主导，
    # 也保证训练集覆盖代表性家族；val/test 簇数下限 5（结构多样性，防验证失真）。
    MAX_SINGLE = 15          # 巨簇阈值
    MIN_CLUSTERS = 5         # val/test 簇数下限
    def greedy(entries_target_frac, excluded, max_single=MAX_SINGLE, min_clusters=MIN_CLUSTERS):
        count, picked = 0, set()
        for k, fs in cluster_items:
            if k in excluded:
                continue
            if len(fs) > max_single:
                continue
            picked.add(k)
            count += len(fs)
            if count >= total_old * entries_target_frac and len(picked) >= min_clusters:
                break
        return picked
    val_clusters = greedy(VAL_RATIO, set())
    test_clusters = greedy(TEST_RATIO, val_clusters)
    train_clusters = set(k for k, _ in cluster_items) - val_clusters - test_clusters
    print(f"val {len(val_clusters)} 簇 / test {len(test_clusters)} 簇 / train {len(train_clusters)} 簇", flush=True)

    train_entries = [f for k, fs in cluster_items if k in train_clusters for f in fs]
    val_entries = [f for k, fs in cluster_items if k in val_clusters for f in fs]
    test_entries = [f for k, fs in cluster_items if k in test_clusters for f in fs]
    indep_entries = new

    # ---- zero-shot 靶标子集 ----
    # 训练集配体集合
    train_ligs = set()
    for f in train_entries:
        train_ligs.update(f["bound_ligands"])
    # 从 test + indep 中挑配体训练期完全不可见的条目
    zs_entries = []
    for f in test_entries + indep_entries:
        if all(l not in train_ligs for l in f["bound_ligands"]):
            zs_entries.append(f)
    # 同时要求条目配体种类 ≥1 且 RNA 链在 16-512（训练池口径，保证可评估）
    zs_entries = [f for f in zs_entries if f["in_train_pool"]]

    print(f"train: {len(train_entries)} 条目 | val: {len(val_entries)} | test: {len(test_entries)} | indep: {len(indep_entries)}", flush=True)
    print(f"训练集配体: {len(train_ligs)} 种 | zero-shot 子集: {len(zs_entries)} 条目", flush=True)

    # ---- 交叉验证报告 ----
    def ligs(entries):
        s = set()
        for f in entries:
            s.update(f["bound_ligands"])
        return s
    def clusters_of(entries):
        s = set()
        for f in entries:
            s.update(entry_clusters.get(f["pdb"], []))
        return s
    print(f"\n集合配体数: train {len(ligs(train_entries))} / val {len(ligs(val_entries))} / test {len(ligs(test_entries))} / indep {len(ligs(indep_entries))}")
    print(f"集合结构簇数: train {len(clusters_of(train_entries))} / val {len(clusters_of(val_entries))} / test {len(clusters_of(test_entries))}")

    # 日期范围
    for name, entries in [("train", train_entries), ("val", val_entries), ("test", test_entries), ("indep", indep_entries)]:
        dates = sorted(f["release_date"] for f in entries if f["release_date"])
        if dates:
            print(f"{name} 日期范围: {dates[0]} ~ {dates[-1]}")

    # 配体交叉检查（泄漏检查）
    for name in ["val", "test"]:
        overlap = ligs(entries := locals()[f"{name}_entries"]) & ligs(train_entries)
        print(f"[配体交叉] {name} ∩ train 配体: {len(overlap)} 种 (允许，主基准以结构簇隔离为准)")

    result = {
        "time_cutoff": TIME_CUTOFF,
        "n_val_clusters": len(val_clusters), "n_test_clusters": len(test_clusters),
        "train": [f["pdb"] for f in train_entries],
        "val": [f["pdb"] for f in val_entries],
        "test": [f["pdb"] for f in test_entries],
        "independent_test": [f["pdb"] for f in indep_entries],
        "zero_shot_targets": [f["pdb"] for f in zs_entries],
        "entry_info": {f["pdb"]: {
            "release_date": f["release_date"], "resolution": f["resolution"],
            "bound_ligands": f["bound_ligands"], "n_rna": f["n_rna"],
            "n_prot": f["n_prot"], "clusters": entry_clusters.get(f["pdb"], []),
            "in_train_pool": f["in_train_pool"],
        } for f in final},
    }
    json.dump(result, open(os.path.join(OUT_DIR, "complex_splits.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n结果写入: {OUT_DIR}/complex_splits.json", flush=True)

    # 详细统计表
    print("\n===== 划分统计 =====")
    print(f"{'集合':<18}{'条目':>6}{'配体种':>8}{'结构簇':>8}")
    for name, entries in [("train", train_entries), ("val", val_entries), ("test", test_entries),
                          ("independent", indep_entries), ("zero_shot", zs_entries)]:
        print(f"{name:<18}{len(entries):>6}{len(ligs(entries)):>8}{len(clusters_of(entries)):>8}")

if __name__ == "__main__":
    main()
