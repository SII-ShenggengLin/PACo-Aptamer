#!/usr/bin/env python3
"""第 49 步：A 层 v2 按子集拆表（independent_test 235 链 / zero_shot_targets 93 链）

背景：A 层 v2 评估集 = indep 235 ∪ zero_shot 93（交 39，并 289）。
两个子集是两种独立定义的测试条件（时间+结构双隔离 / 配体级不可见），论文应分别成表。
输入：survey/metrics_{gRNAde,RhoDesign,NAMPNN,AlignIF}_v2.json（per_chain）+
      survey/interface_metrics_v2.json（界面 per_chain）+ data/splits_v2/splits_v2.json
输出：survey/alayer_v2_subsets.json + 打印两张 markdown 表（供 10 号报告）
"""
import os, json

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
SUR = os.path.join(DATA_DIR, "survey")

METHODS = [  # (显示名, metrics 文件, 样本池说明)
    ("RhoDesign", "metrics_RhoDesign_v2.json", "3温度×100"),
    ("AlignIF", "metrics_AlignIF_v2.json", "t=0.1×20"),
    ("NA-MPNN", "metrics_NAMPNN_v2.json", "t=0.1×10"),
    ("gRNAde", "metrics_gRNAde_v2.json", "3温度×100"),
]

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def subset_summary(chains, per_chain, iface_per, method_key):
    """对给定链子集计算 池内best/t=0.1 均值/覆盖/界面
    字段兼容：gRNAde/RhoDesign 为 recovery_best_overall + recovery_t0.1.mean；
              NA-MPNN/AlignIF 为 recovery_best + recovery_t01_mean"""
    best, t01, cov = [], [], 0
    if_best, if_all = [], []
    for c in chains:
        r = per_chain.get(c)
        if r is None:
            continue
        cov += 1
        best.append(r.get("recovery_best", r.get("recovery_best_overall")))
        # t=0.1 均值口径统一为 mean-of-pool（四方法一致；勿用 t0.1 best）
        t01.append(r.get("recovery_t01_mean", r.get("recovery_t0.1", {}).get("mean")))
        iv = iface_per.get(c, {}).get(method_key)
        if iv is not None:
            if_best.append(iv["iface_t0.1"] if "iface_t0.1" in iv else iv.get("iface"))
            if_all.append(iv["overall_t0.1"] if "overall_t0.1" in iv else iv.get("overall"))
    return {
        "n_subset": len(chains), "coverage": cov,
        "recovery_best_mean": mean(best),
        "recovery_best_median": sorted(best)[len(best) // 2] if best else None,
        "recovery_t01_mean": mean(t01),
        "n_iface": len(if_best),
        "iface_recovery": mean(if_best),
        "iface_overall": mean(if_all),
        "iface_minus_overall": (mean(if_best) - mean(if_all)) if if_best else None,
    }

def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    subsets = {"independent_test": sorted(sp["independent_test"]),
               "zero_shot_targets": sorted(sp["zero_shot_targets"])}
    im = json.load(open(os.path.join(SUR, "interface_metrics_v2.json")))
    iface_per = im.get("per_chain", {})

    # 各方法界面字段名不一致（gRNAde/RhoDesign/AlignIF/NAMPNN），统一取
    def iface_key(pc_val):
        for k in ("RhoDesign", "AlignIF", "NAMPNN", "gRNAde", "GRNADE"):
            if k in pc_val:
                return k
        return None

    out = {}
    for sub_name, chains in subsets.items():
        out[sub_name] = {}
        for mname, mfile, pool in METHODS:
            m = json.load(open(os.path.join(SUR, mfile)))
            # 界面字段键名适配
            iface_per_m = {}
            for c, v in iface_per.items():
                k = None
                for cand in (mname, mname.upper(), "NAMPNN" if mname == "NA-MPNN" else "",
                             "GRNADE" if mname == "gRNAde" else "", "gRNAde"):
                    if cand and cand in v:
                        k = cand
                        break
                if k:
                    iface_per_m[c] = {mname: v[k]}
            s = subset_summary(chains, m["per_chain"], iface_per_m, mname)
            s["pool"] = pool
            out[sub_name][mname] = s

    json.dump(out, open(os.path.join(SUR, "alayer_v2_subsets.json"), "w"), indent=1, ensure_ascii=False)

    for sub_name in subsets:
        n = len(subsets[sub_name])
        print(f"\n### A 层 v2 · {sub_name}（{n} 链）\n")
        print("| 方法 | 池内 best | t=0.1 均值 | 界面恢复率 | 界面-整体 | 覆盖 | 样本池 |")
        print("|---|---|---|---|---|---|---|")
        for mname, _, pool in METHODS:
            s = out[sub_name][mname]
            b = f"{100*s['recovery_best_mean']:.1f}%" if s["recovery_best_mean"] is not None else "—"
            t = f"{100*s['recovery_t01_mean']:.1f}%" if s["recovery_t01_mean"] is not None else "—"
            ir = f"{100*s['iface_recovery']:.1f}%（n={s['n_iface']}）" if s["iface_recovery"] is not None else "—"
            d = f"{100*s['iface_minus_overall']:+.1f}pp" if s["iface_minus_overall"] is not None else "—"
            print(f"| {mname} | {b} | {t} | {ir} | {d} | {s['coverage']}/{s['n_subset']} | {pool} |")
        print(f"| 随机基线 | 25% | 25% | — | — | — | — |")
    print("\n写入 survey/alayer_v2_subsets.json", flush=True)

if __name__ == "__main__":
    main()
