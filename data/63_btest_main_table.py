#!/usr/bin/env python3
"""第 63 步：B 层 v2 test 统一指标汇总（论文主表唯一真值口径）

原则：
- native 一律 = canonical_v2.json seq；'N'（未指明碱基）位跳过评估
- gRNAde/RhoDesign：从 survey/{grnade,rhodesign}_out_btest_v2 的 3 温度样本池重算
  （池内 best = 3 温度全池；t=0.1 均值 = t=0.1 样本池 mean-of-pool）
- PACo：437 条一致链沿用 metrics_PACo_v2_main.json（其 native 已与 canonical 一致）；
  40 条修复链从 survey/paco_fix_pool 重算（含 'N' 跳过）
- NA-MPNN：61 号（含缺 C1' 坐标链的有效位口径）；AlignIF：62 号（'-' 跳过）
  ——两脚本已按 canonical 真值计算，本脚本仅汇总
输出：survey/btest_v2_main_table.json + 打印主表
"""
import os, json
import numpy as np

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
SUR = os.path.join(DATA_DIR, "survey")

TEMPS = ("0.1", "0.5", "1.0")


def load_pool(out_dir, stem, n100=True):
    """返回 {temp: [seqs]}；文件不存在给空"""
    pool = {}
    for t in TEMPS:
        fp = os.path.join(out_dir, f"{stem}.{t}.seqs")
        pool[t] = [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []
    return pool


def rec_chain(pool, native):
    """池内 best（全池）+ t=0.1 均值；'N' 位跳过；长度不符样本过滤"""
    npos = [i for i, c in enumerate(native) if c != "N"]
    alls = [s for t in TEMPS for s in pool[t] if len(s) == len(native)]
    t01 = [s for s in pool["0.1"] if len(s) == len(native)]
    if not alls or not t01 or not npos:
        return None
    rec_best = max(sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in alls)
    rec_t01 = float(np.mean([sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in t01]))
    return rec_best, rec_t01, len(alls)


def main():
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test = sorted(sp["test"])
    out = {}

    for tag, out_dir in [("gRNAde", "grnade_out_btest_v2"), ("RhoDesign", "rhodesign_out_btest_v2")]:
        pc = {}
        for stem in test:
            native = canon[stem]["seq"]
            pool = load_pool(os.path.join(SUR, out_dir), stem)
            r = rec_chain(pool, native)
            if r is None:
                continue
            pc[stem] = {"len": len(native), "recovery_best": r[0], "recovery_t01_mean": r[1], "n_pool": r[2]}
        out[tag] = pc

    # PACo：437 沿用 + 40 修复链从 pool 重算
    # 【2026-08-17 补丁】PACo metrics/修复池路径支持环境变量覆盖（换 ckpt 重建主表用）
    tm = json.load(open(os.environ.get(
        "PACO_METRICS", os.path.join(SUR, "metrics_PACo_v2_main.json"))))
    pc_paco = {c: v for c, v in tm["per_chain"].items()}
    fix_dir = os.environ.get("PACO_FIXPOOL", os.path.join(SUR, "paco_fix_pool"))
    if os.path.isdir(fix_dir):
        for f in os.listdir(fix_dir):
            stem = f[:-5]
            d = json.load(open(os.path.join(fix_dir, f)))
            native = canon[stem]["seq"]
            assert d["native"] == native, f"{stem} 修复 pool 的 native 与 canonical 不一致"
            npos = [i for i, c in enumerate(native) if c != "N"]
            pool = d["pool"]
            recs = [sum(1 for i in npos if i < len(s) and s[i] == native[i]) / len(npos)
                    for s in pool if len(s) == len(native)]
            pc_paco[stem] = {"len": len(native), "recovery_best": max(recs),
                             "recovery_t01_mean": float(np.mean(recs[:100])), "n_pool": len(recs)}
    out["PACoAptamer"] = pc_paco

    # NA-MPNN / AlignIF：61/62 号产物（canonical 口径）
    for tag, mf in [("NA-MPNN", "metrics_NAMPNN_btest_v2.json"), ("AlignIF", "metrics_AlignIF_btest_v2.json")]:
        fp = os.path.join(SUR, mf)
        if os.path.exists(fp):
            out[tag] = json.load(open(fp))["per_chain"]

    # 汇总打印
    print("\n===== B 层 v2 主表（splits_v2 test 477 链，canonical 真值） =====\n")
    print("| 方法 | 池内 best | t=0.1 均值 | 覆盖 |")
    print("|---|---|---|---|")
    table = {}
    for tag in ["PACoAptamer", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]:
        pc = out.get(tag, {})
        if not pc:
            print(f"| {tag} | （缺） | | |")
            continue
        best = float(np.mean([r["recovery_best"] for r in pc.values()]))
        med = float(np.median([r["recovery_best"] for r in pc.values()]))
        t01 = float(np.mean([r["recovery_t01_mean"] for r in pc.values()]))
        table[tag] = {"recovery_best_mean": best, "recovery_best_median": med,
                      "recovery_t01_mean": t01, "coverage": f"{len(pc)}/{len(test)}"}
        print(f"| {tag} | {100*best:.2f}%（中位 {100*med:.1f}） | {100*t01:.2f}% | {len(pc)}/{len(test)} |")
    print(f"| 随机基线 | 25% | 25% | — |")
    json.dump({"table": table, "per_chain": out},
              open(os.path.join(SUR, "btest_v2_main_table.json"), "w"), indent=1, ensure_ascii=False)
    print("\n写入 survey/btest_v2_main_table.json", flush=True)


if __name__ == "__main__":
    main()
