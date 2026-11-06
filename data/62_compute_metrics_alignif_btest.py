#!/usr/bin/env python3
"""第 62 步：AlignIF B 层 v2 test 指标（canonical native 为唯一真值）

- native = canonical_v2.json seq（'N' 位跳过）
- 输出 '-'（缺坐标占位）位跳过；长度不符样本过滤
- 口径：t=0.1 × 20 池内 best + t=0.1 均值
输出：survey/metrics_AlignIF_btest_v2.json
"""
import os, json
import numpy as np

DATA_DIR = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF/data"
AI_OUT = os.path.join(DATA_DIR, "survey", "alignif_out_btest_v2")


def main():
    canon = json.load(open(os.path.join(DATA_DIR, "survey", "canonical_v2.json")))
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test = sorted(sp["test"])
    results = {}
    for stem in test:
        fa = os.path.join(AI_OUT, stem, "out.fasta")
        if not os.path.exists(fa):
            continue
        native = canon[stem]["seq"]
        npos = set(i for i, c in enumerate(native) if c != "N")
        lines = [l.strip() for l in open(fa) if l.strip()]
        seqs = [lines[j + 1] for j, l in enumerate(lines)
                if l.startswith(">id:") and j + 1 < len(lines)
                and len(lines[j + 1]) == len(native)
                and all(c in "ACGU-" for c in lines[j + 1])]
        recs = []
        for s in seqs:
            pos = [i for i in npos if s[i] != '-']
            if not pos:
                continue
            recs.append(sum(1 for i in pos if s[i] == native[i]) / len(pos))
        if not recs:
            continue
        results[stem] = {
            "len": len(native),
            "recovery_best": max(recs),
            "recovery_t01_mean": float(np.mean(recs)),
            "n_samples": len(recs),
        }
    summary = {
        "n_chains": len(results),
        "coverage": f"{len(results)}/{len(test)}",
        "recovery_best_mean": float(np.mean([r["recovery_best"] for r in results.values()])),
        "recovery_best_median": float(np.median([r["recovery_best"] for r in results.values()])),
        "recovery_t01_mean_mean": float(np.mean([r["recovery_t01_mean"] for r in results.values()])),
        "note": "canonical native 真值；'N'/'-' 位跳过",
    }
    json.dump({"summary": summary, "per_chain": results},
              open(os.path.join(DATA_DIR, "survey", "metrics_AlignIF_btest_v2.json"), "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
