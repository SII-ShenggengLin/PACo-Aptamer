#!/usr/bin/env python3
"""104a：63 号主表 gRNAde 行用 _cfix 池重算 → survey/63_main_table_gfix.json

口径与 data/63_btest_main_table.py 完全一致（native=canonical_v2，'N' 位跳过，
池内 best=3 温度全池，t01=t=0.1 池 mean，长度不符过滤），仅把
grnade_out_btest_v2 换成 grnade_out_btest_v2_cfix。
其余方法行从 survey/btest_v2_main_table.json 原样拷贝。
"""
import os, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
TEMPS = ("0.1", "0.5", "1.0")


def atomic_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    test = sorted(sp["test"])
    pc = {}
    for stem in test:
        native = canon[stem]["seq"]
        npos = [i for i, c in enumerate(native) if c != "N"]
        alls, t01 = [], []
        for t in TEMPS:
            fp = os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.{t}.seqs")
            seqs = [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []
            seqs = [s for s in seqs if len(s) == len(native)]
            alls += seqs
            if t == "0.1":
                t01 = seqs
        if not alls or not t01 or not npos:
            continue
        rec_best = max(sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in alls)
        rec_t01 = float(np.mean([sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in t01]))
        pc[stem] = {"len": len(native), "recovery_best": rec_best,
                    "recovery_t01_mean": rec_t01, "n_pool": len(alls)}

    orig = json.load(open(os.path.join(SUR, "btest_v2_main_table.json")))
    out = {"table": dict(orig["table"]), "per_chain": dict(orig["per_chain"])}
    best = float(np.mean([r["recovery_best"] for r in pc.values()]))
    med = float(np.median([r["recovery_best"] for r in pc.values()]))
    t01m = float(np.mean([r["recovery_t01_mean"] for r in pc.values()]))
    out["table"]["gRNAde"] = {"recovery_best_mean": best, "recovery_best_median": med,
                              "recovery_t01_mean": t01m,
                              "coverage": f"{len(pc)}/{len(test)}"}
    out["per_chain"]["gRNAde"] = pc
    out["cfix_note"] = ("gRNAde 行 = grnade_out_btest_v2_cfix（26v2 解码 C/G 互换修正后）"
                        "按 63 号口径重算（2026-10-16）；其余方法行拷贝自 btest_v2_main_table.json。"
                        "修正前对照：best 0.5015188428008936 / t01 0.31211265654392867。")
    atomic_json(out, os.path.join(SUR, "63_main_table_gfix.json"))
    print(f"[104a] gRNAde(cfix) best={best:.4f} med={med:.4f} t01={t01m:.4f} n={len(pc)}")
    print(f"       对照 orig    best=0.5015 t01=0.3121")


if __name__ == "__main__":
    main()
