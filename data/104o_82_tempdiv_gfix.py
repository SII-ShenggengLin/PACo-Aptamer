#!/usr/bin/env python3
"""104o：82 号温度-多样性（Fig5a 数据源 82_temp_diversity.json）的 gRNAde 行
       用 _cfix 分温度池重算 → survey/82_temp_diversity_gfix.json

口径与 data/82_diversity_gc.py phase_temp 完全一致：
  每温度池 = _cfix {stem}.{t}.seqs（len==native 过滤，<10 跳过）；
  rec = 池内逐样本同一性均值；div = 1 − 平均两两一致度（RandomState(0) 抽 ≤50 条）；
  uniq = 唯一序列占比。native = s1 缓存 seq。
其余方法行（PACo/RhoDesign）从 82_temp_diversity.json 原样拷贝。
环境：mpl372_repro（torch 读 s1 缓存）
"""
import os, sys, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
TEMPS = [0.1, 0.5, 1.0]
OUT = os.path.join(SUR, "82_temp_diversity_gfix.json")


def pool_diversity(seqs, k=50):
    if len(seqs) < 2:
        return None, None
    uniq = len(set(seqs)) / len(seqs)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(seqs))[:min(k, len(seqs))]
    sub = [seqs[i] for i in idx]
    ids = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            a, b = sub[i], sub[j]
            if len(a) != len(b):
                continue
            ids.append(sum(1 for x, y in zip(a, b) if x == y) / len(a))
    return (float(1 - np.mean(ids)) if ids else None), float(uniq)


def main():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from PACoAptamer.data_utils import load_s1
    sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
    stems = list(sp["test"])
    data = load_s1(set(stems), cache=os.path.join(ROOT, "data", "paco_s1_v2.pt"))

    per_temp = {t: {"rec": [], "div": [], "uniq": []} for t in TEMPS}
    for stem in stems:
        if stem not in data:
            continue
        native = data[stem]["seq"]
        for t in TEMPS:
            fp = os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.{t}.seqs")
            if not os.path.exists(fp):
                continue
            pool = [l.strip() for l in open(fp) if l.strip() and len(l.strip()) == len(native)]
            if len(pool) < 10:
                continue
            recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
            div, uniq = pool_diversity(pool)
            per_temp[t]["rec"].append(float(np.mean(recs)))
            if div is not None:
                per_temp[t]["div"].append(div)
            per_temp[t]["uniq"].append(uniq)

    gr = {}
    for t, pt in per_temp.items():
        gr[f"t={t}"] = {k: {"mean": float(np.mean(v)) if v else None, "n": len(v)}
                        for k, v in pt.items()}

    orig = json.load(open(os.path.join(SUR, "82_temp_diversity.json")))
    out = dict(orig)
    out["gRNAde"] = gr
    out["cfix_note"] = ("gRNAde 行 = grnade_out_btest_v2_cfix 分温度池按 82 号 phase_temp 口径"
                        "重算（2026-10-16）；PACo/RhoDesign 行拷贝自 82_temp_diversity.json。")
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    for t in TEMPS:
        o, n = orig["gRNAde"][f"t={t}"], gr[f"t={t}"]
        print(f"[104o] t={t}: rec {o['rec']['mean']:.4f}->{n['rec']['mean']:.4f} "
              f"div {o['div']['mean']:.4f}->{n['div']['mean']:.4f} "
              f"uniq {o['uniq']['mean']:.4f}->{n['uniq']['mean']:.4f}")


if __name__ == "__main__":
    main()
