#!/usr/bin/env python3
"""104g：77_p03 SS 保真度 INF 的 gRNAde / gRNAde_t01 行用 _cfix 池重算
       → survey/77_p03_gfix.json

口径与 data/77_p0234_analysis.py phase_inf 完全一致：
  池 = _cfix 三温度拼接（len==native 过滤）；gRNAde_t01 = 池前 100；
  参考配对 = s1 缓存 ss_pairs；INF/PPV/Sens/F1 = inf_scores 同款；
  聚合：链内逐样本均值 → 链间 mean/median/bootstrap CI95(n=2000,seed=0)；
  frac_design_inf_ge0.9、inf_by_identity 分箱、spearman(ident, inf) 同款。
其余方法行（PACo/native/random/...）从 77_p03_ss_fidelity.json 原样拷贝。
输入：survey/104_fold_cache_grnade_cfix.json（104e 产物）
环境：rna_if（纯 numpy/scipy）
"""
import os, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
CFIX_DIR = os.path.join(SUR, "grnade_out_btest_v2_cfix")
TEMPS = ("0.1", "0.5", "1.0")
N_PER_TEMP = 100
OUT = os.path.join(SUR, "77_p03_gfix.json")


def pairs_from_dbn(dbn):
    pairs = set()
    stacks = {"(": [], "[": [], "{": [], "<": []}
    close = {")": "(", "]": "[", "}": "{", ">": "<"}
    for i, c in enumerate(dbn):
        if c in stacks:
            stacks[c].append(i)
        elif c in close:
            if stacks[close[c]]:
                j = stacks[close[c]].pop()
                pairs.add((min(i, j), max(i, j)))
    return pairs


def inf_scores(pred_pairs, ref_pairs):
    NR, NP, TP = len(ref_pairs), len(pred_pairs), len(pred_pairs & ref_pairs)
    if NR == 0 and NP == 0:
        inf = 1.0
    else:
        inf = TP / np.sqrt(NR * NP) if NR * NP > 0 else 0.0
    ppv = TP / NP if NP else (1.0 if NR == 0 else 0.0)
    sen = TP / NR if NR else (1.0 if NP == 0 else 0.0)
    f1 = 2 * TP / (NR + NP) if (NR + NP) else 1.0
    return {"inf": inf, "ppv": ppv, "sens": sen, "f1": f1}


def boot_ci(x, n=2000, seed=0):
    if len(x) < 3:
        return None
    r = np.random.RandomState(seed)
    m = [r.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def agg(chains):
    """chains: stem → [逐样本 {inf,f1,ppv,sens,ident}]；phase_inf 汇总同款"""
    out = {}
    for key in ("inf", "f1", "ppv", "sens"):
        vals = [np.mean([s[key] for s in scs]) for scs in chains.values()]
        out[key] = {"n_chains": len(vals), "mean": float(np.mean(vals)),
                    "ci95": boot_ci(vals), "median": float(np.median(vals))}
    out["frac_design_inf_ge0.9"] = float(np.mean(
        [s["inf"] >= 0.9 for scs in chains.values() for s in scs]))
    id_edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    by_ident = {}
    for lo, hi in zip(id_edges[:-1], id_edges[1:]):
        vals = [s["inf"] for scs in chains.values() for s in scs
                if lo <= s.get("ident", -1) < hi]
        if len(vals) >= 20:
            by_ident[f"{lo:.1f}-{hi:.1f}"] = {"n": len(vals), "inf_mean": float(np.mean(vals))}
    out["inf_by_identity"] = by_ident
    ids = [s["ident"] for scs in chains.values() for s in scs]
    infs = [s["inf"] for scs in chains.values() for s in scs]
    if len(ids) > 30 and np.std(ids) > 0:
        from scipy.stats import spearmanr
        out["spearman_ident_vs_inf"] = float(spearmanr(ids, infs).statistic)
    return out


def main():
    cache = json.load(open(os.path.join(SUR, "104_fold_cache_grnade_cfix.json")))
    ref_pairs = {s: {tuple(p) for p in v} for s, v in cache["ref_pairs"].items()}
    per_chain = cache["per_chain"]

    chains_full, chains_t01 = {}, {}
    for stem, rec in per_chain.items():
        native = rec["native"]
        fmap = rec["fold"]
        rp = ref_pairs[stem]
        pool = []
        for t in TEMPS:
            fp = os.path.join(CFIX_DIR, f"{stem}.{t}.seqs")
            if os.path.exists(fp):
                pool += [l.strip() for l in open(fp)
                         if l.strip() and len(l.strip()) == len(native)]
        if not pool:
            continue

        def scores(seqs):
            sc = []
            for s in seqs:
                if s not in fmap:
                    continue
                r = inf_scores(pairs_from_dbn(fmap[s][0]), rp)
                r["ident"] = sum(1 for a, b in zip(s, native) if a == b) / len(native)
                sc.append(r)
            return sc

        sc_full = scores(pool)
        if sc_full:
            chains_full[stem] = sc_full
        if len(pool) >= N_PER_TEMP:
            sc_t01 = scores(pool[:N_PER_TEMP])
            if sc_t01:
                chains_t01[stem] = sc_t01

    agg_full = agg(chains_full)
    agg_t01 = agg(chains_t01)

    orig = json.load(open(os.path.join(SUR, "77_p03_ss_fidelity.json")))
    out = {"methods": dict(orig["methods"])}
    out["methods"]["gRNAde"] = agg_full
    out["methods"]["gRNAde_t01"] = agg_t01
    out["cfix_note"] = ("gRNAde/gRNAde_t01 行 = grnade_out_btest_v2_cfix 池 RNAfold 回折"
                        "按 77 号 phase_inf 口径重算（2026-10-16，104e 折叠缓存）；"
                        "其余行拷贝自 77_p03_ss_fidelity.json。")
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    for m, a in (("gRNAde", agg_full), ("gRNAde_t01", agg_t01)):
        o = orig["methods"][m]
        print(f"[104g] {m}: INF {o['inf']['mean']:.4f}->{a['inf']['mean']:.4f} "
              f"F1 {o['f1']['mean']:.4f}->{a['f1']['mean']:.4f} "
              f">=0.9 {o['frac_design_inf_ge0.9']:.4f}->{a['frac_design_inf_ge0.9']:.4f} "
              f"(n={a['inf']['n_chains']})")


if __name__ == "__main__":
    main()
