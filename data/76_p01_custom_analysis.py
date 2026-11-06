#!/usr/bin/env python3
"""第 76 步：P0-1 补充分析（75 号 report 之外的两个定制口径）

1. 温度分层跟随：嘌呤家族 P(native@spec)/P(donor@spec) 随采样温度段变化
2. U↔C 互换位分析：经典 U74/C74 型特异性替换位上的条件偏好（最锐利检验）

输出 survey/75_p01_custom.json；主结论见 analysis/14 号报告。
"""
import os, sys, json
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
sys.path.insert(ROOT and 0, os.path.join(ROOT, "data"))
import importlib.util
spec = importlib.util.spec_from_file_location("m75", os.path.join(ROOT, "data", "75_p01_ligand_swap.py"))
m75 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m75)

feats = torch.load(os.path.join(SUR, "75_feats.pt"), weights_only=False)
meta = json.load(open(os.path.join(SUR, "75_meta.json")))


def load_pools(model):
    out = {}
    for fn in os.listdir(os.path.join(SUR, "75_pools")):
        if fn.startswith(model + "__"):
            out[fn[len(model) + 2:-5]] = json.load(open(os.path.join(SUR, "75_pools", fn)))
    return out


def spec_of(stem, donor, uc_only):
    sp = m75.spec_pairs(stem, donor, feats, meta)
    if uc_only:
        sp = [x for x in sp if {x[1], x[2]} == {"U", "C"}]
    return sp


def freqs(pool, native, sp):
    pn, pd, n = np.zeros(len(sp)), np.zeros(len(sp)), 0
    for s in pool:
        if len(s) != len(native):
            continue
        n += 1
        for k, (p, bn, bd) in enumerate(sp):
            if s[p] == bn:
                pn[k] += 1
            if bd is not None and s[p] == bd:
                pd[k] += 1
    return pn / max(n, 1), pd / max(n, 1)


def temp_stratified(model, fam="purine_riboswitch"):
    pools = load_pools(model)
    res = {}
    for seg_name, frac in [("t0.1", (0, 1/3)), ("t0.5", (1/3, 2/3)), ("t1.0", (2/3, 1)), ("all", (0, 1))]:
        dn, dd = [], []
        for stem, rec in pools.items():
            if rec["family"] != fam:
                continue
            native = feats[stem]["seq"]
            for cname in rec["pools"]:
                if not cname.startswith("swap_"):
                    continue
                sp = spec_of(stem, cname[5:], uc_only=False)
                if not sp:
                    continue
                npool = len(rec["pools"]["true"])
                a, b = int(frac[0] * npool), int(frac[1] * npool)
                pn_s, pd_s = freqs(rec["pools"][cname][a:b], native, sp)
                pn_t, pd_t = freqs(rec["pools"]["true"][a:b], native, sp)
                dn.extend((pn_s - pn_t).tolist())
                dd.extend((pd_s - pd_t).tolist())
        if len(dn) >= 10:
            res[seg_name] = {
                "n_obs": len(dn),
                "d_p_native_swap_minus_true": float(np.mean(dn)),
                "p_native_wilcoxon_less": float(wilcoxon(dn, alternative="less").pvalue),
                "d_p_donor_swap_minus_true": float(np.mean(dd)),
                "p_donor_wilcoxon_greater": float(wilcoxon(dd, alternative="greater").pvalue)}
    return res


def uc_positions(model):
    pools = load_pools(model)
    rows = []
    for stem, rec in pools.items():
        if rec["family"] != "purine_riboswitch":
            continue
        native = feats[stem]["seq"]
        for cname in rec["pools"]:
            if not cname.startswith("swap_"):
                continue
            sp = spec_of(stem, cname[5:], uc_only=True)
            if not sp:
                continue
            pn_s, pd_s = freqs(rec["pools"][cname], native, sp)
            pn_t, pd_t = freqs(rec["pools"]["true"], native, sp)
            for k, (p, bn, bd) in enumerate(sp):
                rows.append({"stem": stem, "pos": p, "native_base": bn, "donor_base": bd,
                             "p_native_true": float(pn_t[k]), "p_native_swap": float(pn_s[k]),
                             "p_donor_true": float(pd_t[k]), "p_donor_swap": float(pd_s[k])})
    out = {"n_obs": len(rows), "n_chains": len({r["stem"] for r in rows})}
    if rows:
        dn = [r["p_native_swap"] - r["p_native_true"] for r in rows]
        dd = [r["p_donor_swap"] - r["p_donor_true"] for r in rows]
        out.update({
            "p_native_true": float(np.mean([r["p_native_true"] for r in rows])),
            "p_native_swap": float(np.mean([r["p_native_swap"] for r in rows])),
            "d_p_native": float(np.mean(dn)),
            "p_native_wilcoxon_less": float(wilcoxon(dn, alternative="less").pvalue),
            "p_donor_true": float(np.mean([r["p_donor_true"] for r in rows])),
            "p_donor_swap": float(np.mean([r["p_donor_swap"] for r in rows])),
            "d_p_donor": float(np.mean(dd)),
            "p_donor_wilcoxon_greater": float(wilcoxon(dd, alternative="greater").pvalue)})
        out["rows"] = rows
    return out


def main():
    out = {"temp_stratified": {}, "uc_positions": {}}
    for model in ("ens2noiface", "seed43noiface", "noligand"):
        out["temp_stratified"][model] = temp_stratified(model)
        out["uc_positions"][model] = uc_positions(model)
    json.dump(out, open(os.path.join(SUR, "75_p01_custom.json"), "w"), indent=1)
    for model in out["temp_stratified"]:
        print(f"== {model} ==")
        for seg, v in out["temp_stratified"][model].items():
            print(f"  [{seg}] ΔP(native)={v['d_p_native_swap_minus_true']:+.4f} (p={v['p_native_wilcoxon_less']:.1e}) "
                  f"ΔP(donor)={v['d_p_donor_swap_minus_true']:+.4f} (p={v['p_donor_wilcoxon_greater']:.1e})")
        u = out["uc_positions"][model]
        if u.get("n_obs"):
            print(f"  [U↔C 位 n={u['n_obs']}] P(native) {u['p_native_true']:.3f}->{u['p_native_swap']:.3f} "
                  f"(Δ={u['d_p_native']:+.4f}, p={u['p_native_wilcoxon_less']:.1e}); "
                  f"P(donor) {u['p_donor_true']:.3f}->{u['p_donor_swap']:.3f} "
                  f"(Δ={u['d_p_donor']:+.4f}, p={u['p_donor_wilcoxon_greater']:.1e})")


if __name__ == "__main__":
    main()
