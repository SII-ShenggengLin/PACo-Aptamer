#!/usr/bin/env python3
"""104k：91 号实验 1（逐残基/Macro-F1）与实验 3（家族分层）的 gRNAde 行用 _cfix 池重算
       → survey/91_perbase_macrof1_gfix.json、survey/91_family_strata_gfix.json

口径与 data/91_perbase_macrof1.py 完全一致（CODE=A/U/G/C→0/1/2/3，native=canonical_v2，
'N' 位跳过，长度不符剔除；all=三温度 300 池，t01=0.1 文件）；
主表复现校验改对 63_main_table_gfix.json 的 gRNAde per_chain。
其余方法从原 JSON 原样拷贝。家族分层中 PACo_vs_RhoDesign 配对块不涉及 gRNAde，原样保留。
环境：rna_if（numpy/scipy）
"""
import os, json
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")

BASES = "AUGC"
B2I = {b: i for i, b in enumerate(BASES)}
CODE = {"A": 0, "U": 1, "G": 2, "C": 3}
TEMPS = ("0.1", "0.5", "1.0")
BOOT_N = 1000
SEED = 20261015


def atomic_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def enc_seq(s):
    return np.array([CODE.get(c, 255) for c in s], dtype=np.uint8)


def boot_ci(vals, stat=np.nanmean, n=BOOT_N, seed=SEED):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(vals), len(vals))
        stats.append(stat(vals[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def chain_pairs(native, samples):
    nat = enc_seq(native)
    L = len(nat)
    kept = [s for s in samples if len(s) == L]
    n_filt = len(samples) - len(kept)
    if not kept:
        return None, None, [], n_filt
    P = np.stack([enc_seq(s) for s in kept])
    nat_b = nat[None, :].astype(np.int16)
    valid = (nat_b != 255) & (P != 255)
    match = valid & (P == nat_b)
    vcnt = valid.sum(1)
    recs = [float(m / v) if v > 0 else None for m, v in zip(match.sum(1), vcnt)]
    recs = [r for r in recs if r is not None]
    t = np.broadcast_to(nat_b, P.shape)[valid].astype(np.int64)
    p = P[valid].astype(np.int64)
    return t, p, recs, n_filt


def confusion_from_pairs(t, p):
    return np.bincount(t * 4 + p, minlength=16).reshape(4, 4)


def f1_from_conf(conf):
    tp = np.diag(conf).astype(float)
    row = conf.sum(1).astype(float)
    col = conf.sum(0).astype(float)
    prec = np.where(col > 0, tp / np.maximum(col, 1), 0.0)
    rec = np.where(row > 0, tp / np.maximum(row, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0.0)
    return prec, rec, f1, float(f1.mean())


def load_lines(fp):
    return [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def main():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test = sorted(sp["test"])
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    mt_gfix = json.load(open(os.path.join(SUR, "63_main_table_gfix.json")))

    # ---------------- exp1：gRNAde 块（91 号同款，池换 _cfix）
    chains, conf_all_sum, conf_t01_sum = {}, np.zeros((4, 4), np.int64), np.zeros((4, 4), np.int64)
    v_best, v_t01 = {}, {}
    n_filt_all = n_filt_t01 = 0
    for stem in test:
        native = canon[stem]["seq"]
        pools = {t: load_lines(os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.{t}.seqs"))
                 for t in TEMPS}
        pl = {"t01": pools["0.1"], "all": [s for t in TEMPS for s in pools[t]]}
        t_a, p_a, recs_a, nf_a = chain_pairs(native, pl["all"])
        t_t, p_t, recs_t, nf_t = chain_pairs(native, pl["t01"])
        n_filt_all += nf_a
        n_filt_t01 += nf_t
        if not recs_t:
            continue
        ca = confusion_from_pairs(t_a, p_a) if t_a is not None else np.zeros((4, 4), np.int64)
        ct = confusion_from_pairs(t_t, p_t)
        conf_all_sum += ca
        conf_t01_sum += ct
        _, _, f1a, mf1a = f1_from_conf(ca)
        _, _, f1t, mf1t = f1_from_conf(ct)
        ra = np.diag(ca) / np.maximum(ca.sum(1), 1)
        rt = np.diag(ct) / np.maximum(ct.sum(1), 1)
        chains[stem] = {
            "len": len(native),
            "n_all": len(recs_a), "n_t01": len(recs_t),
            "n_filtered_all": nf_a, "n_filtered_t01": nf_t,
            "recovery_best": float(max(recs_a)) if recs_a else None,
            "recovery_t01_mean": float(np.mean(recs_t)),
            "per_base_all": {b: (float(ra[B2I[b]]) if ca.sum(1)[B2I[b]] > 0 else None) for b in BASES},
            "per_base_t01": {b: (float(rt[B2I[b]]) if ct.sum(1)[B2I[b]] > 0 else None) for b in BASES},
            "macroF1_all": mf1a, "macroF1_t01": mf1t,
            "support_t01": {b: int(ct.sum(1)[B2I[b]]) for b in BASES},
            "support_all": {b: int(ca.sum(1)[B2I[b]]) for b in BASES},
        }
        v_best[stem] = float(max(recs_a)) if recs_a else None
        v_t01[stem] = float(np.mean(recs_t))

    # 复现校验（对 63_gfix 主表 gRNAde 行）
    ref = mt_gfix["per_chain"]["gRNAde"]
    max_db = max(abs(v_best[s] - ref[s]["recovery_best"]) for s in test if s in v_best and s in ref)
    max_dt = max(abs(v_t01[s] - ref[s]["recovery_t01_mean"]) for s in test if s in v_t01 and s in ref)
    n_cmp = len([s for s in test if s in v_t01 and s in ref])
    print(f"[104k] verify vs 63_gfix: n={n_cmp}/{len(ref)} max|Δbest|={max_db:.3g} "
          f"max|Δt01|={max_dt:.3g} {'PASS' if max_db < 1e-9 and max_dt < 1e-9 else 'FAIL'}", flush=True)
    assert max_db < 1e-9 and max_dt < 1e-9 and n_cmp == len(ref)

    summ = {}
    for scope, conf_sum, k_rec, k_f1 in (("all", conf_all_sum, "per_base_all", "macroF1_all"),
                                         ("t01", conf_t01_sum, "per_base_t01", "macroF1_t01")):
        prec, rec, f1, mf1_micro = f1_from_conf(conf_sum)
        row, col = conf_sum.sum(1), conf_sum.sum(0)
        chain_mf1 = np.array([c[k_f1] for c in chains.values()])
        per_base = {}
        for b in BASES:
            arr = np.array([c[k_rec][b] if c[k_rec][b] is not None else np.nan
                            for c in chains.values()])
            per_base[b] = {
                "chain_macro": float(np.nanmean(arr)),
                "chain_macro_ci95": boot_ci(arr),
                "n_chains_with_base": int(np.sum(~np.isnan(arr))),
                "micro": float(np.diag(conf_sum)[B2I[b]] / row[B2I[b]]) if row[B2I[b]] > 0 else None,
                "support": int(row[B2I[b]]),
            }
        summ[scope] = {
            "per_base": per_base,
            "macroF1_chain_mean": float(chain_mf1.mean()),
            "macroF1_chain_mean_ci95": boot_ci(chain_mf1),
            "macroF1_micro": mf1_micro,
            "per_class_micro": {b: {"precision": float(prec[B2I[b]]), "recall": float(rec[B2I[b]]),
                                    "f1": float(f1[B2I[b]]), "support": int(row[B2I[b]]),
                                    "predicted_marginal": int(col[B2I[b]])} for b in BASES},
            "n_pairs": int(conf_sum.sum()),
        }
    gr_block = {"summary": summ, "per_chain": chains,
                "n_samples_filtered": {"all": n_filt_all, "t01": n_filt_t01}}

    orig1 = json.load(open(os.path.join(SUR, "91_perbase_macrof1.json")))
    out1 = {"meta": orig1["meta"],
            "main_table_verification": dict(orig1["main_table_verification"]),
            "methods": dict(orig1["methods"])}
    out1["main_table_verification"]["gRNAde"] = {
        "reference": "63_main_table_gfix.json per_chain.gRNAde（cfix 修正主表）",
        "n_compared": n_cmp, "n_ref": len(ref),
        "max_abs_diff_recovery_best": float(max_db),
        "max_abs_diff_recovery_t01": float(max_dt),
        "pass": True}
    out1["methods"]["gRNAde"] = gr_block
    out1["cfix_note"] = ("gRNAde 块 = grnade_out_btest_v2_cfix 池按 91 号 exp1 口径重算"
                         "（2026-10-16）；其余方法拷贝自 91_perbase_macrof1.json。")
    atomic_json(out1, os.path.join(SUR, "91_perbase_macrof1_gfix.json"))
    o = orig1["methods"]["gRNAde"]["summary"]
    print(f"[104k] macroF1 t01: chain {o['t01']['macroF1_chain_mean']:.4f}->"
          f"{summ['t01']['macroF1_chain_mean']:.4f} micro {o['t01']['macroF1_micro']:.4f}->"
          f"{summ['t01']['macroF1_micro']:.4f}")
    for b in BASES:
        print(f"       per-base t01 {b}: {o['t01']['per_base'][b]['chain_macro']:.4f}->"
              f"{summ['t01']['per_base'][b]['chain_macro']:.4f}")

    # ---------------- exp3：家族分层 gRNAde 行
    orig3 = json.load(open(os.path.join(SUR, "91_family_strata.json")))
    groups = {k: dict(v) for k, v in orig3["groups"].items()}
    for gname, g in groups.items():
        sub = [c for c in g["chains"] if c in chains]
        if not sub:
            g["methods"].pop("gRNAde", None)
            continue
        best = np.array([chains[c]["recovery_best"] for c in sub])
        t01 = np.array([chains[c]["recovery_t01_mean"] for c in sub])
        g["methods"] = dict(g["methods"])
        g["methods"]["gRNAde"] = {
            "n": len(sub),
            "recovery_best": {"mean": float(best.mean()), "ci95": boot_ci(best, stat=np.mean)},
            "recovery_t01_mean": {"mean": float(t01.mean()), "ci95": boot_ci(t01, stat=np.mean)},
            "per_chain": {c: {"recovery_best": float(chains[c]["recovery_best"]),
                              "recovery_t01_mean": float(chains[c]["recovery_t01_mean"])} for c in sub},
        }
        groups[gname] = g
    matrix_t01 = {k: dict(v) for k, v in orig3["matrix_family_x_method"]["recovery_t01_mean"].items()}
    matrix_best = {k: dict(v) for k, v in orig3["matrix_family_x_method"]["recovery_best"].items()}
    for gname, g in groups.items():
        if "gRNAde" in g["methods"]:
            matrix_t01.setdefault(gname, {})["gRNAde"] = g["methods"]["gRNAde"]["recovery_t01_mean"]["mean"]
            matrix_best.setdefault(gname, {})["gRNAde"] = g["methods"]["gRNAde"]["recovery_best"]["mean"]
        else:
            matrix_t01.get(gname, {}).pop("gRNAde", None)
            matrix_best.get(gname, {}).pop("gRNAde", None)
    out3 = {"meta": orig3["meta"], "groups": groups,
            "matrix_family_x_method": {"recovery_t01_mean": matrix_t01, "recovery_best": matrix_best},
            "n_family_chains": orig3["n_family_chains"], "n_other": orig3["n_other"],
            "cfix_note": ("gRNAde 行 = 104k 重算的 cfix 逐链恢复率；其余方法及 "
                          "PACo_vs_RhoDesign 配对块拷贝自 91_family_strata.json（2026-10-16）。")}
    atomic_json(out3, os.path.join(SUR, "91_family_strata_gfix.json"))
    print("[104k] 91_family_strata_gfix.json 写出；各族 gRNAde t01 old->new:")
    for gname in orig3["groups"]:
        o3 = orig3["groups"][gname]["methods"].get("gRNAde")
        n3 = groups[gname]["methods"].get("gRNAde")
        if o3 and n3:
            print(f"       {gname}: {o3['recovery_t01_mean']['mean']:.4f}->"
                  f"{n3['recovery_t01_mean']['mean']:.4f} (n={n3['n']})")


if __name__ == "__main__":
    main()
