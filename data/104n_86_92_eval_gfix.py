#!/usr/bin/env python3
"""104n：gRNAdeCFIX 预测结构的 scTM（USalign）+ scMCC（barnaba）评估
       → survey/86_sctm_gfix.json、survey/92_scmcc_gfix.json

scTM：USalign native 在前，-outfmt 2 第 3 列（86 号 usalign_tm 同款），
      native = canonical_v2/{stem}.pdb。
scMCC：92_scmcc.py 同款——barnaba.annotate 非 XXX 配对（附 all-pairs），
      参考配对 = s1 缓存 ss_pairs（缺失则 RNAfold 现场算），论域 j-i>=4，
      PPV/Sens/MCC + 汇总 mean/median/boot_ci + zero_fill/all_pairs 变体。
输出 JSON = 原文件拷贝 + gRNAde 行/逐链替换为 cfix 值；原值存入
      summary["gRNAde_orig_bugdecode"]（86）/ summary["gRNAde_orig_bugdecode"]（92）对照。
环境：mpl372_repro（barnaba/torch）；PYTHONPATH=data/survey/pyshims
"""
import os, sys, json, subprocess, contextlib, io
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
SC_DIR = os.path.join(SUR, "86_sctm")
CAN_DIR = os.path.join(DATA_DIR, "canonical_v2")
USALIGN = os.path.join(ROOT, "tools", "USalign", "USalign")
S1_CACHE = os.path.join(DATA_DIR, "paco_s1_v2.pt")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
COND = "gRNAdeCFIX"
MIN_LOOP = 4


# ---------------------------------------------------------------- scTM
def usalign_tm(native_pdb, pred_pdb):
    r = subprocess.run([USALIGN, native_pdb, pred_pdb, "-outfmt", "2"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    for line in r.stdout.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            return float(parts[2])
    return None


def _tm_worker(args):
    stem, = args
    pp = os.path.join(SC_DIR, f"{COND}_{stem}.pdb")
    npdb = os.path.join(CAN_DIR, stem + ".pdb")
    if not os.path.exists(pp):
        return stem, None
    try:
        return stem, usalign_tm(npdb, pp)
    except Exception:
        return stem, None


# ---------------------------------------------------------------- scMCC
def db_to_pairs(db):
    stack, pairs = [], []
    for i, c in enumerate(db.strip()):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                pairs.append((stack.pop(), i))
    return set(pairs)


def rnafold_pairs(seq):
    r = subprocess.run([RNAFOLD, "--noPS"], input=seq + "\n",
                       capture_output=True, text=True, check=True)
    db = r.stdout.strip().splitlines()[1].split()[0]
    return db_to_pairs(db)


def barnaba_pairs(pdb_path):
    import barnaba as bb
    with contextlib.redirect_stderr(io.StringIO()):
        st, pr, reslist = bb.annotate(pdb_path)
    pairs_all = {(int(a), int(b)) for a, b in pr[0][0]}
    ann = pr[0][1]
    pairs_ann = {(int(a), int(b)) for (a, b), lab in zip(pr[0][0], ann)
                 if lab != "XXX"}
    seq = "".join(r.split("_")[0][-1] for r in reslist)
    return pairs_ann, pairs_all, len(reslist), seq


def metrics(pred, refset, L):
    uni = {(i, j) for i in range(L) for j in range(i + MIN_LOOP, L)}
    pred = {(i, j) for (i, j) in pred if (i, j) in uni}
    tp = len(pred & refset)
    fp = len(pred - refset)
    fn = len(refset - pred)
    tn = len(uni) - tp - fp - fn
    ppv = tp / (tp + fp) if (tp + fp) else None
    sen = tp / (tp + fn) if (tp + fn) else None
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(den) if den > 0 else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "PPV": ppv, "Sens": sen, "MCC": mcc,
            "n_pred": len(pred), "n_ref": len(refset)}


REF_PAIRS = None   # fork 前填充


def _mcc_worker(stem):
    fa = os.path.join(SC_DIR, f"{COND}_{stem}.fa")
    pdb = os.path.join(SC_DIR, f"{COND}_{stem}.pdb")
    if not os.path.exists(pdb):
        return stem, {"missing": True}
    cond_seq = [l.strip() for l in open(fa) if not l.startswith(">")][0]
    ref_pairs, L_native = REF_PAIRS[stem]
    try:
        pa, pall, n_res, pdb_seq = barnaba_pairs(pdb)
    except Exception as e:
        return stem, {"error": f"{type(e).__name__}: {e}"}
    rec = {"n_res_pdb": n_res, "L_native": L_native,
           "pdb_seq_matches_fa": pdb_seq == cond_seq}
    if n_res != L_native:
        rec["error"] = f"len_mismatch pdb={n_res} native={L_native}"
        return stem, rec
    rec["annotated"] = metrics(pa, ref_pairs, L_native)
    rec["all_pairs"] = metrics(pall, ref_pairs, L_native)
    rec["n_pairs_annotated_raw"] = len(pa)
    rec["n_pairs_all_raw"] = len(pall)
    return stem, rec


def boot_ci(x, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    global REF_PAIRS
    sc86 = json.load(open(os.path.join(SUR, "86_sctm.json")))
    stems = sorted(sc86["per_chain"])
    print(f"[104n] 链数 {len(stems)}", flush=True)

    # ---- scTM
    with ProcessPoolExecutor(max_workers=12) as ex:
        tm_res = dict(ex.map(_tm_worker, [(s,) for s in stems], chunksize=8))
    tm_vals = {s: v for s, v in tm_res.items() if v is not None}
    print(f"[104n] scTM 完成 {len(tm_vals)}/{len(stems)}", flush=True)

    # ---- 参考配对（92 同款：s1 缓存优先，缺失 RNAfold）
    import torch
    cache = torch.load(S1_CACHE, map_location="cpu", weights_only=False) \
        if os.path.exists(S1_CACHE) else {}
    REF_PAIRS = {}
    missing = []
    for stem in stems:
        if stem in cache and "ss_pairs" in cache[stem]:
            a = np.asarray(cache[stem]["ss_pairs"])
            L = a.shape[0]
            REF_PAIRS[stem] = ({(i, j) for i in range(L) for j in range(i + 1, L) if a[i, j]}, L)
        else:
            missing.append(stem)
    for stem in missing:
        fa = os.path.join(SC_DIR, f"native_{stem}.fa")
        seq = [l.strip() for l in open(fa) if not l.startswith(">")][0]
        REF_PAIRS[stem] = (rnafold_pairs(seq), len(seq))
    print(f"[104n] 参考配对：s1 命中 {len(stems)-len(missing)}，RNAfold 重算 {len(missing)}", flush=True)

    # ---- scMCC
    per = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (stem, rec) in enumerate(ex.map(_mcc_worker, stems, chunksize=8)):
            per[stem] = rec
            if (i + 1) % 50 == 0:
                print(f"[104n] scMCC {i+1}/{len(stems)}", flush=True)

    # ---- 汇总 scMCC（92 汇总同款）
    mccs, ppvs, sens, mccs_all, mccs_zf = [], [], [], [], []
    n_missing = n_err = n_lenmm = n_seqmm = 0
    missing_stems = []
    for stem in stems:
        r = per.get(stem, {})
        if r.get("missing"):
            n_missing += 1
            missing_stems.append(stem)
            continue
        if r.get("error"):
            n_err += 1
            if "len_mismatch" in str(r["error"]):
                n_lenmm += 1
            continue
        if not r.get("pdb_seq_matches_fa", True):
            n_seqmm += 1
        m = r["annotated"]
        if m["MCC"] is not None:
            mccs.append(m["MCC"])
        if m["PPV"] is not None:
            ppvs.append(m["PPV"])
        if m["Sens"] is not None:
            sens.append(m["Sens"])
        if r["all_pairs"]["MCC"] is not None:
            mccs_all.append(r["all_pairs"]["MCC"])
        mccs_zf.append(m["MCC"] if m["MCC"] is not None else 0.0)
    n_undef = len(mccs_zf) - len(mccs)
    sctm_v = [tm_vals[s] for s in stems if s in tm_vals]
    xs, ys = [], []
    for stem in stems:
        r = per.get(stem, {})
        v = tm_vals.get(stem)
        if r.get("annotated", {}).get("MCC") is not None and v is not None:
            xs.append(r["annotated"]["MCC"])
            ys.append(v)
    from scipy.stats import spearmanr
    rho = spearmanr(xs, ys) if len(xs) > 10 else None
    summ_cfix = {
        "n_chains": len(mccs),
        "n_mcc_undefined": n_undef,
        "n_missing_pdb": n_missing, "missing_stems": missing_stems,
        "n_error": n_err, "n_len_mismatch": n_lenmm,
        "n_pdb_seq_mismatch_fa": n_seqmm,
        "scMCC_mean": float(np.mean(mccs)) if mccs else None,
        "scMCC_median": float(np.median(mccs)) if mccs else None,
        "scMCC_ci95": boot_ci(mccs),
        "scMCC_mean_zero_fill": float(np.mean(mccs_zf)) if mccs_zf else None,
        "scMCC_mean_all_pairs": float(np.mean(mccs_all)) if mccs_all else None,
        "PPV_mean": float(np.mean(ppvs)) if ppvs else None,
        "Sens_mean": float(np.mean(sens)) if sens else None,
        "scTM_mean": float(np.mean(sctm_v)) if sctm_v else None,
        "spearman_scmcc_sctm": (float(rho.statistic), float(rho.pvalue)) if rho else None,
    }

    # ---- 86_sctm_gfix.json
    vals = list(tm_vals.values())
    sctm_row = {"n": len(vals), "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "pass_045": float(np.mean(np.array(vals) >= 0.45)),
                "pass_031": float(np.mean(np.array(vals) >= 0.31))} if vals else None
    out86 = {"n_stems": sc86["n_stems"],
             "per_chain": {s: dict(v) for s, v in sc86["per_chain"].items()},
             "summary": dict(sc86["summary"])}
    if "paco_minus_randmatch" in sc86:
        out86["paco_minus_randmatch"] = sc86["paco_minus_randmatch"]
    out86["summary"]["gRNAde_orig_bugdecode"] = sc86["summary"]["gRNAde"]
    out86["summary"]["gRNAde"] = sctm_row
    for stem, v in tm_vals.items():
        out86["per_chain"][stem]["sctm_gRNAde_orig_bugdecode"] = sc86["per_chain"][stem].get("sctm_gRNAde")
        out86["per_chain"][stem]["sctm_gRNAde"] = v
    out86["cfix_note"] = ("gRNAde 行 = 修正后序列（grnade_out_btest_v2_cfix 池内 best 设计）"
                          "RhoFold+ 重预测（gRNAdeCFIX_{stem}.pdb，GPU 2，2026-10-16）+ USalign "
                          "同款口径；其余行拷贝自 86_sctm.json。")
    tmp = os.path.join(SUR, f"86_sctm_gfix.json.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(out86, f, indent=1)
    os.replace(tmp, os.path.join(SUR, "86_sctm_gfix.json"))

    # ---- 92_scmcc_gfix.json
    sc92 = json.load(open(os.path.join(SUR, "92_scmcc.json")))
    out92 = {"definition": sc92["definition"],
             "summary": dict(sc92["summary"]),
             "comparisons": sc92["comparisons"],
             "per_chain": {s: dict(v) for s, v in sc92["per_chain"].items()}}
    out92["summary"]["gRNAde_orig_bugdecode"] = sc92["summary"]["gRNAde"]
    out92["summary"]["gRNAde"] = summ_cfix
    for stem, rec in per.items():
        out92["per_chain"].setdefault(stem, {})["gRNAde_orig_bugdecode"] = \
            sc92["per_chain"].get(stem, {}).get("gRNAde")
        out92["per_chain"][stem]["gRNAde"] = rec
    out92["cfix_note"] = ("gRNAde 行 = gRNAdeCFIX_{stem}.pdb（修正序列 RhoFold+ 重预测）"
                          "barnaba 同款口径重算（2026-10-16）；其余行拷贝自 92_scmcc.json。")
    tmp = os.path.join(SUR, f"92_scmcc_gfix.json.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(out92, f, indent=1)
    os.replace(tmp, os.path.join(SUR, "92_scmcc_gfix.json"))

    o86 = sc86["summary"]["gRNAde"]
    o92 = sc92["summary"]["gRNAde"]
    print(f"[104n] scTM gRNAde: {o86['mean']:.4f}->{sctm_row['mean']:.4f} "
          f"med {o86['median']:.4f}->{sctm_row['median']:.4f} "
          f">=0.45 {o86['pass_045']:.4f}->{sctm_row['pass_045']:.4f} (n={sctm_row['n']})")
    print(f"[104n] scMCC gRNAde: {o92['scMCC_mean']:.4f}->{summ_cfix['scMCC_mean']:.4f} "
          f"med {o92['scMCC_median']:.4f}->{summ_cfix['scMCC_median']:.4f} "
          f"PPV {o92['PPV_mean']:.4f}->{summ_cfix['PPV_mean']:.4f} "
          f"Sens {o92['Sens_mean']:.4f}->{summ_cfix['Sens_mean']:.4f} "
          f"(n={summ_cfix['n_chains']})")


if __name__ == "__main__":
    main()
