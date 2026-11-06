#!/usr/bin/env python3
"""第 92-b 步：scMCC（实验 5，纯 CPU）

从 survey/86_sctm/{cond}_{stem}.pdb（RhoFold+ 预测 3D，193 链 × 6 条件：
native / PACo / RhoDesign / gRNAde / NA-MPNN / randmatch）提取碱基配对，与参考
配对集比较，报 PPV / Sensitivity / MCC（scMCC，AlignIF 论文口径）。

- 3D→配对：barnaba.annotate（几何接触 + 平面角 + H-bond donor/acceptor 判定）。
  主口径 = 带有效边标注（非 "XXX"，即通过角度+H键检验）的配对；
  附带记录 all-pairs 变体（所有近接触配对，含 XXX）作敏感性。
- 参考配对集 = native 序列 RNAfold MFE 配对：优先 data/paco_s1_v2.pt[stem]
  ['ss_pairs']（(L,L) bool，45 号缓存同款口径）；缓存缺失则现场用 RNAfold
  --noPS 对 native 序列（86_sctm/native_{stem}.fa）重算 dot-bracket → 配对集。
- MCC 论域：0<=i<j<L 且 j-i>=4（RNAfold 最小发夹环 3，更短配对物理上不可能）。
  TP/FP/FN/TN → MCC；分母为 0 记 None（逐链保留原始计数）。
- 统计：每 cond mean/median/bootstrap CI95；PACo vs randmatch 配对 Wilcoxon
  （单侧 greater + 双侧）；与 86_sctm.json 的 scTM 并排（逐链 + 汇总 +
  每 cond Spearman(scMCC, scTM)）。

输出：survey/92_scmcc.json（临时文件 + rename）
环境：/root/miniconda3/envs/mpl372_repro/bin/python（torch+barnaba+numpy+scipy）
"""
import os, sys, json, subprocess, contextlib, io
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
SC_DIR = os.path.join(SUR, "86_sctm")
S1_CACHE = os.path.join(DATA_DIR, "paco_s1_v2.pt")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
OUT_TMP = os.path.join(SUR, "92_scmcc.json.tmp")
OUT = os.path.join(SUR, "92_scmcc.json")
CONDS = ["native", "PACo", "RhoDesign", "gRNAde", "NA-MPNN", "randmatch"]
MIN_LOOP = 4   # j-i >= 4 才计入论域


# ---------------------------------------------------------------- 参考配对
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


def load_ref_pairs(stems):
    import torch
    ref, missing = {}, []
    cache = torch.load(S1_CACHE, map_location="cpu", weights_only=False) \
        if os.path.exists(S1_CACHE) else {}
    for stem in stems:
        if stem in cache and "ss_pairs" in cache[stem]:
            a = np.asarray(cache[stem]["ss_pairs"])
            L = a.shape[0]
            ref[stem] = {(i, j) for i in range(L) for j in range(i + 1, L)
                         if a[i, j]}
        else:
            missing.append(stem)
    for stem in missing:   # 缓存缺失 → RNAfold 现场重算
        fa = os.path.join(SC_DIR, f"native_{stem}.fa")
        seq = [l.strip() for l in open(fa) if not l.startswith(">")][0]
        ref[stem] = rnafold_pairs(seq)
    return ref, missing


# ---------------------------------------------------------------- barnaba 提取
def barnaba_pairs(pdb_path):
    """返回 (pairs_annotated, pairs_all, n_res, seq_from_pdb)"""
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


def work_one(args):
    stem, cond, ref_pairs, L_native, cond_seq = args
    pdb = os.path.join(SC_DIR, f"{cond}_{stem}.pdb")
    if not os.path.exists(pdb):
        return stem, cond, {"missing": True}
    try:
        pa, pall, n_res, pdb_seq = barnaba_pairs(pdb)
    except Exception as e:
        return stem, cond, {"error": f"{type(e).__name__}: {e}"}
    rec = {"n_res_pdb": n_res, "L_native": L_native,
           "pdb_seq_matches_fa": pdb_seq == cond_seq}
    if n_res != L_native:
        rec["error"] = f"len_mismatch pdb={n_res} native={L_native}"
        return stem, cond, rec
    rec["annotated"] = metrics(pa, ref_pairs, L_native)
    rec["all_pairs"] = metrics(pall, ref_pairs, L_native)
    rec["n_pairs_annotated_raw"] = len(pa)
    rec["n_pairs_all_raw"] = len(pall)
    return stem, cond, rec


# ---------------------------------------------------------------- 统计
def boot_ci(x, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    sc86 = json.load(open(os.path.join(SUR, "86_sctm.json")))
    stems = sorted(sc86["per_chain"])
    print(f"链数 {len(stems)}")

    # native 序列（86 .fa）+ 参考配对
    native_seq = {}
    for stem in stems:
        fa = os.path.join(SC_DIR, f"native_{stem}.fa")
        native_seq[stem] = [l.strip() for l in open(fa)
                            if not l.startswith(">")][0]
    # 各 cond 被折叠序列（86 .fa；design 与 native 不同属正常）
    cond_seq = {}
    for stem in stems:
        for cond in CONDS:
            fa = os.path.join(SC_DIR, f"{cond}_{stem}.fa")
            if os.path.exists(fa):
                cond_seq[(stem, cond)] = [l.strip() for l in open(fa)
                                          if not l.startswith(">")][0]
    ref, missing = load_ref_pairs(stems)
    print(f"参考配对：s1 缓存命中 {len(stems)-len(missing)}，RNAfold 重算 {len(missing)}"
          f" {missing[:5]}")

    jobs = [(stem, cond, ref[stem], len(native_seq[stem]),
             cond_seq.get((stem, cond), ""))
            for stem in stems for cond in CONDS]
    per = {}   # stem -> cond -> rec
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (stem, cond, rec) in enumerate(ex.map(work_one, jobs, chunksize=8)):
            per.setdefault(stem, {})[cond] = rec
            if (i + 1) % 200 == 0:
                print(f"  进度 {i+1}/{len(jobs)}", flush=True)

    # ---- 汇总
    from scipy.stats import wilcoxon, spearmanr
    summary = {}
    for cond in CONDS:
        mccs, ppvs, sens = [], [], []
        mccs_all, mccs_zf = [], []
        n_missing = n_err = n_lenmm = n_seqmm = 0
        missing_stems = []
        for stem in stems:
            r = per[stem].get(cond, {})
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
        sctm = [sc86["per_chain"][s].get(f"sctm_{cond}") for s in stems]
        sctm_v = [x for x in sctm if x is not None]
        # Spearman(scMCC, scTM)（两者都有的链）
        xs, ys = [], []
        for stem in stems:
            r = per[stem].get(cond, {})
            v = sc86["per_chain"][stem].get(f"sctm_{cond}")
            if r.get("annotated", {}).get("MCC") is not None and v is not None:
                xs.append(r["annotated"]["MCC"])
                ys.append(v)
        rho = spearmanr(xs, ys) if len(xs) > 10 else None
        summary[cond] = {
            "n_chains": len(mccs),
            "n_mcc_undefined": n_undef,
            "n_missing_pdb": n_missing, "missing_stems": missing_stems,
            "n_error": n_err,
            "n_len_mismatch": n_lenmm,
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
        print(f"  {cond:10s} scMCC {summary[cond]['scMCC_mean']:.4f} "
              f"(med {summary[cond]['scMCC_median']:.4f}, zf "
              f"{summary[cond]['scMCC_mean_zero_fill']:.4f}, allpairs "
              f"{summary[cond]['scMCC_mean_all_pairs']:.4f}) | "
              f"PPV {summary[cond]['PPV_mean']:.4f} Sens {summary[cond]['Sens_mean']:.4f} | "
              f"scTM {summary[cond]['scTM_mean']:.4f} | n={len(mccs)}")

    # PACo vs randmatch（配对 Wilcoxon，主口径 annotated MCC）
    def paired(a_cond, b_cond, key="MCC"):
        a, b = [], []
        for stem in stems:
            ra = per[stem].get(a_cond, {}).get("annotated", {})
            rb = per[stem].get(b_cond, {}).get("annotated", {})
            if ra.get(key) is not None and rb.get(key) is not None:
                a.append(ra[key])
                b.append(rb[key])
        return np.array(a), np.array(b)
    cmp_out = {}
    for pair in [("PACo", "randmatch"), ("PACo", "native"), ("PACo", "RhoDesign")]:
        a, b = paired(*pair)
        d = a - b
        try:
            p1 = float(wilcoxon(a, b, alternative="greater").pvalue)
            p2 = float(wilcoxon(a, b).pvalue)
        except Exception as e:
            p1 = p2 = None
        cmp_out[f"{pair[0]}_minus_{pair[1]}"] = {
            "n": len(d), "delta_mean": float(d.mean()),
            "delta_ci95": boot_ci(d),
            "wilcoxon_p_one_sided_greater": p1, "wilcoxon_p_two_sided": p2}
        print(f"  {pair[0]}−{pair[1]}: Δ={d.mean():.4f} p1={p1:.3g} p2={p2:.3g}")

    out = {
        "definition": {
            "extractor": "barnaba.annotate（pip 0.1.9）；主口径=非 XXX 标注配对"
                         "（角度+H-bond 通过），all_pairs=全部近接触配对",
            "reference": "native 序列 RNAfold MFE 配对（paco_s1_v2.pt ss_pairs；"
                         "缺失现场 RNAfold --noPS 重算）",
            "universe": "0<=i<j<L 且 j-i>=4",
            "ref_from_rnafold": missing,
            "n_chains": len(stems),
        },
        "summary": summary,
        "comparisons": cmp_out,
        "per_chain": per,
    }
    json.dump(out, open(OUT_TMP, "w"), indent=1)
    os.replace(OUT_TMP, OUT)
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
