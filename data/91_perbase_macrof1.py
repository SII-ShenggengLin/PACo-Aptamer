#!/usr/bin/env python3
"""第 91 步：论文补充分析（纯 CPU，无 CUDA）

实验 1：逐残基类型恢复率 + Macro-F1      → survey/91_perbase_macrof1.json
实验 2：跨方法置信度-恢复率耦合（PACo + AlignIF）→ survey/91_conf_cross.json
实验 3：配体家族分层恢复率                → survey/91_family_strata.json

口径（与 61/62/63 号主表脚本一致，已逐链复现校验）：
- native 一律 = survey/canonical_v2.json seq；native 'N' 位跳过
- 池序列长度 != 评估长度 → 剔除并计数；AlignIF 样本 '-' 位逐样本跳过
- PACo = ens2-noiface 主配置池（437 条 native 一致链 = 77_ens2_pool_btest）+
  paco_fix_pool_ens2noiface（40 条修复链）——经逐链校验，该组合 477/477 精确复现
  主表 btest_v2_main_table.json（paco_fix_pool/ens2/seed43 三个变体均只 3/40 匹配）；
  池顺序 t=0.1/0.5/1.0 各 100，前 100 = t01 子池
- NA-MPNN 42 条 gap 链（缺 C1' 等被官方 parse_PDB 丢弃残基）按 61 号"有效位口径"：
  pos_all = difflib(native, parse_PDB 序列) equal 块的 native 位，样本第 k 位 ↔ native[pos_all[k]]
  pos_all 缓存于 survey/91_nampnn_gap_pos.json；缺失时自动用 torch+prody 环境重建：
      PYTHONPATH=/tmp/prody_lib /root/miniconda3/envs/mpl372_repro/bin/python \
          data/91_perbase_macrof1.py --build-gap-cache
  （/tmp/prody_lib = pip install --target=/tmp/prody_lib --no-deps prody）

所有 JSON 输出先写临时文件再 os.replace，避免并行进程读到半截。
"""
import os, sys, json, difflib, subprocess
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
PY_TORCH = "/root/miniconda3/envs/mpl372_repro/bin/python"  # torch2.14+cpu
GAP_CACHE = os.path.join(SUR, "91_nampnn_gap_pos.json")

BASES = "AUGC"
B2I = {b: i for i, b in enumerate(BASES)}
CODE = {"A": 0, "U": 1, "G": 2, "C": 3}
TEMPS = ("0.1", "0.5", "1.0")
BOOT_N = 1000
SEED = 20261015

METHODS = ["PACoAptamer", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
MAIN_TABLE_TAG = {"PACoAptamer": "PACoAptamer", "RhoDesign": "RhoDesign",
                  "gRNAde": "gRNAde", "NA-MPNN": "NA-MPNN", "AlignIF": "AlignIF"}


# ---------------------------------------------------------------- 基础工具
def atomic_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def enc_seq(s):
    return np.array([CODE.get(c, 255) for c in s], dtype=np.uint8)


def boot_ci(vals, stat=np.nanmean, n=BOOT_N, seed=SEED):
    """链级重抽样 bootstrap 95% CI（percentile）"""
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(vals), len(vals))
        stats.append(stat(vals[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


# ---------------------------------------------------------------- gap 链缓存
def build_gap_cache():
    """用 NA-MPNN 官方 parse_PDB 重建 42 条 gap 链的有效位（需 torch+prody 环境）"""
    os.chdir(os.path.join(ROOT, "tools", "NA-MPNN", "inference"))
    sys.path.insert(0, os.getcwd())
    from data_utils import parse_PDB  # noqa
    restypes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                'UNK', 'DA', 'DC', 'DG', 'DT', 'DX', 'A', 'C', 'G', 'U', 'RX', 'MAS', 'PAD']
    r31 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
           'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
           'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
           'TYR': 'Y', 'VAL': 'V', 'UNK': 'X', 'DA': 'a', 'DC': 'c', 'DG': 'g',
           'DT': 't', 'DX': 'x', 'A': 'b', 'C': 'd', 'G': 'h', 'U': 'u', 'RX': 'y',
           'MAS': '-', 'PAD': '+'}
    alphabet = [r31[r] for r in restypes]
    RNA_MAP = {"b": "A", "d": "C", "h": "G", "u": "U", "y": "U"}
    canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
    met = json.load(open(os.path.join(SUR, "metrics_NAMPNN_btest_v2.json")))
    gaps = sorted(met["summary"]["gap_chains"].keys())
    out = {}
    for stem in gaps:
        d, _, _, _, _ = parse_PDB(os.path.join(DATA_DIR, "canonical_v2", f"{stem}.pdb"),
                                  device="cpu", model_type="na_mpnn", parse_na_only=0,
                                  na_shared_tokens=0, load_residues_with_missing_atoms=0)
        S = d["S"].cpu().numpy()
        parsed = "".join(RNA_MAP.get(alphabet[AA], "?") for AA in S)
        native = canon[stem]["seq"]
        sm = difflib.SequenceMatcher(a=native, b=parsed, autojunk=False)
        pos, ppos = [], []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                pos.extend(range(i1, i2))
                ppos.extend(range(j1, j2))
        assert "".join(native[i] for i in pos) == "".join(parsed[j] for j in ppos)
        # 61 号口径：样本长 = len(pos_all)
        seqs = [l.strip() for l in open(os.path.join(SUR, "nampnn_out_btest_v2", f"{stem}.seqs")) if l.strip()]
        assert all(len(s) == len(pos) for s in seqs), f"{stem} 样本长 {len(seqs[0])} != pos_all {len(pos)}"
        out[stem] = pos
        print(f"[gap] {stem}: native {len(native)} -> valid {len(pos)}", flush=True)
    atomic_json(out, GAP_CACHE)
    print(f"[gap] 写入 {GAP_CACHE}（{len(out)} 链）", flush=True)


def ensure_gap_cache():
    if os.path.exists(GAP_CACHE):
        return json.load(open(GAP_CACHE))
    env = dict(os.environ)
    env["PYTHONPATH"] = "/tmp/prody_lib" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    r = subprocess.run([PY_TORCH, os.path.abspath(__file__), "--build-gap-cache"], env=env)
    if r.returncode != 0:
        raise RuntimeError("gap 缓存构建失败；请手动运行："
                           f"PYTHONPATH=/tmp/prody_lib {PY_TORCH} data/91_perbase_macrof1.py --build-gap-cache")
    return json.load(open(GAP_CACHE))


# ---------------------------------------------------------------- 数据加载
def load_test():
    sp = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    return sorted(sp["test"])


def load_canon():
    return json.load(open(os.path.join(SUR, "canonical_v2.json")))


def load_lines(fp):
    return [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def load_method_pools(method, stems, canon):
    """返回 {stem: {'t01': [seq], 'all': [seq], 'extra': {...}}}
    PACo/AlignIF 额外带 per-sample logp（与 'all' 对齐）。"""
    out = {}
    if method == "PACoAptamer":
        # 主配置 ens2-noiface 修复池（经校验 40/40 复现主表）；缺目录时退回 paco_fix_pool
        fix_dir = os.path.join(SUR, "paco_fix_pool_ens2noiface")
        if not os.path.isdir(fix_dir):
            fix_dir = os.path.join(SUR, "paco_fix_pool")
        fix_stems = {f[:-5] for f in os.listdir(fix_dir)} if os.path.isdir(fix_dir) else set()
        for stem in stems:
            if stem in fix_stems:
                d = json.load(open(os.path.join(fix_dir, stem + ".json")))
                assert d["native"] == canon[stem]["seq"], f"{stem} fix pool native 不一致"
                pool, logp = d["pool"], d.get("logp")
                src = os.path.basename(fix_dir)
            else:
                d = json.load(open(os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")))
                assert d["native"] == canon[stem]["seq"], f"{stem} ens2 pool native 与 canonical 不一致"
                pool, logp = d["pool"], d.get("logp")
                src = "77_ens2_pool_btest"
            out[stem] = {"t01": pool[:100], "all": pool, "logp": logp, "src": src}
    elif method in ("RhoDesign", "gRNAde"):
        tag = "rhodesign_out_btest_v2" if method == "RhoDesign" else "grnade_out_btest_v2"
        for stem in stems:
            pools = {t: load_lines(os.path.join(SUR, tag, f"{stem}.{t}.seqs")) for t in TEMPS}
            out[stem] = {"t01": pools["0.1"],
                         "all": [s for t in TEMPS for s in pools[t]]}
    elif method == "NA-MPNN":
        for stem in stems:
            seqs = load_lines(os.path.join(SUR, "nampnn_out_btest_v2", f"{stem}.seqs"))
            out[stem] = {"t01": seqs, "all": seqs}
    elif method == "AlignIF":
        import re
        rx = re.compile(r"averaged log probability:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
        for stem in stems:
            fa = os.path.join(SUR, "alignif_out_btest_v2", stem, "out.fasta")
            lines = load_lines(fa)
            seqs, logp = [], []
            for j, l in enumerate(lines):
                if l.startswith(">id:") and j + 1 < len(lines):
                    seqs.append(lines[j + 1])
                    m = rx.search(l)
                    logp.append(float(m.group(1)) if m else None)
            out[stem] = {"t01": seqs, "all": seqs, "logp": logp}
    return out


# ---------------------------------------------------------------- 配对引擎
def chain_pairs(native, samples, gap_pos=None):
    """逐链比对 → (valid 掩码拼对的 true_codes, pred_codes, 每样本恢复率, n_filtered)

    gap_pos: NA-MPNN gap 链的 pos_all（native 位索引）；样本第 k 位 ↔ native[pos_all[k]]
    样本长度不符 → 剔除计数；native 'N' 与样本非 ACGU（'-'）位逐对跳过
    """
    nat = enc_seq(native) if gap_pos is None else enc_seq("".join(native[i] for i in gap_pos))
    L = len(nat)
    kept = [s for s in samples if len(s) == L]
    n_filt = len(samples) - len(kept)
    if not kept:
        return None, None, [], n_filt
    P = np.stack([enc_seq(s) for s in kept])          # n×L
    nat_b = nat[None, :].astype(np.int16)
    valid = (nat_b != 255) & (P != 255)               # native 已知 & 预测为 ACGU
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
    """4 类混淆矩阵 → per-class P/R/F1（zero_division=0）与 macro-F1"""
    tp = np.diag(conf).astype(float)
    row = conf.sum(1).astype(float)   # support（true）
    col = conf.sum(0).astype(float)   # 预测边际
    prec = np.where(col > 0, tp / np.maximum(col, 1), 0.0)
    rec = np.where(row > 0, tp / np.maximum(row, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0.0)
    return prec, rec, f1, float(f1.mean())


# ---------------------------------------------------------------- 实验 1
def exp1(test, canon, main_table):
    gap_pos = ensure_gap_cache()
    per_method, verify = {}, {}
    for method in METHODS:
        pools = load_method_pools(method, test, canon)
        chains, conf_all_sum, conf_t01_sum = {}, np.zeros((4, 4), np.int64), np.zeros((4, 4), np.int64)
        v_best, v_t01 = {}, {}
        n_filt_all = n_filt_t01 = 0
        for stem in test:
            native = canon[stem]["seq"]
            gp = gap_pos.get(stem) if method == "NA-MPNN" else None
            pl = pools[stem]
            t_a, p_a, recs_a, nf_a = chain_pairs(native, pl["all"], gp)
            t_t, p_t, recs_t, nf_t = chain_pairs(native, pl["t01"], gp)
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

        # ---- 主表复现校验
        ref = main_table["per_chain"][MAIN_TABLE_TAG[method]]
        diffs = []
        mism = []
        for stem in test:
            if stem not in v_t01 or stem not in ref:
                continue
            db = abs(v_best[stem] - ref[stem]["recovery_best"])
            dt = abs(v_t01[stem] - ref[stem]["recovery_t01_mean"])
            diffs.append((db, dt))
            if db > 1e-9 or dt > 1e-9:
                mism.append({"stem": stem, "abs_diff_best": float(db), "abs_diff_t01": float(dt)})
        max_db = max(d[0] for d in diffs)
        max_dt = max(d[1] for d in diffs)
        verify[method] = {"n_compared": len(diffs), "n_ref": len(ref),
                          "max_abs_diff_recovery_best": float(max_db),
                          "max_abs_diff_recovery_t01": float(max_dt),
                          "n_mismatch": len(mism), "mismatch_chains": mism,
                          "pass": bool(max_db < 1e-9 and max_dt < 1e-9 and len(diffs) == len(ref))}
        print(f"[verify] {method}: n={len(diffs)}/{len(ref)} "
              f"max|Δbest|={max_db:.3g} max|Δt01|={max_dt:.3g} "
              f"{'PASS' if verify[method]['pass'] else 'FAIL'}", flush=True)

        # ---- 汇总（链级 macro + 位点级 micro，两个池口径）
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
        per_method[method] = {"summary": summ, "per_chain": chains,
                              "n_samples_filtered": {"all": n_filt_all, "t01": n_filt_t01}}
        print(f"[exp1] {method}: macroF1(t01) chain={summ['t01']['macroF1_chain_mean']:.4f} "
              f"micro={summ['t01']['macroF1_micro']:.4f}", flush=True)

    out = {
        "meta": {
            "n_test_chains": len(test), "bases_order": list(BASES),
            "convention": "native=canonical_v2；'N' 位跳过；长度不符样本剔除；AlignIF '-' 逐样本跳过；"
                          "PACo=ens2池(437)+paco_fix_pool(40)；NA-MPNN gap 链按 61 号有效位口径",
            "pool_scope": {"all": "全池（PACo/RhoDesign/gRNAde 300=3温度×100；NA-MPNN 10；AlignIF 20）",
                           "t01": "t=0.1 子池（前 100；NA-MPNN/AlignIF 全部样本即 t=0.1）"},
            "bootstrap": {"n": BOOT_N, "seed": SEED, "level": 0.95, "resample": "chain"},
        },
        "main_table_verification": verify,
        "methods": per_method,
    }
    atomic_json(out, os.path.join(SUR, "91_perbase_macrof1.json"))
    print("[exp1] 写入 survey/91_perbase_macrof1.json", flush=True)
    return out


# ---------------------------------------------------------------- 实验 2
def _spearman_safe(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else None


def _boot_spearman(x, y, n=BOOT_N, seed=SEED):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r = _spearman_safe(x[idx], y[idx])
        if r is not None:
            bs.append(r)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def conf_block(per_chain, min_n):
    """通用耦合统计：链内 Spearman 分布 / 链间 Spearman / top-10% 富集"""
    stems = sorted(per_chain)
    in_rho = [per_chain[s]["within_rho"] for s in stems if per_chain[s]["within_rho"] is not None]
    x = [per_chain[s]["mean_logp"] for s in stems]
    yt = [per_chain[s]["recovery_t01_mean"] for s in stems]
    yb = [per_chain[s]["recovery_best"] for s in stems]
    enr = [per_chain[s]["top10pct_enrichment"] for s in stems]
    rho_t = _spearman_safe(x, yt)
    rho_b = _spearman_safe(x, yb)
    return {
        "n_chains": len(stems),
        "within_chain_spearman": {
            "min_samples_per_chain": min_n,
            "n_chains_valid": len(in_rho),
            "mean": float(np.mean(in_rho)), "median": float(np.median(in_rho)),
            "ci95": boot_ci(np.array(in_rho), stat=np.mean),
            "frac_positive": float(np.mean(np.array(in_rho) > 0))},
        "chain_level_spearman": {
            "mean_logp_vs_recovery_t01": {"rho": rho_t, "ci95": _boot_spearman(x, yt)},
            "mean_logp_vs_recovery_best": {"rho": rho_b, "ci95": _boot_spearman(x, yb)}},
        "top10pct_enrichment": {"mean": float(np.mean(enr)), "ci95": boot_ci(np.array(enr), stat=np.mean)},
    }


def exp2(test, canon):
    """PACo（ens2 池逐样本 logp）+ AlignIF（fasta 头 logP）；预留 methods 键"""
    methods_out = {}

    # ---------- PACo：legacy 复现（对照 77_p04）+ canonical 口径（437 一致链）
    legacy_pc, canon_pc = {}, {}
    for stem in test:
        p = os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        lp = d["logp"]
        # legacy：native 用池内存储值，逐样本 recovery 按 77 号原口径（zip 等长比对/len(native)）
        nat_old = d["native"]
        rc_old = [sum(1 for a, b in zip(s, nat_old) if a == b) / len(nat_old) for s in d["pool"]]
        lp_arr, rc_arr = np.array(lp), np.array(rc_old)
        order = np.argsort(-lp_arr)
        k = max(1, len(lp) // 10)
        legacy_pc[stem] = {
            "mean_logp": float(np.mean(lp)),
            "recovery_t01_mean": d["recovery_t01_mean"], "recovery_best": d["recovery_best"],
            "within_rho": _spearman_safe(lp, rc_old) if len(lp) >= 30 else None,
            "top10pct_enrichment": float(rc_arr[order[:k]].mean() - rc_arr.mean()),
        }
        # canonical：仅 native 与 canonical 一致的 437 链；逐样本 recovery 跳 N
        if nat_old == canon[stem]["seq"]:
            native = canon[stem]["seq"]
            npos = np.array([i for i, c in enumerate(native) if c != "N"])
            natv = enc_seq(native)
            rc_new = []
            for s in d["pool"]:
                if len(s) != len(native):
                    continue
                sv = enc_seq(s)[npos]
                rc_new.append(float(np.mean(sv == natv[npos])))
            if len(rc_new) == len(lp):
                rc_new = np.array(rc_new)
                order2 = np.argsort(-lp_arr)
                canon_pc[stem] = {
                    "mean_logp": float(np.mean(lp)),
                    "recovery_t01_mean": float(np.mean(rc_new[:100])),
                    "recovery_best": float(rc_new.max()),
                    "within_rho": _spearman_safe(lp, list(rc_new)) if len(lp) >= 30 else None,
                    "top10pct_enrichment": float(rc_new[order2[:k]].mean() - rc_new.mean()),
                }
    blk_legacy = conf_block(legacy_pc, 30)
    blk_canon = conf_block(canon_pc, 30)

    # 对照既有 77_p04
    p04 = json.load(open(os.path.join(SUR, "77_p04_confidence_recovery.json")))
    crosscheck = {
        "file": "survey/77_p04_confidence_recovery.json",
        "chain_level_rho_t01": {"recomputed": blk_legacy["chain_level_spearman"]["mean_logp_vs_recovery_t01"]["rho"],
                                "p04_file": p04["chain_level_spearman"]["mean_logp_vs_recovery_t01"]["rho"]},
        "chain_level_rho_best": {"recomputed": blk_legacy["chain_level_spearman"]["mean_logp_vs_recovery_best"]["rho"],
                                 "p04_file": p04["chain_level_spearman"]["mean_logp_vs_recovery_best"]["rho"]},
        "within_chain_mean": {"recomputed": blk_legacy["within_chain_spearman"]["mean"],
                              "p04_file": p04["within_chain_spearman"]["mean"]},
        "top10pct_enrichment_mean": {"recomputed": blk_legacy["top10pct_enrichment"]["mean"],
                                     "p04_file": p04["logp_rank_enrichment"]["top10pct_minus_pool_mean"]["mean"]},
    }
    print(f"[exp2] PACo legacy 复现: 链间 ρ(t01)={crosscheck['chain_level_rho_t01']['recomputed']:.4f} "
          f"(p04={crosscheck['chain_level_rho_t01']['p04_file']:.4f})", flush=True)

    methods_out["PACoAptamer"] = {
        "logp_source": "77_ens2_pool_btest pool JSON 逐样本 teacher-forced mean logP/nt",
        "legacy_replication_p04": blk_legacy,
        "canonical_437chains": blk_canon,
        "crosscheck_vs_p04": crosscheck,
        "per_chain_legacy": legacy_pc, "per_chain_canonical": canon_pc,
    }

    # ---------- AlignIF：fasta 头 averaged log probability（20 样本/链）
    ali_pools = load_method_pools("AlignIF", test, canon)
    ali_pc = {}
    for stem in test:
        pl = ali_pools[stem]
        native = canon[stem]["seq"]
        npos = np.array([i for i, c in enumerate(native) if c != "N"])
        natv = enc_seq(native)
        recs, lps = [], []
        for s, lp in zip(pl["all"], pl["logp"]):
            if lp is None or len(s) != len(native) or any(c not in "ACGU-" for c in s):
                continue
            sv = enc_seq(s)
            keep = npos[sv[npos] != 255]      # '-' 逐样本跳过
            if len(keep) == 0:
                continue
            recs.append(float(np.mean(sv[keep] == natv[keep])))
            lps.append(lp)
        if len(recs) < 10:
            continue
        lp_arr, rc_arr = np.array(lps), np.array(recs)
        order = np.argsort(-lp_arr)
        k = max(1, len(lps) // 10)
        ali_pc[stem] = {
            "n_samples": len(recs),
            "mean_logp": float(np.mean(lps)),
            "recovery_t01_mean": float(np.mean(recs)), "recovery_best": float(np.max(recs)),
            "within_rho": _spearman_safe(lps, recs),
            "top10pct_enrichment": float(rc_arr[order[:k]].mean() - rc_arr.mean()),
        }
    blk_ali = conf_block(ali_pc, 10)
    methods_out["AlignIF"] = {
        "logp_source": "out.fasta 头 'averaged log probability'（t=0.1 × ≤20 样本）",
        **blk_ali, "per_chain": ali_pc,
    }
    print(f"[exp2] AlignIF: n={blk_ali['n_chains']} 链间 ρ(t01)="
          f"{blk_ali['chain_level_spearman']['mean_logp_vs_recovery_t01']['rho']:.4f}", flush=True)

    out = {
        "meta": {
            "n_test_chains": len(test),
            "convention": "逐样本 recovery：native 'N' 位跳过、AlignIF '-' 逐样本跳过、长度不符剔除；"
                          "链间=mean(logp) vs 链级恢复率 Spearman；链内=逐样本 logp vs 逐样本恢复率 Spearman；"
                          "top-10% 富集=按 logp 排序前 10% 样本平均恢复率 − 池均值（链级 mean）；"
                          "PACo legacy 块严格复现 77 号 p04 口径（含 N 位不跳过、477 链）用于对照",
            "bootstrap": {"n": BOOT_N, "seed": SEED},
            "pending": "RhoDesign/gRNAde/NA-MPNN 逐样本 logP 由 GPU 实验补齐后并入 methods 键",
        },
        "methods": methods_out,
        "pending_methods": ["RhoDesign", "gRNAde", "NA-MPNN"],
    }
    atomic_json(out, os.path.join(SUR, "91_conf_cross.json"))
    print("[exp2] 写入 survey/91_conf_cross.json", flush=True)
    return out


# ---------------------------------------------------------------- 实验 3
def exp3(test, canon, main_table):
    from scipy.stats import wilcoxon
    fam = json.load(open(os.path.join(SUR, "75_families.json")))
    tset = set(test)
    chain2fam = {}
    for fk, fv in fam.items():
        for c in fv["chains"]:
            if c in tset:
                chain2fam.setdefault(c, fk)
    groups = {fk: sorted([c for c in fv["chains"] if c in tset]) for fk, fv in fam.items()}
    groups = {k: v for k, v in groups.items() if v}
    other = sorted(tset - set(chain2fam))
    groups["other"] = other

    pc_ref = main_table["per_chain"]
    fam_out, matrix_t01, matrix_best = {}, {}, {}
    for gname, chains in groups.items():
        g = {"n_chains": len(chains), "chains": chains, "methods": {}}
        for m in METHODS:
            ref = pc_ref[m]
            sub = [c for c in chains if c in ref]
            if not sub:
                continue
            best = np.array([ref[c]["recovery_best"] for c in sub])
            t01 = np.array([ref[c]["recovery_t01_mean"] for c in sub])
            g["methods"][m] = {
                "n": len(sub),
                "recovery_best": {"mean": float(best.mean()), "ci95": boot_ci(best, stat=np.mean)},
                "recovery_t01_mean": {"mean": float(t01.mean()), "ci95": boot_ci(t01, stat=np.mean)},
                "per_chain": {c: {"recovery_best": float(ref[c]["recovery_best"]),
                                  "recovery_t01_mean": float(ref[c]["recovery_t01_mean"])} for c in sub},
            }
        # PACo vs RhoDesign 逐链配对 Δ + Wilcoxon 单尾（H1: PACo > RhoDesign）
        ta, rh = pc_ref["PACoAptamer"], pc_ref["RhoDesign"]
        both = [c for c in chains if c in ta and c in rh]
        if both:
            d_t01 = np.array([ta[c]["recovery_t01_mean"] - rh[c]["recovery_t01_mean"] for c in both])
            d_best = np.array([ta[c]["recovery_best"] - rh[c]["recovery_best"] for c in both])
            def wilx(d):
                d = d[d != 0]
                if len(d) < 5:
                    return None
                return float(wilcoxon(d, alternative="greater").pvalue)
            g["PACo_vs_RhoDesign"] = {
                "n_paired": len(both),
                "delta_t01": {"mean": float(d_t01.mean()), "median": float(np.median(d_t01)),
                              "frac_positive": float(np.mean(d_t01 > 0)),
                              "wilcoxon_one_sided_p": wilx(d_t01)},
                "delta_best": {"mean": float(d_best.mean()), "median": float(np.median(d_best)),
                               "frac_positive": float(np.mean(d_best > 0)),
                               "wilcoxon_one_sided_p": wilx(d_best)},
                "per_chain_delta": {c: {"delta_t01": float(ta[c]["recovery_t01_mean"] - rh[c]["recovery_t01_mean"]),
                                        "delta_best": float(ta[c]["recovery_best"] - rh[c]["recovery_best"])}
                                    for c in both},
            }
        fam_out[gname] = g
        for m in METHODS:
            if m in g["methods"]:
                matrix_t01.setdefault(gname, {})[m] = g["methods"][m]["recovery_t01_mean"]["mean"]
                matrix_best.setdefault(gname, {})[m] = g["methods"][m]["recovery_best"]["mean"]

    out = {
        "meta": {
            "n_test_chains": len(test),
            "families_source": "survey/75_families.json（12 个多配体家族；与 test 477 取交集）",
            "recovery_source": "survey/btest_v2_main_table.json per_chain（主表权威口径）",
            "wilcoxon": "single-sided, H1: PACoAptamer > RhoDesign；零 Δ 对剔除（scipy 默认）；n<5 给 null",
            "bootstrap": {"n": BOOT_N, "seed": SEED, "resample": "chain"},
        },
        "groups": fam_out,
        "matrix_family_x_method": {"recovery_t01_mean": matrix_t01, "recovery_best": matrix_best},
        "n_family_chains": sum(len(v) for k, v in groups.items() if k != "other"),
        "n_other": len(other),
    }
    atomic_json(out, os.path.join(SUR, "91_family_strata.json"))
    print(f"[exp3] 写入 survey/91_family_strata.json（家族链 {out['n_family_chains']} + other {len(other)}）",
          flush=True)
    return out


# ---------------------------------------------------------------- main
def main():
    test = load_test()
    canon = load_canon()
    main_table = json.load(open(os.path.join(SUR, "btest_v2_main_table.json")))
    print(f"[91] test {len(test)} 链", flush=True)
    exp1(test, canon, main_table)
    exp2(test, canon)
    exp3(test, canon, main_table)
    print("[91] 全部完成", flush=True)


if __name__ == "__main__":
    if "--build-gap-cache" in sys.argv:
        build_gap_cache()
    else:
        main()
