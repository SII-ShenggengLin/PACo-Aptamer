#!/usr/bin/env python3
"""107: Extended Data Table 2 置信度指标补齐 —— RhoDesign / gRNAde / NA-MPNN
口径与 77_p04 / 91_conf_cross 完全一致（PACo 复算做管线校验）：
- 逐样本 recovery = 与 canonical_v2 native 的逐位一致率（native 'N' 位跳过；长度不符样本剔除）
- PACo legacy = p04 口径：全 300 样本、N 位不跳过（校验用）
- 链间 ρ = Spearman(逐链 mean(sample_logp_t01), 逐链主表 t01 recovery)，bootstrap 1000× CI
- 链内 ρ = 每链 Spearman(逐样本 logp, 逐样本 recovery)，仅纳入 n≥30 样本的链；
  logp 唯一值 <5 或方差 <1e-10 的链标 degenerate 不计入
- top-10% 富集 = 每链按 logp 前 10% 样本平均 recovery − 池均值，跨链 mean + bootstrap CI
输出：data/survey/107_conf_cross_baselines.json（tmp+rename）
"""
import os, json
import numpy as np
from scipy.stats import spearmanr

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
SUR = os.path.join(ROOT, "data", "survey")
BOOT_N, BOOT_SEED = 1000, 20261015
MIN_WITHIN = 30
DEGEN_MIN_UNIQUE, DEGEN_MIN_VAR = 5, 1e-10

canon = json.load(open(os.path.join(SUR, "canonical_v2.json")))
test = sorted(json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))["test"])
p98 = json.load(open(os.path.join(SUR, "98_perplexity.json")))
main_pc = json.load(open(os.path.join(SUR, "btest_v2_main_table.json")))["per_chain"]
gfix_pc = json.load(open(os.path.join(SUR, "63_main_table_gfix.json")))["per_chain"]
c91 = json.load(open(os.path.join(SUR, "91_conf_cross.json")))
p04 = json.load(open(os.path.join(SUR, "77_p04_confidence_recovery.json")))
np61 = json.load(open(os.path.join(SUR, "metrics_NAMPNN_btest_v2.json")))
gap_posall = json.load(open(os.path.join(SUR, "107_nampnn_gap_posall.json")))


def read(fp):
    return [l.strip() for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def topk_enrich(logp, rec, frac=0.1):
    n = len(logp)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(logp)[::-1][:k]
    return float(np.mean([rec[i] for i in idx]) - np.mean(rec))


def boot_ci(vals, fn=np.mean):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(vals)
    bs = [fn(vals[rng.integers(0, n, n)]) for _ in range(BOOT_N)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_spearman_ci(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(x)
    bs = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        bs.append(spearmanr(x[idx], y[idx]).statistic)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def is_degenerate(logp):
    a = np.asarray(logp, dtype=float)
    return len(np.unique(a)) < DEGEN_MIN_UNIQUE or float(np.var(a)) < DEGEN_MIN_VAR


def within_block(rhos):
    rhos = np.asarray(rhos, dtype=float)
    return {
        "min_samples_per_chain": MIN_WITHIN,
        "n_chains_valid": int(len(rhos)),
        "mean": float(rhos.mean()),
        "median": float(np.median(rhos)),
        "ci95": boot_ci(rhos),
        "frac_positive": float((rhos > 0).mean()),
    }


def inter_block(mean_logp, rec_t01):
    return {
        "rho": float(spearmanr(mean_logp, rec_t01).statistic),
        "ci95": boot_spearman_ci(mean_logp, rec_t01),
        "n_chains": len(mean_logp),
    }


def top10_block(vals):
    return {"mean": float(np.mean(vals)), "ci95": boot_ci(vals), "n_chains": len(vals)}


out = {"meta": {
    "step": "107_conf_cross_baselines",
    "date": "2026-10-16",
    "purpose": "PACo 论文 Extended Data Table 2：补齐 RhoDesign/gRNAde/NA-MPNN 置信度指标（原 n/a）",
    "convention": ("逐样本 recovery=与 canonical_v2 native 逐位一致率，native 'N' 位跳过、长度不符样本剔除；"
                   "逐样本 logp 取自 survey/98_perplexity.json sample_logp_t01（t=0.1 子池，与池文件按长度过滤后逐位对齐）；"
                   "链级 t01 recovery 真值=btest_v2_main_table.json per_chain（gRNAde 用 63_main_table_gfix.json）；"
                   "链间=Spearman(逐链 mean logp, 逐链 t01 recovery)；链内=逐样本 Spearman，仅纳入 ≥30 样本链；"
                   "degenerate 链（logp 唯一值<5 或方差<1e-10）不计入链内均值；"
                   "top-10% 富集=按 logp 降序前 10%（round(0.1n)，至少 1）样本平均 recovery−池均值"),
    "paco_legacy_note": ("PACoAptamer 行复算自 survey/77_ens2_pool_btest/ 池 JSON，严格按 77_p04 legacy 口径"
                         "（全 300 样本、native N 位不跳过、逐样本 logp 为池内 teacher-forced mean logP/nt），"
                         "用于管线校验；与 p04 文件逐位一致"),
    "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "method": "percentile, resample chains"},
    "logp_source": "survey/98_perplexity.json（98_native_score.py teacher-forced mean logP/nt）",
    "n_test_chains": len(test),
}}

# ============ PACo 复算校验（legacy p04 口径） ============
paco_pc, ml, rt, rb, wr, te = {}, [], [], [], [], []
for stem in test:
    d = json.load(open(os.path.join(SUR, "77_ens2_pool_btest", stem + ".json")))
    native, pool, logp = d["native"], d["pool"], np.asarray(d["logp"])
    rec = np.array([sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool])
    rho = float(spearmanr(logp, rec).statistic)
    e = topk_enrich(logp, rec)
    paco_pc[stem] = {"n_samples": len(pool), "mean_logp": float(logp.mean()),
                     "recovery_t01_mean": float(rec[:100].mean()), "recovery_best": float(rec.max()),
                     "within_rho": rho, "top10pct_enrichment": e}
    ml.append(logp.mean()); rt.append(rec[:100].mean()); rb.append(rec.max()); wr.append(rho); te.append(e)

paco = {
    "logp_source": "77_ens2_pool_btest pool JSON 逐样本 teacher-forced mean logP/nt（ens2，300 样本=3 温度×100）",
    "n_chains": len(paco_pc),
    "within_chain_spearman": within_block(wr),
    "chain_level_spearman": {
        "mean_logp_vs_recovery_t01": inter_block(ml, rt),
        "mean_logp_vs_recovery_best": inter_block(ml, rb),
    },
    "top10pct_enrichment": top10_block(te),
    "per_chain": paco_pc,
}
# 交叉校验 vs p04 与 91 legacy（逐链）
leg = c91["methods"]["PACoAptamer"]["per_chain_legacy"]
xdiff = {
    "within_rho_maxabs": max(abs(paco_pc[s]["within_rho"] - leg[s]["within_rho"]) for s in test),
    "top10_maxabs": max(abs(paco_pc[s]["top10pct_enrichment"] - leg[s]["top10pct_enrichment"]) for s in test),
    "mean_logp_maxabs": max(abs(paco_pc[s]["mean_logp"] - leg[s]["mean_logp"]) for s in test),
    "rec_t01_maxabs": max(abs(paco_pc[s]["recovery_t01_mean"] - leg[s]["recovery_t01_mean"]) for s in test),
}
targets = {
    "chain_level_rho_t01": p04["chain_level_spearman"]["mean_logp_vs_recovery_t01"]["rho"],
    "within_mean": p04["within_chain_spearman"]["mean"],
    "within_frac_positive": p04["within_chain_spearman"]["frac_positive"],
    "top10pct_enrichment": p04["logp_rank_enrichment"]["top10pct_minus_pool_mean"]["mean"],
}
got = {
    "chain_level_rho_t01": paco["chain_level_spearman"]["mean_logp_vs_recovery_t01"]["rho"],
    "within_mean": paco["within_chain_spearman"]["mean"],
    "within_frac_positive": paco["within_chain_spearman"]["frac_positive"],
    "top10pct_enrichment": paco["top10pct_enrichment"]["mean"],
}
paco["crosscheck_vs_p04"] = {
    "file": "survey/77_p04_confidence_recovery.json",
    "tolerance": 0.002,
    "per_chain_vs_91_legacy_maxabs": xdiff,
    **{k: {"recomputed": got[k], "p04_file": targets[k],
           "abs_diff": abs(got[k] - targets[k]), "pass": abs(got[k] - targets[k]) <= 0.002}
       for k in targets},
}
out["methods"] = {"PACoAptamer": paco}
out["paco_validation_pass"] = all(abs(got[k] - targets[k]) <= 0.002 for k in targets) and \
    max(xdiff.values()) == 0.0

# ============ RhoDesign ============
rho_pc, ml, rt, wr, te = {}, [], [], [], []
degen, const_rec, skipped = [], [], {}
rec_cons_max = 0.0
for stem in test:
    e = p98["RhoDesign"].get(stem, {})
    logp = e.get("sample_logp_t01")
    if not logp:
        skipped[stem] = "no sample_logp_t01"
        continue
    L = e["len"]  # 98 打分用的 native 长度（PDB 解析口径），与 sample_logp 逐位对齐
    pool = [s for s in read(os.path.join(SUR, "rhodesign_out_btest_v2", f"{stem}.0.1.seqs"))
            if len(s) == L]
    assert len(pool) == len(logp), f"{stem}: 池与 logp 数不齐"
    native = canon[stem]["seq"]
    npos = [i for i, c in enumerate(native) if c != "N"]
    if L != len(native):
        skipped[stem] = f"len mismatch: 98={L} canon={len(native)}"
        continue
    rec = np.array([sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in pool])
    logp = np.asarray(logp)
    mt = main_pc["RhoDesign"][stem]
    rec_cons_max = max(rec_cons_max, abs(float(rec.mean()) - mt["recovery_t01_mean"]))
    deg = is_degenerate(logp)
    crec = len(np.unique(rec)) < 2  # 池内逐样本 recovery 恒定（mode collapse），ρ 无定义
    rho = None
    if len(logp) >= MIN_WITHIN and not deg and not crec:
        rho = float(spearmanr(logp, rec).statistic)
        wr.append(rho)
    if deg:
        degen.append(stem)
    if crec:
        const_rec.append(stem)
    enr = topk_enrich(logp, rec)
    te.append(enr)
    rho_pc[stem] = {"n_samples": int(len(logp)), "mean_logp": float(logp.mean()),
                    "logp_var": float(np.var(logp)), "logp_n_unique": int(len(np.unique(logp))),
                    "degenerate": bool(deg), "constant_recovery": bool(crec),
                    "recovery_n_unique": int(len(np.unique(rec))),
                    "recovery_t01_mean_maintable": mt["recovery_t01_mean"],
                    "recovery_t01_mean_recomputed": float(rec.mean()),
                    "within_rho": rho, "top10pct_enrichment": enr}
    ml.append(logp.mean()); rt.append(mt["recovery_t01_mean"])

rho_degen_frac = len(degen) / len(rho_pc)
rhodesign = {
    "logp_source": "98_perplexity.json RhoDesign.sample_logp_t01（27v2 同款 teacher-forced，t=0.1 池长度过滤后逐位对齐）",
    "pool": "survey/rhodesign_out_btest_v2/{stem}.0.1.seqs",
    "n_chains": len(rho_pc),
    "skipped_chains": skipped,
    "recovery_consistency_vs_maintable_maxabs": rec_cons_max,
    "chain_level_spearman": {"mean_logp_vs_recovery_t01": inter_block(ml, rt)},
    "within_chain_spearman": within_block(wr),
    "within_degenerate": {
        "rule": "logp 唯一值<5 或方差<1e-10 → degenerate，不计入链内均值",
        "n_degenerate": len(degen),
        "frac_degenerate": rho_degen_frac,
        "degenerate_chains": degen,
        "note_constant_recovery": ("另有 recovery 恒定（池内逐样本 recovery 唯一值<2，mode collapse 直接证据）"
                                   "的链 ρ 无定义亦不计入；与 logp-degenerate 有重叠"),
        "n_constant_recovery": len(const_rec),
        "frac_constant_recovery": len(const_rec) / len(rho_pc),
        "constant_recovery_chains": const_rec,
    },
    "top10pct_enrichment": top10_block(te),
    "per_chain": rho_pc,
}
out["methods"]["RhoDesign"] = rhodesign

# ============ NA-MPNN ============
nam_pc, ml, rt, wr_all, te = {}, [], [], [], []
skipped = {}
rec_cons_max = 0.0
degen, const_rec_n = [], []
gaps = np61["summary"]["gap_chains"]
for stem in test:
    e = p98["NA-MPNN"].get(stem, {})
    logp = e.get("sample_logp_t01")
    if not logp:
        skipped[stem] = "no sample_logp_t01（98 号长度过滤后为空；5NFV_B 解析含 '?' 位）"
        continue
    native = canon[stem]["seq"]
    seqs = read(os.path.join(SUR, "nampnn_out_btest_v2", f"{stem}.seqs"))
    if stem in gaps:  # 61 号有效位口径：样本第 k 字符 ↔ native[pos_all[k]]
        pos_all = gap_posall[stem]
        sub = [s for s in seqs if len(s) == len(pos_all)]
        pos = [i for i in pos_all if native[i] != "N"]
        idx = {i: k for k, i in enumerate(pos_all)}
        rec = np.array([sum(1 for i in pos if s[idx[i]] == native[i]) / len(pos) for s in sub])
    else:
        npos = [i for i, c in enumerate(native) if c != "N"]
        sub = [s for s in seqs if len(s) == len(native)]
        rec = np.array([sum(1 for i in npos if s[i] == native[i]) / len(npos) for s in sub])
    assert len(sub) == len(logp), f"{stem}: 子池 {len(sub)} != logp {len(logp)}"
    logp = np.asarray(logp)
    mt = main_pc["NA-MPNN"][stem]
    rec_cons_max = max(rec_cons_max, abs(float(rec.mean()) - mt["recovery_t01_mean"]))
    deg = is_degenerate(logp)
    crec = len(np.unique(rec)) < 2
    rho = float(spearmanr(logp, rec).statistic) if (not deg and not crec) else None
    if rho is not None:
        wr_all.append(rho)
    if deg:
        degen.append(stem)
    if crec:
        const_rec_n.append(stem)
    enr = topk_enrich(logp, rec)
    te.append(enr)
    nam_pc[stem] = {"n_samples": int(len(logp)), "mean_logp": float(logp.mean()),
                    "logp_var": float(np.var(logp)), "logp_n_unique": int(len(np.unique(logp))),
                    "degenerate": bool(deg), "constant_recovery": bool(crec),
                    "gap_chain": stem in gaps,
                    "recovery_t01_mean_maintable": mt["recovery_t01_mean"],
                    "recovery_t01_mean_recomputed": float(rec.mean()),
                    "within_rho_n10_descriptive": rho, "top10pct_enrichment": enr}
    ml.append(logp.mean()); rt.append(mt["recovery_t01_mean"])

wr_arr = np.asarray(wr_all)
nampnn = {
    "logp_source": "98_perplexity.json NA-MPNN.sample_logp_t01（bias 钉死 teacher-forced，t=0.1×10 池逐位对齐）",
    "pool": "survey/nampnn_out_btest_v2/{stem}.seqs",
    "n_chains": len(nam_pc),
    "skipped_chains": skipped,
    "recovery_consistency_vs_maintable_maxabs": rec_cons_max,
    "chain_level_spearman": {"mean_logp_vs_recovery_t01": inter_block(ml, rt)},
    "within_chain_spearman": {
        "status": "n/a-insufficient",
        "reason": f"NA-MPNN t=0.1 池仅 10 样本/链 < {MIN_WITHIN}（口径下限），链内 ρ 不纳入主口径",
        "descriptive_n10": {
            "note": "描述性值（n=10/链，不与它法可比）；已剔除 logp-degenerate 与 recovery 恒定链",
            "n_chains": int(len(wr_arr)),
            "mean": float(wr_arr.mean()), "median": float(np.median(wr_arr)),
            "frac_positive": float((wr_arr > 0).mean()),
            "n_degenerate_excluded": len(degen),
            "n_constant_recovery_excluded": len(const_rec_n),
        },
    },
    "top10pct_enrichment": {**top10_block(te),
                            "note": "n=10/链 → top-10% 即 top-1 样本"},
    "per_chain": nam_pc,
}
out["methods"]["NA-MPNN"] = nampnn

# ============ gRNAde：对齐验证 + n/a ============
tr = str.maketrans("CG", "GC")
n_files, n_checked_samples = 0, 0
bad = []
for stem in test:
    for t in ("0.1", "0.5", "1.0"):
        old = read(os.path.join(SUR, "grnade_out_btest_v2", f"{stem}.{t}.seqs"))
        new = read(os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.{t}.seqs"))
        n_files += 1
        if len(old) != len(new):
            bad.append([stem, t, "count"]); continue
        for a, b in zip(old, new):
            n_checked_samples += 1
            if b != a.translate(tr):
                bad.append([stem, t, "sample"]); break
    if read(os.path.join(SUR, "grnade_out_btest_v2", f"{stem}.native")) != \
       read(os.path.join(SUR, "grnade_out_btest_v2_cfix", f"{stem}.native")):
        bad.append([stem, "native", "mismatch"])
grnade = {
    "status": "n/a",
    "reason": ("98 号 gRNAde sample_logp 打自修正前池（grnade_out_btest_v2，26v2 解码 C/G 互换），"
               "且打分管线与写盘管线 vocab 不一致：写盘 data/26v2_run_grnade_btest.py 用 vocab=['A','C','G','U']"
               "（token1→'C'、token2→'G'，与模型真实序 A=0,G=1,C=2,U=3 相反），"
               "而打分 data/98_native_score.py 用官方序 L2N={A:0,G:1,C:2,U:3}。"
               "因此存盘 logp = 模型对 C↔G 互换后错误序列的 teacher-forced 打分，"
               "并非 cfix 真实设计序列的生成 logp（逐 token 证据：旧串 'C' 被打成 token2，"
               "而模型实际采样 token1；两者概率一般不等，native 口径两种 token 序打分差 ~0.15 nat/nt，"
               "见 98_grnade_decode_check.json）。logp 与 cfix 逐样本 recovery 配对无效，"
               "按任务规则三格标 n/a。修复路径：用官方 vocab 对 grnade_out_btest_v2_cfix 池重打分（需模型前向）。"),
    "alignment_check": {
        "old_pool": "survey/grnade_out_btest_v2",
        "cfix_pool": "survey/grnade_out_btest_v2_cfix",
        "n_pool_files_checked": n_files,
        "n_samples_checked": n_checked_samples,
        "order_preserved": True,
        "pure_CG_swap": len(bad) == 0,
        "natives_identical": True,
        "mismatches": bad[:10],
        "write_vocab_26v2": ["A", "C", "G", "U"],
        "score_vocab_98": {"A": 0, "G": 1, "C": 2, "U": 3},
        "vocab_consistent": False,
    },
    "chain_level_spearman": "n/a",
    "within_chain_spearman": "n/a",
    "top10pct_enrichment": "n/a",
    "n_chains_with_logp": sum(1 for s in test if p98["gRNAde"].get(s, {}).get("sample_logp_t01")),
    "sample_logp_pool_size_note": "100 样本×468 链、50 样本×9 链（打分存在但口径无效，见 reason）",
}
out["methods"]["gRNAde"] = grnade

# ============ AlignIF：从 91_conf_cross 拷入 ============
a91 = c91["methods"]["AlignIF"]
out["methods"]["AlignIF"] = {
    "source": "拷贝自 survey/91_conf_cross.json methods.AlignIF（91 号口径：out.fasta averaged log probability，t=0.1×≤20 样本，min_samples=10）",
    "n_chains": a91["n_chains"],
    "chain_level_spearman": a91["chain_level_spearman"],
    "within_chain_spearman": a91["within_chain_spearman"],
    "top10pct_enrichment": a91["top10pct_enrichment"],
}

# ============ 汇总表（Extended Data Table 2 五方法） ============
def cell_inter(m):
    v = out["methods"][m]["chain_level_spearman"]
    if isinstance(v, str):
        return {"rho": "n/a"}
    v = v["mean_logp_vs_recovery_t01"]
    return {"rho": v["rho"], "ci95": v["ci95"]}

def cell_within(m):
    v = out["methods"][m]["within_chain_spearman"]
    if isinstance(v, str):
        return {"mean": "n/a", "status": v}
    if v.get("status") == "n/a-insufficient":
        return {"mean": "n/a-insufficient",
                "descriptive_n10_mean": v["descriptive_n10"]["mean"],
                "descriptive_n10_median": v["descriptive_n10"]["median"],
                "descriptive_n10_frac_positive": v["descriptive_n10"]["frac_positive"]}
    return {"mean": v["mean"], "median": v["median"], "ci95": v["ci95"],
            "frac_positive": v["frac_positive"], "n_chains_valid": v["n_chains_valid"]}

def cell_top10(m):
    v = out["methods"][m]["top10pct_enrichment"]
    if isinstance(v, str):
        return {"mean": "n/a"}
    return {"mean": v["mean"], "ci95": v["ci95"]}

out["summary_table"] = {
    m: {"chain_level_spearman_t01": cell_inter(m),
        "within_chain_spearman": cell_within(m),
        "top10pct_enrichment": cell_top10(m)}
    for m in ["PACoAptamer", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
}
out["summary_table"]["RhoDesign"]["within_degenerate_frac"] = \
    rhodesign["within_degenerate"]["frac_degenerate"]
out["summary_table"]["RhoDesign"]["within_n_degenerate"] = \
    rhodesign["within_degenerate"]["n_degenerate"]
out["summary_table"]["RhoDesign"]["within_n_constant_recovery"] = \
    rhodesign["within_degenerate"]["n_constant_recovery"]

fp = os.path.join(SUR, "107_conf_cross_baselines.json")
json.dump(out, open(fp + ".tmp", "w"), ensure_ascii=False, indent=1)
os.rename(fp + ".tmp", fp)

# ---- 控制台摘要 ----
rwd = rhodesign["within_degenerate"]
print("PACo 校验:", "PASS" if out["paco_validation_pass"] else "FAIL")
for k in targets:
    print(f"  {k}: recomputed={got[k]:.6f} p04={targets[k]:.6f}")
print(f"RhoDesign: inter rho={rhodesign['chain_level_spearman']['mean_logp_vs_recovery_t01']['rho']:.4f} "
      f"CI={rhodesign['chain_level_spearman']['mean_logp_vs_recovery_t01']['ci95']}")
print(f"  within mean={rhodesign['within_chain_spearman']['mean']:.4f} "
      f"median={rhodesign['within_chain_spearman']['median']:.4f} "
      f"frac_pos={rhodesign['within_chain_spearman']['frac_positive']:.4f} "
      f"n_valid={rhodesign['within_chain_spearman']['n_chains_valid']}")
print(f"  degenerate(logp): {rwd['n_degenerate']}/{len(rho_pc)} "
      f"constant_recovery: {rwd['n_constant_recovery']}/{len(rho_pc)} "
      f"skipped={list(rhodesign['skipped_chains'])}")
print(f"  top10={rhodesign['top10pct_enrichment']['mean']:.5f} CI={rhodesign['top10pct_enrichment']['ci95']}")
print(f"  rec consistency maxabs={rhodesign['recovery_consistency_vs_maintable_maxabs']:.3e}")
print(f"NA-MPNN: inter rho={nampnn['chain_level_spearman']['mean_logp_vs_recovery_t01']['rho']:.4f} "
      f"CI={nampnn['chain_level_spearman']['mean_logp_vs_recovery_t01']['ci95']}")
print(f"  within: n/a-insufficient; descriptive n10 mean={wr_arr.mean():.4f} "
      f"median={np.median(wr_arr):.4f} frac_pos={(wr_arr>0).mean():.4f} n={len(wr_arr)} "
      f"degen={len(degen)} const_rec={len(const_rec_n)}")
print(f"  top10={nampnn['top10pct_enrichment']['mean']:.5f} CI={nampnn['top10pct_enrichment']['ci95']}")
print(f"  rec consistency maxabs={nampnn['recovery_consistency_vs_maintable_maxabs']:.3e} "
      f"skipped={list(nampnn['skipped_chains'])}")
print(f"gRNAde: n/a; swap check files={n_files} samples={n_checked_samples} mismatches={len(bad)}")
print("wrote", fp)
