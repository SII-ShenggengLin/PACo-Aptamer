#!/usr/bin/env python3
"""第 94 步：PACo(ens2) 在 RhoFold+ 预测骨架上的设计（86_sctm 193 链）

干净对照设计：坐标来自预测结构（86_sctm/native_{stem}.pdb），native 序列与
RNAfold(native) SS 保持不变（与晶体骨架口径唯一差异 = 9 原子坐标来源）。

相位：
  --phase crystal  复现校验：同协议 ens2×300 采样，但特征走 load_s1 缓存（晶体
                   骨架口径，与 77 号池/主表同路径）→ 94_predbb_pool_crystal/{stem}.json
  --phase predbb   主实验：变体 build_feature 从预测 PDB 建 9 原子坐标
                   → 94_predbb_pool/{stem}.json（77 号池同格式：native/pool/logp/
                   logp_native/recovery_best/recovery_t01_mean）
  --phase summary  汇总 → 94_predbb_paco.json：
                   per_chain best/t01；与主表（btest_v2_main_table PACoAptamer）
                   逐链 Δ 分布（mean/median/bootstrap CI、配对 Wilcoxon）；
                   Spearman(sctm_native, predbb 恢复率) 与 Spearman(sctm_native, Δ)；
                   validation 段：crystal 重跑 vs 主表同链 的一致性。

采样协议（ens2 主配置）：torch.manual_seed(42) 全局一次；两 ckpt
s2_v2_seed43_noiface + s2_v2_ssdo15_noiface 逐步 softmax 概率平均；
3 温度 (0.1/0.5/1.0) × 100 = 300 样本/链（39/77 号同函数同口径）。

环境（2026-10-16 起）：aptamer_eval 丢失 → odesign_service_env
（torch 2.3.1+cu121 + 真实 torch_scatter/cluster）。
用法：CUDA_VISIBLE_DEVICES=0 python data/94_predbb_paco.py --phase predbb
"""
import os, sys, json, time, argparse, subprocess, re
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

TEMPS = [0.1, 0.5, 1.0]
N_PER_TEMP = 100
POOL_DIR = os.path.join(SUR, "94_predbb_pool")
POOL_DIR_CRY = os.path.join(SUR, "94_predbb_pool_crystal")
RNAFOLD = "/root/miniconda3/envs/viennarna_fold_env/bin/RNAfold"
ENS_CKPTS = ["s2_v2_seed43_noiface", "s2_v2_ssdo15_noiface"]

import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m74 = _imp("m74", os.path.join(DATA_DIR, "74_p00_iface_ablation.py"))
m39 = _imp("m39", os.path.join(DATA_DIR, "39_eval_paco_ensemble.py"))
m32 = _imp("m32", os.path.join(DATA_DIR, "32v2_compute_metrics_nampnn.py"))
MOD_MAP = m32.MOD_MAP
STD = {"A", "C", "G", "U"}

from PACoAptamer.data_utils import PACO9, ss_to_pairs


def stems_193():
    d = json.load(open(os.path.join(SUR, "86_sctm.json")))
    return sorted(d["per_chain"].keys())


def build_feature_predbb(stem, native):
    """68 号 build_feature 变体：PDB 换为 RhoFold+ 预测结构；
    native 序列 = load_s1 缓存 seq（== canonical_v2 == 86_sctm native fa，已抽查 30/30 全等），
    SS = RNAfold(native) MFE（已抽查与 s1 缓存 ss_pairs 30/30 全等 → 与主表 SS 条件相同）。"""
    pdb_p = os.path.join(SUR, "86_sctm", f"native_{stem}.pdb")
    if not os.path.exists(pdb_p):
        return None, "no_pred_pdb"
    residues = {}
    order = []
    with open(pdb_p) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            rn = line[17:20].strip()
            if rn not in STD and rn not in MOD_MAP:
                continue
            key = (line[22:26].strip(), line[26].strip())
            atom = line[12:16].strip().replace("*", "'")
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            if key not in residues:
                residues[key] = {}
                order.append(key)
            residues[key][atom] = xyz
    L = len(order)
    if L != len(native):
        return None, f"len_mismatch pdb={L} native={len(native)}"
    coords9 = np.full((L, 9, 3), np.nan, dtype=np.float32)
    for i, key in enumerate(order):
        for j, atom in enumerate(PACO9):
            if atom in residues[key]:
                coords9[i, j] = residues[key][atom]
    r = subprocess.run([RNAFOLD, "--noPS"], input=native + "\n",
                       capture_output=True, text=True, timeout=60)
    m = re.findall(r"[\(\)\.\[\]\{\}\<\>]+", r.stdout)
    ss = m[0] if m else "." * len(native)
    return {"coords9": coords9, "ss_pairs": ss_to_pairs(ss), "seq": native}, "ok"


def load_ens2(device):
    import torch
    models = []
    alphabet = None
    for n in ENS_CKPTS:
        p = os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
        m, alphabet = m74.load_model(p, device)
        models.append(m)
    return models, alphabet


def make_score_seqs(models, alphabet, device):
    """77 号 score_seqs 同款：teacher-forced mean logP，集成 = softmax 概率平均"""
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def score_seqs(seqs, d, chunk=100):
        L = len(d['seq'])
        coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        bos = alphabet.get_idx("<cath>")
        out = []
        for s0 in range(0, len(seqs), chunk):
            batch = seqs[s0:s0 + chunk]
            B = len(batch)
            prev = torch.zeros(B, L, dtype=torch.long, device=device)
            tgt = torch.zeros(B, L, dtype=torch.long, device=device)
            for bi, sq in enumerate(batch):
                toks = [alphabet.get_idx(c) for c in sq]
                prev[bi, 0] = bos
                prev[bi, 1:] = torch.tensor(toks[:-1], device=device)
                tgt[bi] = torch.tensor(toks, device=device)
            probs = None
            for m in models:
                eo = m.forward_embedding(coords, ss, lig, ligm, pmask, conf)
                eoB = {"encoder_out": [eo["encoder_out"][0].repeat(1, B, 1)],
                       "encoder_padding_mask": [eo["encoder_padding_mask"][0].repeat(B, 1)],
                       "encoder_embedding": [], "encoder_states": []}
                logits, _ = m.decoder(prev, encoder_out=eoB)
                p = F.softmax(logits, dim=1)
                probs = p if probs is None else probs + p
            probs = probs / len(models)
            lp = torch.log(probs + 1e-12).gather(1, tgt.unsqueeze(1)).squeeze(1)
            out.extend(lp.mean(dim=1).float().cpu().numpy().tolist())
        return out
    return score_seqs


def run_pool(mode):
    """mode: 'predbb'（预测 PDB 特征）或 'crystal'（load_s1 缓存特征）"""
    import torch
    from PACoAptamer.data_utils import load_s1
    device = 0
    torch.manual_seed(42)
    models, alphabet = load_ens2(device)
    score_seqs = make_score_seqs(models, alphabet, device)
    out_dir = POOL_DIR if mode == "predbb" else POOL_DIR_CRY
    os.makedirs(out_dir, exist_ok=True)

    stems = stems_193()
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    data = load_s1(set(sp2["test"]), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))

    t0 = time.time()
    n_done, skipped = 0, {}
    for stem in stems:
        out_p = os.path.join(out_dir, stem + ".json")
        if os.path.exists(out_p):
            continue
        if stem not in data:
            skipped[stem] = "not_in_s1"
            continue
        if mode == "crystal":
            d = data[stem]
            if len(d['seq']) > 512:
                skipped[stem] = "too_long"
                continue
        else:
            d, status = build_feature_predbb(stem, data[stem]['seq'])
            if d is None:
                skipped[stem] = status
                print(f"  [SKIP] {stem}: {status}", flush=True)
                continue
        native = d['seq']
        try:
            pool = []
            for t in TEMPS:
                pool.extend(m39.sample_chain_ens(models, alphabet, d, device, t, N_PER_TEMP))
            logp = score_seqs(pool, d)
            logp_native = score_seqs([native], d)[0]
        except RuntimeError as e:
            skipped[stem] = f"runtime: {e}"
            print(f"  [pool 跳过 {stem}: {e}]", flush=True)
            continue
        recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
        tmp = out_p + ".tmp"
        json.dump({"len": len(native), "native": native, "pool": pool, "logp": logp,
                   "logp_native": logp_native, "n_per_temp": N_PER_TEMP,
                   "recovery_best": max(recs),
                   "recovery_t01_mean": float(np.mean(recs[:N_PER_TEMP]))},
                  open(tmp, "w"))
        os.replace(tmp, out_p)
        n_done += 1
        if n_done % 25 == 0:
            print(f"[{mode}] {n_done} 链完成 | {time.time()-t0:.0f}s", flush=True)
    print(f"[{mode}] 完成 {n_done}，跳过 {len(skipped)}: {skipped}", flush=True)


def boot_ci(x, n=2000, seed=0):
    if len(x) < 3:
        return None
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def phase_summary():
    from scipy.stats import wilcoxon, spearmanr
    sctm = json.load(open(os.path.join(SUR, "86_sctm.json")))["per_chain"]
    main_tab = json.load(open(os.path.join(SUR, "btest_v2_main_table.json")))["per_chain"]["PACoAptamer"]
    stems = stems_193()

    per_chain, missing = {}, []
    for stem in stems:
        p = os.path.join(POOL_DIR, stem + ".json")
        if os.path.exists(p):
            v = json.load(open(p))
            per_chain[stem] = {"len": v["len"],
                               "recovery_best": v["recovery_best"],
                               "recovery_t01_mean": v["recovery_t01_mean"],
                               "logp_native": v["logp_native"],
                               "n_pool": len(v["pool"])}
        else:
            missing.append(stem)

    common = [s for s in stems if s in per_chain and s in main_tab]
    d_best = np.array([per_chain[s]["recovery_best"] - main_tab[s]["recovery_best"] for s in common])
    d_t01 = np.array([per_chain[s]["recovery_t01_mean"] - main_tab[s]["recovery_t01_mean"] for s in common])

    def wil(x):
        try:
            return float(wilcoxon(x, alternative="two-sided").pvalue)
        except Exception:
            return None

    # 预测骨架恢复率 vs RhoFold+ 质量（sctm_native = 预测结构与晶体 TM）
    xs = np.array([sctm[s]["sctm_native"] for s in common])
    y_best = np.array([per_chain[s]["recovery_best"] for s in common])
    y_t01 = np.array([per_chain[s]["recovery_t01_mean"] for s in common])
    spear = {
        "sctm_vs_predbb_best": {"rho": float(spearmanr(xs, y_best).statistic),
                                "p": float(spearmanr(xs, y_best).pvalue)},
        "sctm_vs_predbb_t01": {"rho": float(spearmanr(xs, y_t01).statistic),
                               "p": float(spearmanr(xs, y_t01).pvalue)},
        "sctm_vs_delta_best": {"rho": float(spearmanr(xs, d_best).statistic),
                               "p": float(spearmanr(xs, d_best).pvalue)},
        "sctm_vs_delta_t01": {"rho": float(spearmanr(xs, d_t01).statistic),
                              "p": float(spearmanr(xs, d_t01).pvalue)},
    }

    # ---- 复现校验：crystal 重跑池 vs 主表同链 ----
    val = {"note": "crystal = 94 号 crystal 相位（load_s1 缓存特征，ens2×300，seed42，新环境重跑）；"
                   "main_tab = btest_v2_main_table per_chain PACoAptamer（旧环境晶体骨架口径）"}
    cry = {}
    for stem in stems:
        p = os.path.join(POOL_DIR_CRY, stem + ".json")
        if os.path.exists(p):
            v = json.load(open(p))
            cry[stem] = (v["recovery_best"], v["recovery_t01_mean"])
    vc = [s for s in stems if s in cry and s in main_tab]
    if vc:
        cb = np.array([cry[s][0] for s in vc]); mb = np.array([main_tab[s]["recovery_best"] for s in vc])
        ct = np.array([cry[s][1] for s in vc]); mt = np.array([main_tab[s]["recovery_t01_mean"] for s in vc])
        val.update({
            "n_chains": len(vc),
            "crystal_rerun_best_mean": float(cb.mean()),
            "main_tab_best_mean_same_chains": float(mb.mean()),
            "crystal_rerun_t01_mean": float(ct.mean()),
            "main_tab_t01_mean_same_chains": float(mt.mean()),
            "delta_best_mean": float((cb - mb).mean()),
            "delta_t01_mean": float((ct - mt).mean()),
            "perchain_spearman_best": float(spearmanr(cb, mb).statistic),
            "perchain_spearman_t01": float(spearmanr(ct, mt).statistic),
            "perchain_pearson_best": float(np.corrcoef(cb, mb)[0, 1]),
        })
    val["main_tab_full477"] = {"best": 0.7490, "t01": 0.6040}

    out = {
        "task": "PACo ens2 × RhoFold+ 预测骨架（86_sctm native_*.pdb，193 链）；"
                "native 序列与 RNAfold(native) SS 与晶体口径一致，唯一差异=9 原子坐标来源",
        "n_chains": len(per_chain),
        "missing": missing,
        "predbb_summary": {
            "recovery_best_mean": float(np.mean([v["recovery_best"] for v in per_chain.values()])),
            "recovery_best_median": float(np.median([v["recovery_best"] for v in per_chain.values()])),
            "recovery_t01_mean": float(np.mean([v["recovery_t01_mean"] for v in per_chain.values()])),
            "recovery_t01_median": float(np.median([v["recovery_t01_mean"] for v in per_chain.values()])),
        },
        "delta_vs_crystal_maintab": {
            "n_common": len(common),
            "best": {"mean": float(d_best.mean()), "median": float(np.median(d_best)),
                     "ci95": boot_ci(d_best), "wilcoxon_p": wil(d_best)},
            "t01": {"mean": float(d_t01.mean()), "median": float(np.median(d_t01)),
                    "ci95": boot_ci(d_t01), "wilcoxon_p": wil(d_t01)},
        },
        "spearman_vs_sctm": spear,
        "validation": val,
        "per_chain": per_chain,
    }
    tmp = os.path.join(SUR, "94_predbb_paco.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUR, "94_predbb_paco.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "per_chain"}, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["crystal", "predbb", "summary"])
    args = ap.parse_args()
    if args.phase == "summary":
        phase_summary()
    else:
        run_pool(args.phase)
