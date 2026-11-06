#!/usr/bin/env python3
"""第 80 步：AptaDB 回顾性结果的机制诊断（79 号结果的补充对照）

79 号初版结果反直觉：适配体 logP 低于随机对照（仅 ~8% 反超）。三个诊断排除计算假象、
定位机制：

  D1  实验结构 vs 预测结构：PDB test 链 native 序列在自身实验 coords9 与 RhoFold+
      预测 coords9 下分别打分（rhofold_out 已有 286 条）——预测结构是否系统性抬高 logP
  D2  随机序列 + 自身预测结构：随机序列经 RhoFold+ 预测后打分——排除"结构为适配体
      量身定制"的混淆（若 random-own-structure ≈ random-under-aptamer-structure，
      则随机对照的高分不是结构适配造成的）
  D3  组成效应：适配体碱基组成（G/C 含量）与 logP 的相关——SELEX 序列 G 偏置假说

用法：CUDA_VISIBLE_DEVICES=1 python 80_aptadb_diag.py --diag 1|2|3
输出：survey/80_diag.json（追加式）
"""
import os, sys, json, argparse
import numpy as np

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, DATA_DIR)

import importlib.util
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m74 = _imp("m74", os.path.join(DATA_DIR, "74_p00_iface_ablation.py"))
m77 = _imp("m77", os.path.join(DATA_DIR, "77_p0234_analysis.py"))
m79 = _imp("m79", os.path.join(DATA_DIR, "79_aptadb_retro.py"))
OUT_J = os.path.join(SUR, "80_diag.json")


def get_scorer(device=0):
    import torch
    import torch.nn.functional as F
    from PACoAptamer.data_utils import ss_to_pairs
    models, alpha = [], None
    for n in m79.ENS_CKPTS:
        m, a = m74.load_model(os.path.join(ROOT, "models", "PACoAptamer", n + ".pth"), device)
        models.append(m); alpha = a
    alphabet = alpha
    bos = alphabet.get_idx("<cath>")

    @torch.no_grad()
    def logp(seqs, coords9, ss_pairs):
        L = coords9.shape[0]
        coords = torch.tensor(coords9, dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(ss_pairs, dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        B = len(seqs)
        prev = torch.zeros(B, L, dtype=torch.long, device=device)
        tgt = torch.zeros(B, L, dtype=torch.long, device=device)
        for bi, sq in enumerate(seqs):
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
        return lp.mean(dim=1).float().cpu().numpy()

    return logp, ss_to_pairs


def load_out():
    if os.path.exists(OUT_J):
        return json.load(open(OUT_J))
    return {}


def diag1(n_chains=100):
    """PDB test 链：native 序列在实验 coords9 vs RhoFold 预测 coords9 下的 logP"""
    from PACoAptamer.data_utils import load_s1
    stems = m77.test_stems()
    data = load_s1(set(stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    pairs = []
    for stem in stems:
        if stem not in data:
            continue
        p = os.path.join(SUR, "rhofold_out", stem + ".pdb")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            pairs.append(stem)
        if len(pairs) >= n_chains:
            break
    logp, ss_to_pairs = get_scorer()
    out = {"n": 0, "exp_per_nt": [], "pred_per_nt": []}
    for i, stem in enumerate(pairs):
        d = data[stem]
        if len(d["seq"]) > 512:
            continue
        c_pred = m79.coords9_from_pdb(os.path.join(SUR, "rhofold_out", stem + ".pdb"))
        if c_pred is None or c_pred.shape[0] != len(d["seq"]):
            continue
        ss = d["ss_pairs"]   # 缓存里已是 (L,L) 配对矩阵
        try:
            lp_exp = float(logp([d["seq"]], d["coords9"], ss)[0])
            lp_pred = float(logp([d["seq"]], c_pred, ss)[0])
        except RuntimeError:
            continue
        out["exp_per_nt"].append(lp_exp / len(d["seq"]))
        out["pred_per_nt"].append(lp_pred / len(d["seq"]))
        if (i + 1) % 20 == 0:
            print(f"[D1] {i+1}/{len(pairs)}", flush=True)
    out["n"] = len(out["exp_per_nt"])   # 实际进入统计的对数（跳过 >512/形状不匹配）
    d = np.array(out["exp_per_nt"]) - np.array(out["pred_per_nt"])
    out["exp_minus_pred_mean"] = float(d.mean())
    r = np.random.RandomState(0)
    bs = [np.mean(r.choice(d, len(d))) for _ in range(2000)]
    out["exp_minus_pred_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    o = load_out()
    o["D1_exp_vs_pred_structure"] = {k: v for k, v in out.items()}
    json.dump(o, open(OUT_J, "w"), indent=1)
    print(f"[D1] n={out['n']} exp−pred = {out['exp_minus_pred_mean']:+.4f}/nt"
          f" CI {out['exp_minus_pred_ci95']}")


def diag2(n_random=30):
    """随机序列：RhoFold 预测自身结构后打分 vs 借用适配体预测结构打分"""
    rng = np.random.RandomState(7)
    logp, ss_to_pairs = get_scorer()
    fd = os.path.join(SUR, "79_rhofold")
    recs = []
    keys = sorted(x[:-4] for x in os.listdir(fd) if x.startswith("apta") and x.endswith(".pdb"))
    import subprocess
    RHOFOLD_PY = "/root/miniconda3/envs/rhofold_service_env/bin/python"
    REPO = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_real_agent/rna_agent_system/external_repos/RhoFold"
    for k in keys[:n_random * 3]:   # 多取一些，长度过滤后保 n_random 量
        apta_seq = open(os.path.join(fd, k + ".fa")).read().splitlines()[1]
        L = len(apta_seq)
        if not (30 <= L <= 200):
            continue
        if len(recs) >= n_random:
            break
        seqs = ["".join(rng.choice(list("ACGU"), L))]
        # 1 条随机序列/适配体结构，RhoFold 预测
        fa = os.path.join(fd, f"rand_{k}.fa")
        with open(fa, "w") as f:
            f.write(f">rand_{k}\n{seqs[0]}\n")
        wd = os.path.join(fd, f"tmp_rand_{k}")
        r = subprocess.run([RHOFOLD_PY, "inference.py", "--input_fas", fa,
                            "--output_dir", wd, "--device", "cuda:0",
                            "--single_seq_pred", "True", "--relax_steps", "0",
                            "--ckpt", f"{REPO}/pretrained/RhoFold_pretrained.pt"],
                           cwd=REPO, capture_output=True, timeout=600)
        pred_p = os.path.join(wd, "unrelaxed_model.pdb")
        if r.returncode != 0 or not os.path.exists(pred_p):
            continue
        c_own = m79.coords9_from_pdb(pred_p)
        import shutil
        os.rename(pred_p, os.path.join(fd, f"rand_{k}.pdb"))
        shutil.rmtree(wd, ignore_errors=True)
        c_apta = m79.coords9_from_pdb(os.path.join(fd, k + ".pdb"))
        if c_own is None or c_own.shape[0] != L or c_apta is None or c_apta.shape[0] != L:
            continue
        ss_own = ss_to_pairs(m77.fold_batch(seqs)[0])
        ss_apta = ss_to_pairs(m77.fold_batch([apta_seq])[0])
        lp_own = float(logp(seqs, c_own, ss_own)[0])
        lp_borrow = float(logp(seqs, c_apta, ss_apta)[0])
        recs.append({"key": k, "lp_own_per_nt": lp_own / L, "lp_borrow_per_nt": lp_borrow / L})
    o = load_out()
    own = [r["lp_own_per_nt"] for r in recs]
    bor = [r["lp_borrow_per_nt"] for r in recs]
    o["D2_random_own_vs_borrowed"] = {
        "n": len(recs), "own_mean": float(np.mean(own)) if own else None,
        "borrowed_mean": float(np.mean(bor)) if bor else None}
    json.dump(o, open(OUT_J, "w"), indent=1)
    print(f"[D2] n={len(recs)} own={np.mean(own):+.4f} borrowed={np.mean(bor):+.4f} /nt")


def diag3():
    """组成效应：适配体 G/C 含量 vs logP；对照组组成固定 0.25"""
    sc = json.load(open(os.path.join(SUR, "79_aptadb_scored.json")))["entries"]
    d = json.load(open(m79.CLEAN_J))
    from scipy.stats import spearmanr
    rows = []
    for e in d["entries"]:
        # 与 79 号主分析同口径：剔除 cluster_train 泄漏与 exact-train（clean 214 条）
        if e["leak"]["cluster_train"] or e["leak"]["exact"] is not None:
            continue
        s = sc.get(e["key"])
        if s and "logp_self" in s:
            seq = e["seq"]
            rows.append({"gc": (seq.count("G") + seq.count("C")) / len(seq),
                         "g": seq.count("G") / len(seq),
                         "lp": s["logp_self_per_nt"]})
    rho_gc = spearmanr([r["gc"] for r in rows], [r["lp"] for r in rows]).statistic
    rho_g = spearmanr([r["g"] for r in rows], [r["lp"] for r in rows]).statistic
    o = load_out()
    o["D3_composition"] = {
        "n": len(rows),
        "mean_gc": float(np.mean([r["gc"] for r in rows])),
        "mean_g": float(np.mean([r["g"] for r in rows])),
        "spearman_gc_vs_logp": float(rho_gc),
        "spearman_g_vs_logp": float(rho_g),
        "gc_hist": {
            f"{lo:.1f}-{lo+0.1:.1f}": float(np.mean([r["lp"] for r in rows if lo <= r["gc"] < lo + 0.1]))
            for lo in np.arange(0.2, 0.9, 0.1)
            if any(lo <= r["gc"] < lo + 0.1 for r in rows)}}
    json.dump(o, open(OUT_J, "w"), indent=1)
    print(f"[D3] n={len(rows)} GC={np.mean([r['gc'] for r in rows]):.2f}"
          f" ρ(GC,logP)={rho_gc:.3f} ρ(G,logP)={rho_g:.3f}")
    print(o["D3_composition"]["gc_hist"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', type=int, required=True, choices=[1, 2, 3])
    args = ap.parse_args()
    {1: diag1, 2: diag2, 3: diag3}[args.diag]()


if __name__ == "__main__":
    main()
