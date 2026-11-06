#!/usr/bin/env python3
"""第 96 步：配体 token 注意力可解释性（74 号 115 条界面链）

问题：解码界面位时模型是否在看配体 token？
方法：teacher-forced 喂 native 序列，forward hook 收 decoder 每层 encoder_attn
（cross-attention）返回的注意力权重 (B, L_tgt, L_src)；encoder 输出最后 n_lig 个
位置 = 配体 token，其注意力质量按核苷酸位置求和 → 逐位"配体注意力质量"(L,)，
对 3 层取平均；ens2 = 两模型注意力均值（与概率平均集成语义平行）。

三条件：
  ens2     主配置（s2_v2_seed43_noiface + s2_v2_ssdo15_noiface）+ 真实配体
  shuffle  同上但配体 token 顺序随机置换（torch 种子固定）——总质量应不变，验证指标
  seed43   s2_v2_seed43_noiface 单模对照 + 真实配体

统计（每链）：
  - 界面位 vs 非界面位配体注意力质量差 Δattn（跨链配对 Wilcoxon，单侧 greater）
  - 逐位注意力质量 vs 该位到配体最近原子距离的 Spearman（77 号 dist 逻辑）
输出：survey/96_attention.json（per_chain + summary + 8HZD_A 逐位明细）
用法：CUDA_VISIBLE_DEVICES=0 python data/96_attention_ligand.py
"""
import os, sys, json, time, difflib
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
m35 = m74.m35
from PACoAptamer.data_utils import load_s1, lig_features

C1P_IDX = 1
MAX_LEN = 512
CKPT_ENS2 = ["s2_v2_seed43_noiface", "s2_v2_ssdo15_noiface"]


def usable_iface_stems():
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    return {k: v for k, v in meta.items()
            if v and v.get("usable") and 0 < v["n_iface"] <= 50}


def dist_to_ligand(stem, native):
    """77 号 phase_dist 同款：逐位到配体最近原子距离（native 坐标系）"""
    from scipy.spatial import cKDTree
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    mv = meta[stem]
    pdb_id, ch = stem.rsplit("_", 1)
    residues, seq35, _ = m35.extract_rna(pdb_id, ch)
    if seq35 is None or len(residues) != len(seq35):
        return None
    if seq35 == native:
        mapping = list(range(len(native)))
    else:
        sm = difflib.SequenceMatcher(None, seq35, native, autojunk=False)
        m2n = {}
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                m2n[a + k] = b + k
        if len(m2n) != len(native):
            return None
        mapping = [m2n[i] for i in range(len(seq35))]
    tree_lig = cKDTree(np.array(mv["lig_coords"], dtype=np.float32))
    dists = np.full(len(native), np.nan)
    for i, (_, _, _, atoms) in enumerate(residues):
        pts = np.array(list(atoms.values()), dtype=np.float32) if atoms else np.empty((0, 3))
        if len(pts):
            dd, _ = tree_lig.query(pts, k=1)
            dists[mapping[i]] = float(dd.min())
    return dists


def main():
    import torch
    device = 0
    torch.manual_seed(42)
    meta = usable_iface_stems()
    stems = sorted(meta.keys())
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    data = load_s1(set(sp2["test"]), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    stems = [s for s in stems if s in data and len(data[s]['seq']) <= MAX_LEN]
    print(f"[attn] 界面链 {len(stems)}", flush=True)

    models = {}
    for n in CKPT_ENS2:
        p = os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
        m, alphabet = m74.load_model(p, device)
        models[n] = m

    # hook：收每层 cross-attention 权重（eval 模式 need_attn=True → out[1] 非 None）
    def collect_attn(model, coords, ss, lig, ligm, conf, pmask, prev):
        store = {}
        handles = []

        def mk(idx):
            def hook(mod, inp, out):
                if out[1] is not None:
                    store[idx] = out[1].detach().float().cpu().numpy()
            return hook
        for i, layer in enumerate(model.decoder.layers):
            if getattr(layer, "encoder_attn", None) is not None:
                handles.append(layer.encoder_attn.register_forward_hook(mk(i)))
        try:
            eo = model.forward_embedding(coords, ss, lig, ligm, pmask, conf)
            with torch.no_grad():
                model.decoder(prev, encoder_out=eo)
        finally:
            for h in handles:
                h.remove()
        return store   # idx → (B, L_tgt, L_src)

    @torch.no_grad()
    def ligand_attn_mass(model, d, lig_np, shuffle_seed=None):
        """逐位配体注意力质量 (L,)：层平均、头已由模块平均、对 n_lig 列求和"""
        L = len(d['seq'])
        coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        n_lig = lig_np.shape[0]
        if shuffle_seed is not None:
            g = torch.Generator().manual_seed(shuffle_seed)
            perm = torch.randperm(n_lig, generator=g)
            lig_np = lig_np[perm.numpy()]
        lig = torch.tensor(lig_np, dtype=torch.float32, device=device).unsqueeze(0)
        ligm = torch.ones(1, n_lig, dtype=torch.bool, device=device)
        bos = alphabet.get_idx("<cath>")
        prev = torch.zeros(1, L, dtype=torch.long, device=device)
        toks = [alphabet.get_idx(c) for c in d['seq']]
        prev[0, 0] = bos
        prev[0, 1:] = torch.tensor(toks[:-1], device=device)
        store = collect_attn(model, coords, ss, lig, ligm, conf, pmask, prev)
        if not store:
            return None
        # (n_layers, L_tgt, L_src) → 层平均 → 配体列求和
        A = np.stack([store[i][0] for i in sorted(store.keys())]).mean(axis=0)  # (L, L+n_lig)
        return A[:, L:].sum(axis=1)   # (L,)

    t0 = time.time()
    per_chain = {}
    case_detail = None
    for ci, stem in enumerate(stems):
        d = data[stem]
        native = d['seq']
        L = len(native)
        mv = meta[stem]
        iface = np.array(mv["interface"], dtype=bool)
        c1p = np.nan_to_num(np.asarray(d['coords9'], dtype=np.float32)[:, C1P_IDX, :], nan=1e-5)
        lig_np = lig_features(np.array(mv["lig_coords"], dtype=np.float32),
                              mv["lig_elems"], c1p)
        if lig_np.shape[0] == 0:
            continue
        # ens2：两模型注意力平均；shuffle 同 ens2 但配体 token 置换；seed43 单模
        m_ens = [ligand_attn_mass(models[n], d, lig_np) for n in CKPT_ENS2]
        if any(x is None for x in m_ens):
            continue
        mass_ens2 = np.stack(m_ens).mean(axis=0)
        mass_shuf = np.stack([ligand_attn_mass(models[n], d, lig_np, shuffle_seed=1234)
                              for n in CKPT_ENS2]).mean(axis=0)
        mass_seed43 = ligand_attn_mass(models[CKPT_ENS2[0]], d, lig_np)

        dists = dist_to_ligand(stem, native)
        noniface = ~iface
        rec = {
            "len": L, "n_lig": int(lig_np.shape[0]), "n_iface": int(iface.sum()),
            "attn_iface_mean": float(mass_ens2[iface].mean()),
            "attn_noniface_mean": float(mass_ens2[noniface].mean()),
            "delta_iface_minus_noniface": float(mass_ens2[iface].mean() - mass_ens2[noniface].mean()),
            "delta_shuffle": float(mass_shuf[iface].mean() - mass_shuf[noniface].mean()),
            "delta_seed43": float(mass_seed43[iface].mean() - mass_seed43[noniface].mean()),
            "attn_total_mean": float(mass_ens2.mean()),
            "attn_total_mean_shuffle": float(mass_shuf.mean()),
        }
        if dists is not None:
            from scipy.stats import spearmanr
            valid = np.isfinite(dists)
            if valid.sum() > 10 and np.std(mass_ens2[valid]) > 0 and np.std(dists[valid]) > 0:
                r = spearmanr(mass_ens2[valid], dists[valid])
                rec["spearman_attn_vs_dist"] = float(r.statistic)
                rec["spearman_p"] = float(r.pvalue)
        per_chain[stem] = rec

        if stem == "8HZD_A":
            case_detail = {
                "native": native, "n_lig": int(lig_np.shape[0]),
                "interface": iface.tolist(),
                "dist_to_ligand": dists.tolist() if dists is not None else None,
                "attn_mass_ens2": mass_ens2.tolist(),
                "attn_mass_shuffle": mass_shuf.tolist(),
                "attn_mass_seed43": mass_seed43.tolist(),
            }
        if (ci + 1) % 25 == 0:
            print(f"[attn] {ci+1}/{len(stems)} | {time.time()-t0:.0f}s", flush=True)

    # ---- 汇总 ----
    from scipy.stats import wilcoxon
    chains = list(per_chain.values())
    d_ens = np.array([c["delta_iface_minus_noniface"] for c in chains])
    d_shuf = np.array([c["delta_shuffle"] for c in chains])
    d_s43 = np.array([c["delta_seed43"] for c in chains])
    rhos = np.array([c["spearman_attn_vs_dist"] for c in chains if "spearman_attn_vs_dist" in c])

    def wil(x, alt="greater"):
        try:
            return float(wilcoxon(x, alternative=alt).pvalue)
        except Exception:
            return None

    def boot_ci(x, n=2000, seed=0):
        rng = np.random.RandomState(seed)
        x = np.asarray(x, dtype=float)
        m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    summary = {
        "n_chains": len(chains),
        "ens2_delta_iface_minus_noniface": {
            "mean": float(d_ens.mean()), "median": float(np.median(d_ens)),
            "ci95": boot_ci(d_ens), "wilcoxon_p_greater": wil(d_ens),
            "frac_positive": float((d_ens > 0).mean())},
        "shuffle_delta": {
            "mean": float(d_shuf.mean()), "median": float(np.median(d_shuf)),
            "ci95": boot_ci(d_shuf), "wilcoxon_p_greater": wil(d_shuf)},
        "seed43_delta": {
            "mean": float(d_s43.mean()), "median": float(np.median(d_s43)),
            "ci95": boot_ci(d_s43), "wilcoxon_p_greater": wil(d_s43)},
        "shuffle_total_mass_check": {
            "mean_abs_diff_vs_ens2": float(np.mean(np.abs(
                [c["attn_total_mean"] - c["attn_total_mean_shuffle"] for c in chains]))),
            "note": "置换配体 token 后总注意力质量应≈不变（指标稳健性）"},
        "spearman_attn_vs_dist": {
            "n": int(len(rhos)), "mean": float(rhos.mean()) if len(rhos) else None,
            "median": float(np.median(rhos)) if len(rhos) else None,
            "frac_negative": float((rhos < 0).mean()) if len(rhos) else None,
            "note": "预期负相关：离配体越近注意力越大"},
    }
    out = {"summary": summary, "per_chain": per_chain,
           "case_8HZD_A": case_detail,
           "method": "teacher-forced native；hook 收 decoder 每层 encoder_attn 权重；"
                     "配体注意力质量 = 最后 n_lig 个 encoder 位置的注意力列求和，层平均；"
                     "ens2 = 两模型注意力均值；shuffle 种子 1234；teacher-forced 全局种子 42"}
    tmp = os.path.join(SUR, "96_attention.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(SUR, "96_attention.json"))
    print(json.dumps(summary, indent=1), flush=True)


def phase_case8hzd():
    """8HZD_A 逐位注意力明细（A 层链，不在 74 号 meta/test 里，现场建特征+界面标注）"""
    import torch
    device = 0
    torch.manual_seed(42)
    m68 = _imp("m68", os.path.join(DATA_DIR, "68_eval_paco_av2.py"))
    stem = "8HZD_A"
    d, status = m68.build_feature(stem)
    assert d is not None, status
    native = d['seq']
    L = len(native)
    # 界面标注（74 号 meta_one 同款逻辑）
    func_ligs = m74.load_func_ligs()
    elems, lig_coords = m74.extract_ligand_atoms("8HZD", func_ligs)
    from scipy.spatial import cKDTree
    residues, seq35, _ = m35.extract_rna("8HZD", "A")
    assert seq35 == native and len(residues) == len(native), "8HZD_A 序列不一致"
    tree_lig = cKDTree(lig_coords)
    iface = np.zeros(L, dtype=bool)
    dists = np.full(L, np.nan)
    for i, (_, _, _, atoms) in enumerate(residues):
        pts = np.array(list(atoms.values()), dtype=np.float32) if atoms else np.empty((0, 3))
        if len(pts):
            dd, _ = tree_lig.query(pts, k=1)
            dists[i] = float(dd.min())
            if dd.min() < 5.0:
                iface[i] = True
    print(f"[case8hzd] L={L} n_lig={len(elems)} n_iface={iface.sum()}", flush=True)

    models = {}
    for n in CKPT_ENS2:
        p = os.path.join(ROOT, "models", "PACoAptamer", n + ".pth")
        m, alphabet = m74.load_model(p, device)
        models[n] = m

    def collect_attn(model, coords, ss, lig, ligm, conf, pmask, prev):
        store = {}
        handles = []

        def mk(idx):
            def hook(mod, inp, out):
                if out[1] is not None:
                    store[idx] = out[1].detach().float().cpu().numpy()
            return hook
        for i, layer in enumerate(model.decoder.layers):
            if getattr(layer, "encoder_attn", None) is not None:
                handles.append(layer.encoder_attn.register_forward_hook(mk(i)))
        try:
            eo = model.forward_embedding(coords, ss, lig, ligm, pmask, conf)
            with torch.no_grad():
                model.decoder(prev, encoder_out=eo)
        finally:
            for h in handles:
                h.remove()
        return store

    @torch.no_grad()
    def mass_for(model, lig_np, shuffle_seed=None):
        coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
        ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
        conf = torch.ones(1, L, device=device)
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        n_lig = lig_np.shape[0]
        if shuffle_seed is not None:
            g = torch.Generator().manual_seed(shuffle_seed)
            lig_np = lig_np[torch.randperm(n_lig, generator=g).numpy()]
        lig = torch.tensor(lig_np, dtype=torch.float32, device=device).unsqueeze(0)
        ligm = torch.ones(1, n_lig, dtype=torch.bool, device=device)
        bos = alphabet.get_idx("<cath>")
        prev = torch.zeros(1, L, dtype=torch.long, device=device)
        toks = [alphabet.get_idx(c) for c in native]
        prev[0, 0] = bos
        prev[0, 1:] = torch.tensor(toks[:-1], device=device)
        store = collect_attn(model, coords, ss, lig, ligm, conf, pmask, prev)
        A = np.stack([store[i][0] for i in sorted(store.keys())]).mean(axis=0)
        return A[:, L:].sum(axis=1)

    c1p = np.nan_to_num(np.asarray(d['coords9'], dtype=np.float32)[:, C1P_IDX, :], nan=1e-5)
    lig_np = lig_features(lig_coords, elems, c1p)
    mass_ens2 = np.stack([mass_for(models[n], lig_np) for n in CKPT_ENS2]).mean(axis=0)
    mass_shuf = np.stack([mass_for(models[n], lig_np, 1234) for n in CKPT_ENS2]).mean(axis=0)
    mass_s43 = mass_for(models[CKPT_ENS2[0]], lig_np)

    case = {
        "native": native, "n_lig": int(lig_np.shape[0]), "n_iface": int(iface.sum()),
        "interface": iface.tolist(), "dist_to_ligand": dists.tolist(),
        "attn_mass_ens2": mass_ens2.tolist(),
        "attn_mass_shuffle": mass_shuf.tolist(),
        "attn_mass_seed43": mass_s43.tolist(),
        "note": "8HZD_A 为 A 层链（grnade_inputs_v2 晶体 PDB + RNAfold SS 特征），"
                "界面=配体原子 5Å 接触（74 号同款）；非 test 115 链统计成员",
    }
    out_p = os.path.join(SUR, "96_attention.json")
    out = json.load(open(out_p))
    out["case_8HZD_A"] = case
    tmp = out_p + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, out_p)
    print(f"[case8hzd] 已写入 96_attention.json；iface 位均质量 "
          f"{mass_ens2[iface].mean():.4f} vs 非界面 {mass_ens2[~iface].mean():.4f}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main", choices=["main", "case8hzd"])
    args = ap.parse_args()
    if args.phase == "main":
        main()
    else:
        phase_case8hzd()
