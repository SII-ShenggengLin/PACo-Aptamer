#!/usr/bin/env python3
"""第 74 步：P0-0 界面口径消融重析（2026-08-18，蓝图 analysis/12 §3.1 决策实验）

背景：no_ligand/-0.84pp、no_iface/-1.00pp 在全局恢复率口径下为噪声级；但 B 层界面
绝大多数仅 1-2 残基（占位 2-3%），配体/界面模块即使完美起效，全局恢复率也只能动
0.6-1.0pp —— 现有消融口径物理上看不见界面效应。本脚本把消融放到界面恢复率口径下重评。

相位：
  --phase meta     界面元数据：test 477 链，35 号同款逻辑（功能配体 HETATM + 残基原子
                   5Å 接触）重建 interface 标注与配体原子，核对与 s1 缓存序列一致
                   → survey/74_iface_meta.json
  --phase eval     单 ckpt 38 号同口径评估（test 477，quick 10样本/温度×3温度=30 池），
                   额外保存完整序列池 → survey/74_pools/{name}.json
  --phase ligand   界面链三条件推理：空配体 / 真实配体 / 错配配体（供体坐标+当前 RNA
                   C1' 重算 RBF）→ survey/74_ligand/{name}.json
  --phase report   汇总：全局/界面双口径 + Δ + bootstrap CI + 配对 Wilcoxon
                   → survey/74_p00_iface_ablation.json

GPU：CUDA_VISIBLE_DEVICES=<物理号> python 74_p00_iface_ablation.py ...（脚本内 device=0）
环境：/root/miniconda3/envs/aptamer_eval/bin/python
"""
import os, sys, json, argparse, time, difflib
import numpy as np
import torch

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from PACoAptamer.model import PACoModel
from PACoAptamer.alphabet import Alphabet
from PACoAptamer.data_utils import load_s1, lig_features

import importlib.util
spec = importlib.util.spec_from_file_location("m35", os.path.join(DATA_DIR, "35_build_s2_samples.py"))
m35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m35)

TEMPS = [0.1, 0.5, 1.0]
N_QUICK = 10
MAX_LEN = 512
CONTACT = 5.0
C1P_IDX = 1   # coords9 = [C4', C1', N1, C2, C5', O5', P, C2', C3']


# ---------------------------------------------------------------- meta 相位
def load_func_ligs():
    cands = json.load(open(os.path.join(SUR, "candidates_ccd.json")))
    s = set()
    for c in cands:
        s.update(c["ligands"].get("other", []))
    return s


def extract_ligand_atoms(pdb_id, func_ligs):
    """35 号 work_entry 同款：功能配体 HETATM → (elems, coords)"""
    elems, coords = [], []
    path = os.path.join(m35.PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        return elems, np.zeros((0, 3), dtype=np.float32)
    for line in open(path):
        if line[:6].strip() != "HETATM" or line[16] not in (" ", "A"):
            continue
        if line[17:20].strip() not in func_ligs:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        elems.append(line[76:78].strip() or "C")
        coords.append((x, y, z))
    return elems, np.array(coords, dtype=np.float32)


def meta_one(args):
    pdb_id, chain, native, func_ligs = args
    from scipy.spatial import cKDTree
    stem = f"{pdb_id}_{chain}"
    elems, lig_coords = extract_ligand_atoms(pdb_id, func_ligs)
    if len(elems) == 0:
        return stem, None
    residues, seq35, _ = m35.extract_rna(pdb_id, chain)
    if seq35 is None or len(residues) != len(seq35):   # 蛋白痕迹/异构 guard
        return stem, None
    # 与 s1 缓存 native 对齐（一致 → 直接映射；否则 difflib；再否则不可用）
    mapping, how = None, None
    if seq35 == native:
        mapping, how = list(range(len(native))), "identical"
    else:
        sm = difflib.SequenceMatcher(None, seq35, native, autojunk=False)
        m2n = {}
        ok = True
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                m2n[a + k] = b + k
        for i in range(len(seq35)):
            if i not in m2n:
                ok = False
                break
        if ok and len(m2n) == len(native):
            mapping, how = [m2n[i] for i in range(len(seq35))], "difflib"
    if mapping is None:
        return stem, {"usable": False, "reason": "seq_mismatch",
                      "n_iface35": None}
    # 35 号同款界面：残基原子 vs 最近配体原子 < 5Å（seq35 坐标系）
    interface35 = np.zeros(len(seq35), dtype=bool)
    tree_lig = cKDTree(lig_coords)
    for i, (rn, rnum, ic, atoms) in enumerate(residues):
        pts = np.array(list(atoms.values()), dtype=np.float32) if atoms else np.empty((0, 3))
        if len(pts):
            d, _ = tree_lig.query(pts, k=1)
            if d.min() < CONTACT:
                interface35[i] = True
    # 映射到 native 坐标系（difflib 时 native 可能更长：未对上位置标 False）
    interface = np.zeros(len(native), dtype=bool)
    for i35, inat in enumerate(mapping):
        if interface35[i35]:
            interface[inat] = True
    return stem, {
        "usable": True, "align": how,
        "interface": interface.tolist(),
        "n_iface": int(interface.sum()),
        "lig_elems": elems, "lig_coords": lig_coords.tolist(),
    }


def phase_meta():
    from concurrent.futures import ProcessPoolExecutor
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test_stems = list(sp2["test"])
    data = load_s1(set(test_stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    test_stems = [s for s in test_stems if s in data]
    func_ligs = load_func_ligs()
    args_list = []
    for stem in test_stems:
        pdb_id, ch = stem.rsplit("_", 1)
        args_list.append((pdb_id, ch, data[stem]["seq"], func_ligs))
    out = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (stem, rec) in enumerate(ex.map(meta_one, args_list, chunksize=8)):
            out[stem] = rec
            if (i + 1) % 100 == 0:
                n_lig = sum(1 for v in out.values() if v)
                print(f"meta 进度 {i+1}/{len(args_list)}（含配体条目 {n_lig}）", flush=True)
    usable = {k: v for k, v in out.items() if v and v.get("usable")}
    n_iface = sum(1 for v in usable.values() if v["n_iface"] > 0)
    hist = {}
    for v in usable.values():
        n = min(v["n_iface"], 6)
        hist[n] = hist.get(n, 0) + 1
    meta_all = {"n_test": len(test_stems), "n_with_ligand": len(usable), "n_iface_chains": n_iface,
                "n_iface_hist(6=≥6)": hist, "stems": out}
    json.dump(meta_all, open(os.path.join(SUR, "74_iface_meta.json"), "w"))
    print(f"meta 完成：test {len(test_stems)} | 功能配体链 {len(usable)} | 界面链 {n_iface}")
    print(f"界面残基数分布(钳6): {hist}")


# ---------------------------------------------------------------- 模型/采样
def load_model(ckpt_path, device=0):
    args_cls = type('A', (), {'local_rank': -1, 'device_id': [0],
        'encoder_embed_dim': 512, 'decoder_embed_dim': 512, 'dropout': 0.1,
        'gvp_top_k_neighbors': 15, 'gvp_node_hidden_dim_vector': 256,
        'gvp_node_hidden_dim_scalar': 512, 'gvp_edge_hidden_dim_scalar': 32,
        'gvp_edge_hidden_dim_vector': 1, 'gvp_num_encoder_layers': 3,
        'gvp_dropout': 0.1, 'encoder_layers': 3, 'encoder_attention_heads': 4,
        'attention_dropout': 0.1, 'encoder_ffn_embed_dim': 512,
        'decoder_layers': 3, 'decoder_attention_heads': 4,
        'decoder_ffn_embed_dim': 512})()
    alphabet = Alphabet(['A', 'G', 'C', 'U', 'X'])
    model = PACoModel(args_cls, alphabet).cuda(device)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model, alphabet


@torch.no_grad()
def sample_chain(model, alphabet, d, device, temperature, n_samples,
                 no_ss=False, lig_feats_np=None):
    L = len(d['seq'])
    coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
    ss = None if no_ss else torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
    conf = torch.ones(1, L, device=device)
    if lig_feats_np is not None and len(lig_feats_np) > 0:
        lig = torch.tensor(np.asarray(lig_feats_np), dtype=torch.float32, device=device).unsqueeze(0)
        ligm = torch.ones(1, lig_feats_np.shape[0], dtype=torch.bool, device=device)
    else:
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
    try:
        preds = model.sample(coords, ss, lig, ligm, conf, device, temperature=temperature,
                             n_samples=n_samples)
    except RuntimeError as e:   # 个别链 NaN logits（no_aug 权重数值敏感）→ 上层记录跳过
        print(f"    [sample 失败 t={temperature}: {e}]", flush=True)
        return None
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    return [''.join(alphabet.get_tok(t) for t in row.tolist()) for row in preds]


def phase_eval(name, ckpt, no_ss, n_per_temp=N_QUICK):
    device = 0
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    test_stems = list(sp2["test"])
    data = load_s1(set(test_stems), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    test_stems = [s for s in test_stems if s in data]
    model, alphabet = load_model(ckpt, device)
    out = {}
    t0 = time.time()
    for i, stem in enumerate(test_stems):
        d = data[stem]
        if len(d['seq']) > MAX_LEN:
            continue
        native = d['seq']
        pool = []
        failed = False
        for t in TEMPS:
            seqs = sample_chain(model, alphabet, d, device, t, n_per_temp, no_ss)
            if seqs is None:
                failed = True
                break
            pool.extend(seqs)
        if failed:
            out[stem] = {"len": len(native), "native": native, "failed": True}
            print(f"    [eval 跳过失败链 {stem}]", flush=True)
            continue
        recs = [sum(1 for a, b in zip(s, native) if a == b) / len(native) for s in pool]
        out[stem] = {"len": len(native), "native": native, "pool": pool,
                     "n_per_temp": n_per_temp,
                     "recovery_best": max(recs), "recovery_t01_mean": float(np.mean(recs[:n_per_temp]))}
        if (i + 1) % 100 == 0:
            print(f"[{name}] {i+1}/{len(test_stems)} | {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.join(SUR, "74_pools"), exist_ok=True)
    json.dump(out, open(os.path.join(SUR, "74_pools", f"{name}.json"), "w"))
    rb = [v["recovery_best"] for v in out.values() if "recovery_best" in v]
    rt = [v["recovery_t01_mean"] for v in out.values() if "recovery_t01_mean" in v]
    print(f"[{name}] n={len(out)} n/temp={n_per_temp} best={np.mean(rb):.4f} t01={np.mean(rt):.4f}", flush=True)


def phase_ligand(name, ckpt, no_ss=False):
    """界面链三条件：none / true / mismatch（错配供体 = 其他条目配体坐标，RBF 对当前 C1' 重算）"""
    device = 0
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    usable = {k: v for k, v in meta.items()
              if v and v.get("usable") and 0 < v["n_iface"] <= 50}   # 同 report 相位口径
    stems = sorted(usable)
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    data = load_s1(set(sp2["test"]), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
    stems = [s for s in stems if s in data]
    donors = [s for s in stems]
    model, alphabet = load_model(ckpt, device)
    rng = np.random.RandomState(0)
    out = {}
    t0 = time.time()
    for i, stem in enumerate(stems):
        d = data[stem]
        if len(d['seq']) > MAX_LEN:
            continue
        c1p = np.asarray(d['coords9'], dtype=np.float32)[:, C1P_IDX, :]
        # to_paco9 把缺失原子转成了 NaN；训练期（35 号）用原始 FILL=1e-5 算距离
        # → 恢复同款语义，否则 NaN 传入 RBF 污染 lig_feats 导致解码 NaN
        c1p = np.nan_to_num(c1p, nan=1e-5)
        m = usable[stem]
        lig_true = lig_features(np.array(m["lig_coords"], dtype=np.float32),
                                m["lig_elems"], c1p)
        # 错配供体（固定种子、异 PDB）
        k = rng.randint(len(donors))
        for _ in range(10):
            donor = donors[k]
            if donor.rsplit("_", 1)[0] != stem.rsplit("_", 1)[0]:
                break
            k = (k + 1) % len(donors)
        md = usable[donor]
        lig_mis = lig_features(np.array(md["lig_coords"], dtype=np.float32),
                               md["lig_elems"], c1p)
        rec = {}
        skip = False
        for cond, lf in (("none", None), ("true", lig_true), ("mismatch", lig_mis)):
            pool = []
            for t in TEMPS:
                seqs = sample_chain(model, alphabet, d, device, t, N_QUICK,
                                    no_ss, lf)
                if seqs is None:
                    skip = True
                    break
                pool.extend(seqs)
            if skip:
                break
            rec[cond] = pool
        if skip:
            print(f"    [ligand 跳过失败链 {stem}]", flush=True)
            continue
        out[stem] = {"len": len(d['seq']), "native": d['seq'],
                     "interface": m["interface"], "donor": donor,
                     "pools": rec}
        if (i + 1) % 10 == 0:
            print(f"[{name}:ligand] {i+1}/{len(stems)} | {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.join(SUR, "74_ligand"), exist_ok=True)
    json.dump(out, open(os.path.join(SUR, "74_ligand", f"{name}.json"), "w"))
    print(f"[{name}:ligand] 完成 {len(out)} 链 × 3 条件", flush=True)


# ---------------------------------------------------------------- report
def pool_metrics(pool, native, iface_pos, n_q=N_QUICK):
    """64 号口径：池内 best（界面/整体独立取 max）；N 位不进界面分母"""
    npos = set(i for i, c in enumerate(native) if c != "N")
    ipos = sorted(set(iface_pos) & npos)
    best_if = best_ov = -1.0
    t01_if = t01_ov = None
    ifs, ovs = [], []
    for si, s in enumerate(pool):
        if len(s) != len(native):
            continue
        ov = sum(1 for i in npos if s[i] == native[i]) / len(npos)
        iv = (sum(1 for i in ipos if s[i] == native[i]) / len(ipos)) if ipos else None
        ovs.append(ov)
        if iv is not None:
            ifs.append(iv)
        best_ov = max(best_ov, ov)
        if iv is not None:
            best_if = max(best_if, iv)
    if ovs[:n_q]:
        t01_ov = float(np.mean(ovs[:n_q]))
    if ifs[:n_q]:
        t01_if = float(np.mean(ifs[:n_q]))
    return (best_if if best_if >= 0 else None, best_ov if best_ov >= 0 else None,
            t01_if, t01_ov)


def boot_ci(x, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def phase_report():
    from scipy.stats import wilcoxon
    meta = json.load(open(os.path.join(SUR, "74_iface_meta.json")))["stems"]
    pool_dir = os.path.join(SUR, "74_pools")
    lig_dir = os.path.join(SUR, "74_ligand")
    report = {"phase1_ablation": {}, "phase2_ligand": {}}

    iface_meta = {k: v for k, v in meta.items()
                  if v and v.get("usable") and 0 < v["n_iface"] <= 50}   # >50 = 拼接假序列等异常（1EI2_A=291）
    n3 = {k: v for k, v in iface_meta.items() if v["n_iface"] >= 3}
    print(f"界面链 {len(iface_meta)}（≥3 残基 {len(n3)}；剔除异常 >50 残基："
          f"{[k for k, v in meta.items() if v and v.get('usable') and v['n_iface'] > 50]}）")

    # ---- Phase 1
    per_chain_all = {}
    for fn in sorted(os.listdir(pool_dir)):
        name = fn[:-5]
        pools = json.load(open(os.path.join(pool_dir, fn)))
        n_q = next((v.get("n_per_temp") for v in pools.values() if v.get("n_per_temp")), N_QUICK)
        # 全局复现口径（failed 链跳过）
        rb = [v["recovery_best"] for v in pools.values() if "recovery_best" in v]
        rt = [v["recovery_t01_mean"] for v in pools.values() if "recovery_t01_mean" in v]
        n_fail = sum(1 for v in pools.values() if v.get("failed"))
        row = {"n_477": len(pools), "n_failed": n_fail,
               "global_best_477": float(np.mean(rb)),
               "global_t01_477": float(np.mean(rt))}
        per_chain = {}
        for sub_name, sub in (("iface", iface_meta), ("iface_ge3", n3)):
            ifs_b, ovs_b, difs, ifs_t, ovs_t, chain_recs = [], [], [], [], [], []
            for stem, mv in sub.items():
                if stem not in pools or "pool" not in pools[stem]:
                    continue
                v = pools[stem]
                bi, bo, ti, to = pool_metrics(v["pool"], v["native"], n_q=n_q,
                                              iface_pos=[i for i, x in enumerate(mv["interface"]) if x])
                if bi is None:
                    continue
                ifs_b.append(bi); ovs_b.append(bo); difs.append(bi - bo)
                chain_recs.append((stem, bi, bo))
                if ti is not None:
                    ifs_t.append(ti); ovs_t.append(to)
            if sub_name == "iface":
                per_chain = {s: {"iface_best": a, "overall_best": b} for s, a, b in chain_recs}
            row[sub_name] = {
                "n": len(ifs_b),
                "iface_best": float(np.mean(ifs_b)), "overall_best": float(np.mean(ovs_b)),
                "iface_minus_overall": float(np.mean(difs)),
                "iface_minus_overall_ci95": boot_ci(difs),
                "iface_t01": float(np.mean(ifs_t)) if ifs_t else None,
                "overall_t01": float(np.mean(ovs_t)) if ovs_t else None,
            }
        report["phase1_ablation"][name] = row
        per_chain_all[name] = per_chain

    # Phase1 界面对比（模型间配对 Wilcoxon：iface_best）
    names = list(report["phase1_ablation"])
    if "baseline" in names:
        for nm in names:
            if nm == "baseline":
                continue
            common = [s for s in per_chain_all.get("baseline", {})
                      if s in per_chain_all.get(nm, {})]
            if len(common) >= 10:
                a = [per_chain_all["baseline"][s]["iface_best"] for s in common]
                b = [per_chain_all[nm][s]["iface_best"] for s in common]
                try:
                    p = float(wilcoxon(a, b, alternative="greater").pvalue)
                except Exception:
                    p = None
                d = float(np.mean(np.array(a) - np.array(b)))
                report["phase1_ablation"][nm]["vs_baseline_iface"] = {
                    "n": len(common), "delta_iface_best": d, "wilcoxon_p_one_sided": p}

    # ---- Phase 2（--phase ligand 的产物；目录不存在时跳过）
    for fn in (sorted(os.listdir(lig_dir)) if os.path.isdir(lig_dir) else []):
        name = fn[:-5]
        d = json.load(open(os.path.join(lig_dir, fn)))
        conds = ("none", "true", "mismatch")
        agg = {c: {"iface_b": [], "ov_b": [], "iface_t": [], "ov_t": []} for c in conds}
        for stem, v in d.items():
            iface_pos = [i for i, x in enumerate(v["interface"]) if x]
            for c in conds:
                bi, bo, ti, to = pool_metrics(v["pools"][c], v["native"], iface_pos)
                if bi is None:
                    continue
                agg[c]["iface_b"].append(bi); agg[c]["ov_b"].append(bo)
                if ti is not None:
                    agg[c]["iface_t"].append(ti); agg[c]["ov_t"].append(to)
        row = {}
        for c in conds:
            row[c] = {"n": len(agg[c]["iface_b"]),
                      "iface_best": float(np.mean(agg[c]["iface_b"])),
                      "overall_best": float(np.mean(agg[c]["ov_b"])),
                      "iface_t01": float(np.mean(agg[c]["iface_t"])) if agg[c]["iface_t"] else None}
        # 配对效应
        for ca, cb in (("true", "none"), ("true", "mismatch"), ("mismatch", "none")):
            dif, difov = [], []
            for stem, v in d.items():
                iface_pos = [i for i, x in enumerate(v["interface"]) if x]
                ia, oa, _, _ = pool_metrics(v["pools"][ca], v["native"], iface_pos)
                ib, ob, _, _ = pool_metrics(v["pools"][cb], v["native"], iface_pos)
                if ia is None or ib is None:
                    continue
                dif.append(ia - ib); difov.append(oa - ob)
            key = f"delta_{ca}_minus_{cb}"
            try:
                p = float(wilcoxon(dif, alternative="greater").pvalue) if len(dif) >= 10 else None
            except Exception:
                p = None
            row[key] = {"n": len(dif), "iface_best": float(np.mean(dif)) if dif else None,
                        "iface_ci95": boot_ci(dif), "overall_best": float(np.mean(difov)) if difov else None,
                        "wilcoxon_p_one_sided": p}
        report["phase2_ligand"][name] = row

    json.dump(report, open(os.path.join(SUR, "74_p00_iface_ablation.json"), "w"), indent=1)
    print(json.dumps(report, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['meta', 'eval', 'ligand', 'report'])
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--no_ss', action='store_true')
    ap.add_argument('--n', type=int, default=N_QUICK, help='每温度样本数（10=quick 30 池；100=主表 300 池）')
    args = ap.parse_args()
    if args.phase == 'meta':
        phase_meta()
    elif args.phase == 'eval':
        phase_eval(args.name, args.ckpt, args.no_ss, args.n)
    elif args.phase == 'ligand':
        phase_ligand(args.name, args.ckpt, args.no_ss)
    elif args.phase == 'report':
        phase_report()


if __name__ == "__main__":
    main()
