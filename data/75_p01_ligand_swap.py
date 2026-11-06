#!/usr/bin/env python3
"""第 75 步：P0-1 错配配体负对照（升级）+ 条件跟随性 + 同结构多配体辨别
（蓝图 analysis/12 §五 P0-1；P0-0 §三.3 把主战场调整为"同结构多配体辨别"）

相对 P0-0 Phase 2 的三处关键升级：
1. 口袋叠合置换：P0-0 的错配配体用供体原始坐标（PDB 坐标系不同 → 配体常在
   几十 Å 外，等效"远处配体≈空配体"）。本脚本用家族内 C1' Kabsch 叠合把供体
   配体移植进当前口袋 → 真正的"同口袋不同化学"counterfactual。
2. 主战场 = 同骨架多配体辨别：序列同源聚类得"家族"，家族内跨 CCD 配体置换
   grid + 特异性位点条件偏好（SeFMol 式条件跟随性，Fig4h）。
3. teacher-forced 似然探针：逐位 logP(native)（无采样噪声，一次前向）。

相位：
  --phase meta      界面/配体元数据（test 477 + indep∪zs；配体按 CCD 分组、
                   只留链口袋 10Å 内原子）→ survey/75_meta.json + 75_feats.pt
  --phase families  家族聚类（成对 difflib 一致性 ≥0.4 且覆盖 ≥0.8 单连通；
                   家族 ≥3 链、≥2 种 CCD 组合）→ survey/75_families.json
  --phase align     家族内 Kabsch 叠合 + 置换 grid（true / geo[同CCD异条目] /
                   swap×≤4[异CCD，口袋叠合] / none）→ survey/75_grid.json
  --phase score     teacher-forced 逐位 logP（模型: ens2noiface / noligand）
                   → survey/75_score/{model}__{stem}.json
  --phase sample    集成采样（purine 家族 100/温度，其余 30/温度，3 温度）
                   → survey/75_pools/{model}__{stem}.json
  --phase report    统计汇总 → survey/75_p01_report.json

GPU：CUDA_VISIBLE_DEVICES=<物理号>（脚本内 device=0）；环境 aptamer_eval
"""
import os, sys, json, argparse, time, difflib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
DATA_DIR = os.path.join(ROOT, "data")
SUR = os.path.join(DATA_DIR, "survey")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from PACoAptamer.data_utils import lig_features

import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m35 = _load("m35", os.path.join(DATA_DIR, "35_build_s2_samples.py"))
m68 = _load("m68", os.path.join(DATA_DIR, "68_eval_paco_av2.py"))
m74 = _load("m74", os.path.join(DATA_DIR, "74_p00_iface_ablation.py"))

TEMPS = [0.1, 0.5, 1.0]
MAX_LEN = 512
CONTACT = 5.0
POCKET_CUT = 10.0     # 配体原子归属当前链口袋的距离阈值
IDENT_MIN = 0.40      # 家族聚类：difflib ratio 下限
COVER_MIN = 0.60      # 对齐覆盖下限（NW 口径；add↔xpt 亚家族 cov≈0.64）
RMSD_MAX = 4.0        # 叠合质量：C1' RMSD 上限
CENTROID_MAX = 6.0    # 移植配体质心 vs 原配体质心距离上限
MAX_SWAP = 4          # 每链最多异 CCD 置换条件数
N_IFACE_MAX = 50      # >50 残基界面 = 拼接假序列（74 号口径）
C1P_IDX = 1

CKPT_DIR = os.path.join(ROOT, "models", "PACoAptamer")
MODEL_SETS = {
    "ens2noiface": [os.path.join(CKPT_DIR, f) for f in
                    ("s2_v2_seed43_noiface.pth", "s2_v2_ssdo15_noiface.pth")],
    "noligand": [os.path.join(CKPT_DIR, "s2_v2_no_ligand.pth")],
    "seed43noiface": [os.path.join(CKPT_DIR, "s2_v2_seed43_noiface.pth")],
}

# 家族命名（report 可读性；聚类编号 → 生物学名，按种子链 PDB）
FAMILY_NAMES = {
    "1U8D": "purine_riboswitch", "1Y26": "purine_riboswitch", "6UBU": "purine_riboswitch",
    "7ELP": "xanthine_riboswitch", "3SLM": "dG_riboswitch",
    "1O15": "theophylline_aptamer", "8D5L": "theophylline_aptamer",
    "4XW7": "ztp_riboswitch", "5BTP": "ztp_riboswitch", "6WZR": "ztp_riboswitch",
    "6E8T": "mango3_aptamer", "6E8S": "mango3_aptamer", "6UP0": "mango3_aptamer",
    "6E1S": "mango3_aptamer",
    "8XZL": "folate_riboswitch", "8XZE": "folate_riboswitch",
    "8HZD": "fluo_aptamer_8hz", "9BUN": "rhobast_aptamer", "8JY0": "rhobast_aptamer",
}


# ---------------------------------------------------------------- meta
def load_func_ligs():
    cands = json.load(open(os.path.join(SUR, "candidates_ccd.json")))
    s = set()
    for c in cands:
        s.update(c["ligands"].get("other", []))
    return s


def extract_ligands_per_ccd(pdb_id, chain_residue_pts, func_ligs):
    """功能配体 HETATM 按 CCD 分组；只留距链任何原子 <POCKET_CUT 的原子。
    chain_residue_pts: list of (n_atoms_i, 3) 各残基原子坐标（35 号坐标系）"""
    by_ccd = {}
    path = os.path.join(m35.PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        return by_ccd
    allpts = np.concatenate(chain_residue_pts, axis=0) if chain_residue_pts else \
        np.zeros((0, 3), dtype=np.float32)
    from scipy.spatial import cKDTree
    tree = cKDTree(allpts) if len(allpts) else None
    for line in open(path):
        if line[:6].strip() != "HETATM" or line[16] not in (" ", "A"):
            continue
        if line[17:20].strip() not in func_ligs:
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        if tree is not None:
            d, _ = tree.query(xyz, k=1)
            if d > POCKET_CUT:
                continue
        ccd = line[17:20].strip()
        rec = by_ccd.setdefault(ccd, {"elems": [], "coords": []})
        rec["elems"].append(line[76:78].strip() or "C")
        rec["coords"].append(xyz.tolist())
    return by_ccd


def align_35_to_native(seq35, native):
    """35 号坐标系 → native 的映射。identical 直通；否则 difflib。
    接受条件：映射单射且覆盖 ≥90% native 位点（grnade 管线可能丢 1-2 个
    未解析残基 → seq35 比 native 长，属正常；未覆盖 native 位点界面记 False）。
    返回 {i35: inat} 或 None"""
    if seq35 == native:
        return {i: i for i in range(len(native))}, "identical"
    sm = difflib.SequenceMatcher(None, seq35, native, autojunk=False)
    m2n = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            m2n[a + k] = b + k
    if len(set(m2n.values())) >= 0.9 * len(native) and len(m2n) == len(set(m2n.values())):
        return m2n, "difflib"
    return None, None


def meta_one(args):
    stem, native, func_ligs, split = args
    from scipy.spatial import cKDTree
    pdb_id, ch = stem.rsplit("_", 1)
    residues, seq35, _ = m35.extract_rna(pdb_id, ch)
    if seq35 is None or len(residues) != len(seq35):
        return stem, None
    mapping, how = align_35_to_native(seq35, native)
    if mapping is None:
        return stem, {"usable": False, "reason": "seq_mismatch"}
    pts_per_res = [np.array(list(a.values()), dtype=np.float32) if a else np.empty((0, 3))
                   for _, _, _, a in residues]
    ccds = extract_ligands_per_ccd(pdb_id, pts_per_res, func_ligs)
    ccds = {k: v for k, v in ccds.items() if len(v["coords"]) > 0}
    if not ccds:
        return stem, {"usable": False, "reason": "no_pocket_ligand"}
    lig_all = np.array([c for v in ccds.values() for c in v["coords"]], dtype=np.float32)
    tree_lig = cKDTree(lig_all)
    iface35 = np.zeros(len(seq35), dtype=bool)
    for i, pts in enumerate(pts_per_res):
        if len(pts):
            d, _ = tree_lig.query(pts, k=1)
            if d.min() < CONTACT:
                iface35[i] = True
    interface = np.zeros(len(native), dtype=bool)
    for i35, inat in mapping.items():
        if iface35[i35]:
            interface[inat] = True
    return stem, {
        "usable": True, "align": how, "split": split,
        "interface": interface.tolist(), "n_iface": int(interface.sum()),
        "ccds": {k: {"elems": v["elems"], "coords": v["coords"]} for k, v in ccds.items()},
    }


def phase_meta(only_av2=False):
    from concurrent.futures import ProcessPoolExecutor
    func_ligs = load_func_ligs()
    sp2 = json.load(open(os.path.join(DATA_DIR, "splits_v2", "splits_v2.json")))
    feats, args_list = {}, []
    # B 层 test 477：paco_s1_v2 缓存（P0-0 同源）
    if not only_av2:
        from PACoAptamer.data_utils import load_s1
        data = load_s1(set(sp2["test"]), cache=os.path.join(DATA_DIR, "paco_s1_v2.pt"))
        for stem in sp2["test"]:   # 缓存命中时 load_s1 不过滤 subset——必须按 split 键遍历
            if stem not in data:
                continue
            d = data[stem]
            feats[stem] = {k: d[k] for k in ("coords9", "ss_pairs", "seq")}
            args_list.append((stem, d["seq"], func_ligs, "test"))
    # A 层 indep∪zs：m68.build_feature（grnade_inputs_v2 → coords9 + RNAfold SS + native）
    av2 = sorted(set(sp2["independent_test"]) | set(sp2["zero_shot_targets"]))
    for stem in av2:
        try:
            d, why = m68.build_feature(stem)
        except Exception:
            d, why = None, "exception"
        if d is None:
            print(f"  [av2 跳过 {stem}: {why}]", flush=True)
            continue
        feats[stem] = {"coords9": d["coords9"], "ss_pairs": d["ss_pairs"], "seq": d["seq"]}
        args_list.append((stem, d["seq"], func_ligs, "av2"))
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (stem, rec) in enumerate(ex.map(meta_one, args_list, chunksize=4)):
            out[stem] = rec
            if (i + 1) % 100 == 0:
                print(f"meta {i+1}/{len(args_list)} {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(SUR, "75_meta.json"), "w"))
    torch.save(feats, os.path.join(SUR, "75_feats.pt"))
    usable = [k for k, v in out.items() if v and v.get("usable")]
    iface = [k for k in usable if 0 < out[k]["n_iface"] <= N_IFACE_MAX]
    print(f"meta 完成：候选 {len(out)} | 可用 {len(usable)} | 界面链 {len(iface)}")


# ---------------------------------------------------------------- families
def nw_pairs(a, b, match=2, mis=-1, gap=-2):
    """全局 NW 对齐（match=2/mismatch=-1/gap=-2）→ 恒等匹配位对 [(ia, ib)]。
    difflib 替代品：SequenceMatcher 贪心块匹配不对称且对应噪声大（1Y26~6UBU
    两个方向叠合 RMSD 3.8 vs 6.4Å），NW 全局最优对齐是同源全长序列的正确模型。"""
    la, lb = len(a), len(b)
    D = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        D[i][0] = gap * i
    for j in range(1, lb + 1):
        D[0][j] = gap * j
    for i in range(1, la + 1):
        ai = a[i - 1]
        Di, Dm = D[i], D[i - 1]
        for j in range(1, lb + 1):
            s = match if ai == b[j - 1] else mis
            v = Dm[j - 1] + s
            u = Dm[j] + gap
            if u > v:
                v = u
            u = Di[j - 1] + gap
            if u > v:
                v = u
            Di[j] = v
    pairs = []
    i, j = la, lb
    while i > 0 and j > 0:
        s = match if a[i - 1] == b[j - 1] else mis
        if D[i][j] == D[i - 1][j - 1] + s:
            pairs.append((i - 1, j - 1))   # 全部对齐列（含错配位）
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def pair_align(a, b):
    """NW 对齐 → (ident_ratio, coverage, ident_pairs)；ident_pairs = 恒等匹配位
    （叠合锚点口径：家族聚类 / Kabsch 移植均用恒等位，结果与旧版一致）"""
    full = nw_pairs(a, b)
    ident_pairs = [(i, j) for i, j in full if a[i] == b[j]]
    cov = len(ident_pairs) / min(len(a), len(b)) if min(len(a), len(b)) else 0.0
    ident = len(ident_pairs) / max(len(a), len(b)) if max(len(a), len(b)) else 0.0
    return ident, cov, ident_pairs


def struct_rmsd(pairs, c1p_a, c1p_b):
    """匹配位 C1' Kabsch 叠合 RMSD（NaN 位剔除）；返回 (rmsd, n_used)"""
    P = np.array([c1p_a[x] for x, _ in pairs])
    Q = np.array([c1p_b[y] for _, y in pairs])
    ok = ~(np.isnan(P).any(1) | np.isnan(Q).any(1))
    if ok.sum() < 20:
        return None, int(ok.sum())
    _, _, _, _, rmsd = kabsch_transform(P[ok], Q[ok])
    return float(rmsd), int(ok.sum())


def phase_families():
    """结构感知家族聚类：序列 ratio ≥0.45 预筛 → 匹配位 C1' Kabsch 叠合。
    边判据（实测校准）：cov ≥0.65 且（RMSD ≤2.5 或 [RMSD ≤4.5 且 n_used ≥30]）。
    同家族实测 RMSD ≤3.9（add↔xpt，difflib 对应噪声）；异家族 ≥7.9，区分度充足。
    纯序列准则不可用：小 RNA 组成相似（茶碱 vs 8HZ 荧光适配体 ratio 0.70）。
    与 splits_v2 的 TM 结构簇方法论一致；移植阶段（align）仍逐对再验证。"""
    meta = json.load(open(os.path.join(SUR, "75_meta.json")))
    stems = sorted(k for k, v in meta.items()
                   if v and v.get("usable") and 0 < v["n_iface"] <= N_IFACE_MAX)
    feats = torch.load(os.path.join(SUR, "75_feats.pt"), weights_only=False)
    seqs = {s: feats[s]["seq"] for s in stems}
    c1p = {s: np.asarray(feats[s]["coords9"], dtype=np.float32)[:, C1P_IDX, :]
           for s in stems}
    ccd_set = {s: "+".join(sorted(meta[s]["ccds"])) for s in stems}
    parent = {s: s for s in stems}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges, cand = [], 0
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            r, cov, pairs = pair_align(seqs[a], seqs[b])
            if r < 0.45 or cov < COVER_MIN:
                continue
            cand += 1
            rmsd, n_used = struct_rmsd(pairs, c1p[a], c1p[b])
            if rmsd is None or n_used < 0.7 * len(pairs):
                continue
            if rmsd <= 2.5 or (rmsd <= 4.5 and n_used >= 30):
                edges.append([a, b, round(rmsd, 3)])
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    comps = {}
    for s in stems:
        comps.setdefault(find(s), []).append(s)
    fams = {}
    for root, members in comps.items():
        if len(members) < 3:
            continue
        csets = {ccd_set[s] for s in members}
        if len(csets) < 2:
            continue
        seed_pdb = sorted(members)[0].rsplit("_", 1)[0]
        name = FAMILY_NAMES.get(seed_pdb, f"fam_{seed_pdb}")
        fams[name] = {"chains": sorted(members),
                      "ccd_sets": sorted(csets),
                      "splits": sorted({meta[s]["split"] for s in members})}
    json.dump(fams, open(os.path.join(SUR, "75_families.json"), "w"), indent=1)
    print(f"families 完成：{len(stems)} 界面链 | 预筛对 {cand} | 结构边 {len(edges)} "
          f"→ {len(fams)} 个多配体家族")
    for name, f in fams.items():
        print(f"  {name}: {len(f['chains'])} 链 {f['splits']} | CCD 组合 {f['ccd_sets']}")
    json.dump(edges, open(os.path.join(SUR, "75_edges.json"), "w"))


# ---------------------------------------------------------------- align/grid
def kabsch_transform(P, Q):
    """刚体变换 P→Q（行对应点）；返回变换后的 P 与 RMSD"""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    Pt = (R @ (P - Pc).T).T + Qc
    rmsd = float(np.sqrt(((Pt - Q) ** 2).sum(1).mean()))
    return Pt, R, Pc, Qc, rmsd


def pooled_ligand(meta_stem):
    ccds = meta_stem["ccds"]
    coords = np.array([c for v in ccds.values() for c in v["coords"]], dtype=np.float32)
    elems = [e for v in ccds.values() for e in v["elems"]]
    return coords, elems


def phase_align():
    from scipy.spatial import cKDTree
    meta = json.load(open(os.path.join(SUR, "75_meta.json")))
    fams = json.load(open(os.path.join(SUR, "75_families.json")))
    edges = {(a, b) for a, b, _ in json.load(open(os.path.join(SUR, "75_edges.json")))}
    edges |= {(b, a) for a, b in edges}
    feats = torch.load(os.path.join(SUR, "75_feats.pt"), weights_only=False)
    grid, pair_stats = {}, []
    for fname, fam in fams.items():
        members = fam["chains"]
        ccd_set = {s: "+".join(sorted(meta[s]["ccds"])) for s in members}
        # c1p（nan_to_num 前保留 mask：NaN 位置不参与叠合）
        c1p_raw = {s: np.asarray(feats[s]["coords9"], dtype=np.float32)[:, C1P_IDX, :]
                   for s in members}
        seqs = {s: feats[s]["seq"] for s in members}
        # 候选移植 (i 接受, j 供体)
        offers = {s: [] for s in members}
        for i in members:
            for j in members:
                if i == j or ccd_set[i] == ccd_set[j]:
                    continue
                if i.rsplit("_", 1)[0] == j.rsplit("_", 1)[0]:
                    continue    # 同一条目内不同链化学组合同 → 无意义；不同条目才算移植
                if (j, i) not in edges:
                    continue    # 家族结构边（RMSD ≤3Å）为准入前提
                r, cov, pairs = pair_align(seqs[j], seqs[i])
                if cov < COVER_MIN:
                    continue
                pj = np.array([c1p_raw[j][a] for a, b in pairs])
                pi = np.array([c1p_raw[i][b] for a, b in pairs])
                ok = ~(np.isnan(pj).any(1) | np.isnan(pi).any(1))
                if ok.sum() < 0.7 * len(pairs) or ok.sum() < 20:
                    continue
                _, R, Pc, Qc, rmsd = kabsch_transform(pj[ok], pi[ok])
                if rmsd > RMSD_MAX:
                    continue
                lc, le = pooled_ligand(meta[j])
                lc_t = (R @ (lc - Pc).T).T + Qc
                own_lc, _ = pooled_ligand(meta[i])
                off = float(np.linalg.norm(lc_t.mean(0) - own_lc.mean(0)))
                if off > CENTROID_MAX:
                    continue
                # 冲突检查：移植原子与链任何原子 <1.3 Å 视为穿膜
                chain_pts = np.asarray(feats[i]["coords9"], dtype=np.float32).reshape(-1, 3)
                chain_pts = chain_pts[~np.isnan(chain_pts).any(1)]
                if len(chain_pts):
                    dmin = cKDTree(chain_pts).query(lc_t, k=1)[0].min()
                    if dmin < 1.3:
                        continue
                offers[i].append({"donor": j, "ccd_set": ccd_set[j],
                                  "rmsd": round(rmsd, 3), "centroid_off": round(off, 3),
                                  "lig_coords": lc_t.astype(np.float32).tolist(),
                                  "lig_elems": le})
                pair_stats.append({"family": fname, "accept": i, "donor": j,
                                   "rmsd": round(rmsd, 3), "centroid_off": round(off, 3)})
        for i in members:
            conds = {"true": {"lig_elems": [e for v in meta[i]["ccds"].values() for e in v["elems"]],
                              "lig_coords": [c for v in meta[i]["ccds"].values() for c in v["coords"]]}}
            # geo：同 CCD 组合、异条目、Kabsch 叠合、最佳质心距
            geo_pool = {s: pooled_ligand(meta[s]) for s in members
                        if s != i and ccd_set[s] == ccd_set[i]
                        and s.rsplit("_", 1)[0] != i.rsplit("_", 1)[0]}
            if geo_pool:
                best, best_off = None, 1e9
                own_lc, _ = pooled_ligand(meta[i])
                for s, (lc, le) in geo_pool.items():
                    pj, pi = c1p_raw[s], c1p_raw[i]
                    r, cov, pairs = pair_align(seqs[s], seqs[i])
                    if cov < COVER_MIN:
                        continue
                    P = np.array([pj[a] for a, b in pairs]); Q = np.array([pi[b] for a, b in pairs])
                    ok = ~(np.isnan(P).any(1) | np.isnan(Q).any(1))
                    if ok.sum() < 20:
                        continue
                    _, R, Pc, Qc, rmsd = kabsch_transform(P[ok], Q[ok])
                    lc_t = (R @ (lc - Pc).T).T + Qc
                    off = float(np.linalg.norm(lc_t.mean(0) - own_lc.mean(0)))
                    if off < best_off and rmsd <= RMSD_MAX and off <= CENTROID_MAX:
                        best, best_off = (lc_t.astype(np.float32).tolist(), le, rmsd, off), off
                if best is not None:
                    conds["geo"] = {"lig_coords": best[0], "lig_elems": best[1],
                                    "from": "aligned_same_ccd",
                                    "rmsd": round(best[2], 3), "centroid_off": round(best[3], 3)}
            # swap：异 CCD 组合，每组和取质心距最佳者；扰动谱选择——最优拟合 +
            # 最大扰动 + 中间分位（保证化学跨度：纯最优排序会偏向化学最接近的
            # 类似物，如 ADE 链只换到嘌呤变体而漏掉 GUN/HPA 跨类对比）
            by_set = {}
            for o in offers[i]:
                if o["ccd_set"] == ccd_set[i]:
                    continue
                cur = by_set.get(o["ccd_set"])
                if cur is None or o["centroid_off"] < cur["centroid_off"]:
                    by_set[o["ccd_set"]] = o
            ranked = [o for _, o in sorted(by_set.items(),
                                           key=lambda x: x[1]["centroid_off"])]
            if len(ranked) <= MAX_SWAP:
                picked = ranked
            else:
                idx = sorted({0, len(ranked) - 1,
                              int(round((len(ranked) - 1) / 3)),
                              int(round(2 * (len(ranked) - 1) / 3))})
                picked = [ranked[j] for j in idx][:MAX_SWAP]
            for o in picked:
                conds[f"swap_{o['donor']}"] = {"lig_coords": o["lig_coords"],
                                               "lig_elems": o["lig_elems"],
                                               "ccd_set": o["ccd_set"], "rmsd": o["rmsd"],
                                               "centroid_off": o["centroid_off"]}
            grid[i] = {"family": fname, "own_ccd_set": ccd_set[i], "conds": conds}
    json.dump(grid, open(os.path.join(SUR, "75_grid.json"), "w"))
    n_cond = sum(len(g["conds"]) for g in grid.values())
    n_swap = sum(1 for g in grid.values() for c in g["conds"] if c.startswith("swap"))
    n_geo = sum(1 for g in grid.values() for c in g["conds"] if c == "geo")
    print(f"align 完成：{len(grid)} 链 | 条件总数 {n_cond}（swap {n_swap}，geo {n_geo}）")
    offs = [p["centroid_off"] for p in pair_stats]
    rms = [p["rmsd"] for p in pair_stats]
    if offs:
        print(f"  移植质量：centroid_off 中位 {np.median(offs):.2f}Å（p90 {np.percentile(offs,90):.2f}）| "
              f"C1' RMSD 中位 {np.median(rms):.2f}Å")


# ---------------------------------------------------------------- 模型
def load_model(ckpt_path, device=0):
    return m74.load_model(ckpt_path, device)


def lig_feats_of(cond, c1p):
    if cond is None:
        return None
    lc = np.array(cond["lig_coords"], dtype=np.float32)
    if len(lc) == 0:
        return None
    return lig_features(lc, cond["lig_elems"], c1p)


def c1p_of(d):
    c1p = np.asarray(d['coords9'], dtype=np.float32)[:, C1P_IDX, :]
    return np.nan_to_num(c1p, nan=1e-5)   # 74 号修复语义


@torch.no_grad()
def score_stem(models, alphabet, d, lig_np, device):
    """teacher-forced 逐位 logP(native)。多模型 = softmax 概率平均（与集成采样同语义）"""
    L = len(d['seq'])
    coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
    ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
    conf = torch.ones(1, L, device=device)
    if lig_np is not None and len(lig_np) > 0:
        lig = torch.tensor(lig_np, dtype=torch.float32, device=device).unsqueeze(0)
        ligm = torch.ones(1, lig_np.shape[0], dtype=torch.bool, device=device)
    else:
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
    pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
    tokens = [alphabet.get_idx("<cath>")] + [alphabet.get_idx(c) for c in d['seq']]
    prev = torch.tensor([tokens[:-1]], device=device)
    tgt = torch.tensor([tokens[1:]], device=device)
    probs = None
    for m in models:
        logits = m(coords, ss, lig, ligm, pmask, conf, prev)   # (1, V, L)
        p = F.softmax(logits, dim=1)
        probs = p if probs is None else probs + p
    probs = probs / len(models)
    logp = torch.log(probs[0] + 1e-12)     # (V, L)
    lp = logp.gather(0, tgt[0].unsqueeze(0)).squeeze(0)   # (L,)
    return lp.float().cpu().numpy()


@torch.no_grad()
def sample_ens(models, alphabet, d, lig_np, device, temperature, n_samples):
    """39 号集成采样 + 配体上下文"""
    L = len(d['seq'])
    coords = torch.tensor(d['coords9'], dtype=torch.float32, device=device).unsqueeze(0)
    ss = torch.tensor(d['ss_pairs'], dtype=torch.bool, device=device).unsqueeze(0)
    conf = torch.ones(1, L, device=device)
    if lig_np is not None and len(lig_np) > 0:
        lig = torch.tensor(lig_np, dtype=torch.float32, device=device).unsqueeze(0)
        ligm = torch.ones(1, lig_np.shape[0], dtype=torch.bool, device=device)
    else:
        lig = torch.zeros(1, 0, 21, device=device)
        ligm = torch.zeros(1, 0, dtype=torch.bool, device=device)
    n, K = int(n_samples), len(models)
    enc_list = []
    for m in models:
        pmask = torch.zeros(1, L, dtype=torch.bool, device=device)
        eo = m.forward_embedding(coords, ss, lig, ligm, pmask, conf)
        eo = {"encoder_out": [eo["encoder_out"][0].repeat(1, n, 1)],
              "encoder_padding_mask": [eo["encoder_padding_mask"][0].repeat(n, 1)],
              "encoder_embedding": [], "encoder_states": []}
        enc_list.append(eo)
    bos = alphabet.get_idx("<cath>")
    pred = torch.zeros(n, L + 1, dtype=torch.long, device=device)
    pred[:, 0] = bos
    states = [{} for _ in range(K)]
    for i in range(L):
        prev = pred[:, :i + 1]
        probs_sum = None
        for m, eo, st in zip(models, enc_list, states):
            logits, _ = m.decoder(prev, encoder_out=eo, incremental_state=st)
            p = F.softmax(logits[:, :, -1] / temperature, dim=-1)
            probs_sum = p if probs_sum is None else probs_sum + p
        sample = torch.multinomial(probs_sum / K, 1).squeeze(-1)
        pred[:, i + 1] = sample
    return [''.join(alphabet.get_tok(t) for t in row.tolist()) for row in pred[:, 1:]]


# ---------------------------------------------------------------- score/sample
def phase_score(model_key, families=None):
    device = 0
    pairs_ = [load_model(p, device) for p in MODEL_SETS[model_key]]
    models = [m for m, _ in pairs_]
    alphabet = pairs_[0][1]
    grid = json.load(open(os.path.join(SUR, "75_grid.json")))
    feats = torch.load(os.path.join(SUR, "75_feats.pt"), weights_only=False)
    meta = json.load(open(os.path.join(SUR, "75_meta.json")))
    fams = json.load(open(os.path.join(SUR, "75_families.json")))
    fam_of = {s: f for f, v in fams.items() for s in v["chains"]}
    out_dir = os.path.join(SUR, "75_score")
    os.makedirs(out_dir, exist_ok=True)
    stems = [s for s in sorted(grid)
             if families is None or fam_of.get(s) in families]
    t0 = time.time()
    n_done = 0
    for stem in stems:
        out_p = os.path.join(out_dir, f"{model_key}__{stem}.json")
        if os.path.exists(out_p):
            continue
        d = feats[stem]
        if len(d['seq']) > MAX_LEN:
            continue
        c1p = c1p_of(d)
        rec = {}
        conds = dict(grid[stem]["conds"])
        conds["none"] = None
        try:
            for cname, cond in conds.items():
                lf = lig_feats_of(cond, c1p)
                rec[cname] = score_stem(models, alphabet, d, lf, device).tolist()
        except RuntimeError as e:
            print(f"  [score 跳过 {stem}: {e}]", flush=True)
            continue
        json.dump({"family": grid[stem]["family"], "logp": rec,
                   "interface": meta[stem]["interface"]}, open(out_p, "w"))
        n_done += 1
        if n_done % 20 == 0:
            print(f"[score:{model_key}] {n_done}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    print(f"[score:{model_key}] 完成 {n_done}/{len(stems)} 链")


def phase_sample(model_key, n_map, families=None, seed=42):
    device = 0
    torch.manual_seed(seed)
    pairs_ = [load_model(p, device) for p in MODEL_SETS[model_key]]
    models = [m for m, _ in pairs_]
    alphabet = pairs_[0][1]
    grid = json.load(open(os.path.join(SUR, "75_grid.json")))
    feats = torch.load(os.path.join(SUR, "75_feats.pt"), weights_only=False)
    fams = json.load(open(os.path.join(SUR, "75_families.json")))
    fam_of = {s: f for f, v in fams.items() for s in v["chains"]}
    out_dir = os.path.join(SUR, "75_pools")
    os.makedirs(out_dir, exist_ok=True)
    stems = [s for s in sorted(grid)
             if families is None or fam_of.get(s) in families]
    t0 = time.time()
    for si, stem in enumerate(stems):
        out_p = os.path.join(out_dir, f"{model_key}__{stem}.json")
        if os.path.exists(out_p):
            continue
        fam = fam_of[stem]
        n_per = n_map.get(fam, min(n_map.values()))
        d = feats[stem]
        if len(d['seq']) > MAX_LEN:
            continue
        c1p = c1p_of(d)
        conds = dict(grid[stem]["conds"])
        conds["none"] = None
        pools, failed = {}, False
        try:
            for cname, cond in conds.items():
                lf = lig_feats_of(cond, c1p)
                pool = []
                for t in TEMPS:
                    seqs = sample_ens(models, alphabet, d, lf, device, t, n_per)
                    pool.extend(seqs)
                pools[cname] = pool
        except RuntimeError as e:
            print(f"  [sample 跳过 {stem}: {e}]", flush=True)
            failed = True
        if failed:
            continue
        json.dump({"family": fam, "n_per_temp": n_per, "pools": pools},
                  open(out_p, "w"))
        if (si + 1) % 5 == 0:
            print(f"[sample:{model_key}] {si+1}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    print(f"[sample:{model_key}] 完成")


# ---------------------------------------------------------------- report
def boot_ci(x, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def spec_pairs(stem, donor, feats, meta):
    """特异性位点：interface 内、native_acceptor 与 native_donor（对齐后）不同的位置。
    返回 [(pos_in_acceptor, base_acceptor, base_donor), ...]"""
    a_seq = feats[stem]["seq"]
    d_seq = feats[donor]["seq"]
    pairs = nw_pairs(d_seq, a_seq)    # 全部对齐列（含错配）——特异性位点恰是错配位
    amap = {b: a for a, b in pairs}   # pairs=(d_idx, a_idx) → a_idx → d_idx
    iface = np.array(meta[stem]["interface"], dtype=bool)
    out = []
    for ia in range(len(a_seq)):
        idn = amap.get(ia)
        if idn is None or not iface[ia]:
            continue
        if a_seq[ia] != d_seq[idn] and a_seq[ia] != "N" and d_seq[idn] != "N":
            out.append((ia, a_seq[ia], d_seq[idn]))
    return out


def phase_report():
    from scipy.stats import wilcoxon
    grid = json.load(open(os.path.join(SUR, "75_grid.json")))
    feats = torch.load(os.path.join(SUR, "75_feats.pt"), weights_only=False)
    meta = json.load(open(os.path.join(SUR, "75_meta.json")))
    fams = json.load(open(os.path.join(SUR, "75_families.json")))
    fam_of = {s: f for f, v in fams.items() for s in v["chains"]}
    report = {"families": fams, "E1_likelihood": {}, "E2_recovery": {},
              "E3_following": {}, "E3_purine_curve": {}, "E4_placebo": {}}

    # ---------- E1 teacher-forced 似然
    sdir = os.path.join(SUR, "75_score")
    for model_key in ("ens2noiface", "noligand", "seed43noiface"):
        rows = []
        per_fam = {}
        for fn in sorted(os.listdir(sdir)):
            if not fn.startswith(model_key + "__"):
                continue
            stem = fn[len(model_key) + 2:-5]
            r = json.load(open(os.path.join(sdir, fn)))
            iface = [i for i, x in enumerate(r["interface"]) if x]
            native = feats[stem]["seq"]
            if not iface:
                continue
            row = {"stem": stem, "family": r["family"]}
            for cname, lp in r["logp"].items():
                lp = np.array(lp)
                row[f"lp_iface_{cname}"] = float(lp[iface].mean())
                row[f"lp_all_{cname}"] = float(lp.mean())
            rows.append(row)
            per_fam.setdefault(r["family"], []).append(row)
        if not rows:
            continue
        agg = {}
        cond_names = sorted({c[len("lp_iface_"):] for r in rows for c in r
                             if c.startswith("lp_iface_") and c != "lp_iface_true"})
        conds = ["lp_iface_" + cn for cn in cond_names]
        for c in conds:
            cn = c[len("lp_iface_"):]
            dif = [r["lp_iface_true"] - r[c] for r in rows if c in r]
            try:
                p = float(wilcoxon(dif, alternative="greater").pvalue) if len(dif) >= 10 else None
            except Exception:
                p = None
            agg[cn] = {"n": len(dif), "d_lp_iface": float(np.mean(dif)),
                       "ci95": boot_ci(dif), "wilcoxon_p_one_sided": p}
        # spec 位点似然：swap 条件下，特异性位点（native 与 donor 不同且在界面内）
        # 的 ΔlogP(true − swap) —— 模型是否在被告知正确配体时更倾向 native 碱基
        spec_dlp = []
        for fn2 in sorted(os.listdir(sdir)):
            if not fn2.startswith(model_key + "__"):
                continue
            stem2 = fn2[len(model_key) + 2:-5]
            r2 = json.load(open(os.path.join(sdir, fn2)))
            if stem2 not in grid:
                continue
            for cname in r2["logp"]:
                if not cname.startswith("swap_"):
                    continue
                donor = cname[5:]
                sp = spec_pairs(stem2, donor, feats, meta)
                if not sp:
                    continue
                lp_t = np.array(r2["logp"]["true"])
                lp_s = np.array(r2["logp"][cname])
                pos = [p for p, _, _ in sp]
                spec_dlp.append(float(np.mean(lp_t[pos] - lp_s[pos])))
        if len(spec_dlp) >= 10:
            try:
                p = float(wilcoxon(spec_dlp, alternative="greater").pvalue)
            except Exception:
                p = None
            agg["spec_dlp"] = {"n": len(spec_dlp), "mean": float(np.mean(spec_dlp)),
                               "ci95": boot_ci(spec_dlp), "wilcoxon_p_one_sided": p}
        agg["_per_family"] = {f: {"n": len(v)} for f, v in per_fam.items()}
        report["E1_likelihood"][model_key] = agg

    # ---------- E2/E3 采样
    pdir = os.path.join(SUR, "75_pools")
    for model_key in ("ens2noiface", "noligand"):
        e2, e3, curve = {}, {}, []
        for fn in sorted(os.listdir(pdir)):
            if not fn.startswith(model_key + "__"):
                continue
            stem = fn[len(model_key) + 2:-5]
            rec = json.load(open(os.path.join(pdir, fn)))
            native = feats[stem]["seq"]
            iface = [i for i, x in enumerate(meta[stem]["interface"]) if x]
            npt = rec["n_per_temp"]
            npos = [i for i, c in enumerate(native) if c != "N"]
            ipos = sorted(set(iface) & set(npos))
            fam = rec["family"]
            conds = rec["pools"]
            # 恢复率（池内 best / t01）
            mrow = {}
            for cname, pool in conds.items():
                ifs, ovs = [], []
                for s in pool:
                    if len(s) != len(native):
                        continue
                    ovs.append(sum(1 for i in npos if s[i] == native[i]) / len(npos))
                    if ipos:
                        ifs.append(sum(1 for i in ipos if s[i] == native[i]) / len(ipos))
                mrow[cname] = {"iface_best": max(ifs) if ifs else None,
                               "ov_best": max(ovs) if ovs else None,
                               "iface_t01": float(np.mean(ifs[:npt])) if ifs[:npt] else None,
                               "ov_t01": float(np.mean(ovs[:npt])) if ovs[:npt] else None}
            e2[stem] = {"family": fam, **mrow}
            # E3 条件跟随：spec 位点 base 频率随条件变化
            for cname, pool in conds.items():
                if not cname.startswith("swap_") and cname != "true":
                    continue
                donor = cname[5:] if cname.startswith("swap_") else None
                sp = spec_pairs(stem, donor, feats, meta) if donor else \
                    [(i, native[i], None) for i in ipos]
                if not sp:
                    continue
                keep = [s for s in pool if len(s) == len(native)]
                n0 = npt   # t=0.1 段
                for seg_name, seg in (("t01", keep[:n0]), ("all", keep)):
                    cnt = {p: {"nat": 0, "don": 0, "tot": 0} for p, _, _ in sp}
                    for s in seg:
                        for p, bn, bd in sp:
                            cnt[p]["tot"] += 1
                            if s[p] == bn:
                                cnt[p]["nat"] += 1
                            if bd is not None and s[p] == bd:
                                cnt[p]["don"] += 1
                    for p, bn, bd in sp:
                        t = max(cnt[p]["tot"], 1)
                        curve.append({"stem": stem, "family": fam, "cond": cname,
                                      "donor": donor, "pos": p, "seg": seg_name,
                                      "p_native": cnt[p]["nat"] / t,
                                      "p_donor": (cnt[p]["don"] / t) if bd is not None else None,
                                      "n_obs": cnt[p]["tot"],
                                      "base_native": bn, "base_donor": bd})
        # E2 汇总（配对对比）
        e2_agg = {}
        fams_present = {v["family"] for v in e2.values()}
        for fam in fams_present | {"ALL"}:
            sel = {s: v for s, v in e2.items() if fam == "ALL" or v["family"] == fam}
            for metric in ("iface_best", "iface_t01", "ov_best"):
                for ref in ("none", "geo"):
                    pairs = [(v["true"][metric], v[ref][metric]) for v in sel.values()
                             if "true" in v and ref in v and v.get(ref) and
                             v["true"][metric] is not None and v[ref][metric] is not None]
                    if len(pairs) >= 5:
                        d = [a - b for a, b in pairs]
                        try:
                            p = float(wilcoxon(d, alternative="greater").pvalue)
                        except Exception:
                            p = None
                        e2_agg[f"{fam}|true-{ref}|{metric}"] = {
                            "n": len(d), "delta": float(np.mean(d)), "ci95": boot_ci(d),
                            "wilcoxon_p_one_sided": p}
                # swap 汇总：每链取其所有 swap 条件的均值（链为配对单位）
                pairs = []
                for v in sel.values():
                    sw = [v[c][metric] for c in v if c.startswith("swap_")
                          and v[c][metric] is not None]
                    if "true" in v and v["true"][metric] is not None and sw:
                        pairs.append((v["true"][metric], float(np.mean(sw))))
                if len(pairs) >= 5:
                    d = [a - b for a, b in pairs]
                    try:
                        p = float(wilcoxon(d, alternative="greater").pvalue)
                    except Exception:
                        p = None
                    e2_agg[f"{fam}|true-swap|{metric}"] = {
                        "n": len(d), "delta": float(np.mean(d)), "ci95": boot_ci(d),
                        "wilcoxon_p_one_sided": p}
        report["E2_recovery"][model_key] = e2_agg
        # E3 汇总：swap 条件下 p_native 的下降 / p_donor 的上升（true vs swap 配对）
        fol = {}
        rows_by = {}
        for c in curve:
            rows_by.setdefault((c["stem"], c["pos"], c["seg"]), {})[c["cond"]] = c
        dn, dd = [], []
        for (stem, pos, seg), cd in rows_by.items():
            for cname, row in cd.items():
                if not cname.startswith("swap_"):
                    continue
                if "true" in cd:
                    dn.append(row["p_native"] - cd["true"]["p_native"])
                    if row["p_donor"] is not None and cd["true"]["p_donor"] is not None:
                        dd.append(row["p_donor"] - cd["true"]["p_donor"])
        if len(dn) >= 10:
            try:
                p = float(wilcoxon(dn, alternative="less").pvalue)
            except Exception:
                p = None
            fol["p_native_drop"] = {"n": len(dn), "mean": float(np.mean(dn)),
                                    "ci95": boot_ci(dn), "wilcoxon_p_less": p}
        if len(dd) >= 10:
            try:
                p = float(wilcoxon(dd, alternative="greater").pvalue)
            except Exception:
                p = None
            fol["p_donor_rise"] = {"n": len(dd), "mean": float(np.mean(dd)),
                                   "ci95": boot_ci(dd), "wilcoxon_p_greater": p}
        report["E3_following"][model_key] = fol
        report.setdefault("E3_rows", {})[model_key] = curve
        # purine 曲线（供 Fig4h）：spec 位点 U/C 频率按配体化学型
        if model_key == "ens2noiface":
            pur = [c for c in curve if fam_of.get(c["stem"]) == "purine_riboswitch"
                   and c["seg"] == "t01"]
            chem = {}
            for c in pur:
                cs = grid[c["stem"]]["conds"].get(c["cond"], {}).get("ccd_set") \
                    if c["cond"] != "true" else grid[c["stem"]]["own_ccd_set"]
                if cs is None and c["cond"] == "geo":
                    cs = grid[c["stem"]]["own_ccd_set"]
                chem.setdefault(cs, {"p_native": [], "p_donor": []})
                chem[cs]["p_native"].append(c["p_native"])
                if c["p_donor"] is not None:
                    chem[cs]["p_donor"].append(c["p_donor"])
            report["E3_purine_curve"] = {
                cs: {"n": len(v["p_native"]),
                     "p_native_mean": float(np.mean(v["p_native"])),
                     "p_donor_mean": float(np.mean(v["p_donor"])) if v["p_donor"] else None}
                for cs, v in chem.items()}

    json.dump(report, open(os.path.join(SUR, "75_p01_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "E3_rows"}, indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True,
                    choices=['meta', 'families', 'align', 'score', 'sample', 'report'])
    ap.add_argument('--model', type=str, default="ens2noiface")
    ap.add_argument('--families', type=str, default=None, help="逗号分隔家族过滤")
    ap.add_argument('--n_purine', type=int, default=100)
    ap.add_argument('--n_other', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    fams = set(args.families.split(",")) if args.families else None
    if args.phase == 'meta':
        phase_meta()
    elif args.phase == 'families':
        phase_families()
    elif args.phase == 'align':
        phase_align()
    elif args.phase == 'score':
        phase_score(args.model, fams)
    elif args.phase == 'sample':
        fams_all = json.load(open(os.path.join(SUR, "75_families.json")))
        n_map = {f: (args.n_purine if f == "purine_riboswitch" else args.n_other)
                 for f in fams_all}
        phase_sample(args.model, n_map, fams, args.seed)
    elif args.phase == 'report':
        phase_report()


if __name__ == "__main__":
    main()
