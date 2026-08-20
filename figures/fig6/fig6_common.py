# -*- coding: utf-8 -*-
"""Fig 6 公共模块:统一画布几何 + 数据准备。

所有子图脚本(fig6_a..e)与组合脚本(fig6_assemble)共用:
- make_canvas():同一张 183×132 mm 画布、同一 GridSpec(2,16),
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据准备(配体=O2I/N618 荧光染料,非 5′-GDP 帽子):
  iface      链 A 中任一原子距任一 O2I 原子 <5 Å 的残基号
  seqs       各方法最优设计(best-of-pool)及对 natA 的恢复率
  d_apt/d_utx/d_all/fam_* 回顾性审计数组;cs 筛选预算曲线
"""
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, ROOT, SUR, apply_style, load  # noqa: E402

# ---------- 画布几何(所有面板唯一事实来源) ----------
# 2026-08-20 起:画布加高为 183×245 mm,顶部 ~104 mm 为 12 案画廊(panel a,
# fig6_gallery.py 按英寸坐标自摆 axes),下方沿用原 2×16 布局(panel b-e)。
FIG_W_IN = 183 * MM
FIG_H_IN = 245 * MM
GALLERY_H_IN = 104 * MM          # 画廊块高度(含图例),底部 gs 从其下方开始
GRID_KW = dict(hspace=0.55, wspace=1.05, height_ratios=[1.15, 1.0])

# PyMOL 渲染图(由 fig6_pymol.py 生成,画幅比例与槽位一致)
PYMOL_OVERVIEW = os.path.join(ROOT, "figures", "out", "fig6_pymol_overview.png")
PYMOL_POCKET = os.path.join(ROOT, "figures", "out", "fig6_pymol_pocket.png")


def make_canvas():
    """返回 (fig, gs):183×245mm 画布;gs 为下部 2×16 布局(画廊 a 占顶部)。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(2, 16, figure=fig, **GRID_KW)
    # bottom 需容纳 panel e 悬挂于坐标区下方的 legend(约 0.9 in),
    # 否则 bbox_inches="tight" 会向下扩张画布超出 247 mm 版心高。
    gs.update(top=(FIG_H_IN - GALLERY_H_IN - 0.4 * MM) / FIG_H_IN, bottom=0.095)
    return fig, gs


# ================= 8HZD 结构几何(配体 = O2I / N618) =================
LIG_NAME = "O2I"
CONTACT = 5.0
POCKET = 8.0


def parse_pdb(path):
    chains = defaultdict(lambda: defaultdict(list))  # ch → resseq → [(atom,xyz)]
    hetatm = defaultdict(list)                       # resname → [xyz]
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])])
            except ValueError:
                continue
            if line.startswith("ATOM"):
                chains[line[21]][int(line[22:26])].append(
                    (line[12:16].strip(), xyz))
            else:
                rn = line[17:20].strip()
                if rn not in ("HOH", "MG", "K", "PO4"):
                    hetatm[rn].append(xyz)
    return chains, hetatm


CHAINS, HETATM = parse_pdb(os.path.join(ROOT, "data", "rcsb_pdb", "8HZD.pdb"))
LIG_XYZ = np.array(HETATM[LIG_NAME])
RESIDS = sorted(CHAINS["A"])
RES_XYZ = {rn: np.array([x for _, x in atoms]) for rn, atoms in CHAINS["A"].items()}

IFACE = [rn for rn in RESIDS
         if np.linalg.norm(RES_XYZ[rn][:, None, :] - LIG_XYZ[None, :, :],
                           axis=2).min() < CONTACT]              # <5 Å 残基号
POCKET_RES = sorted(set(
    rn for rn in RESIDS
    if np.linalg.norm(RES_XYZ[rn][:, None, :] - LIG_XYZ[None, :, :],
                      axis=2).min() < POCKET))                   # <8 Å 残基号

# ================= 序列条数据 =================
natA = "GGAAGAUUGUAAACAUGCCGAAAGGCAGACACUUCC"


def best_seq_from_pool(seqs, native):
    best, bv = None, -1
    for s in seqs:
        if len(s) != len(native):
            continue
        v = np.mean([a == b for a, b in zip(s, native)])
        if v > bv:
            best, bv = s, v
    return best, bv


def _load_seqs():
    seqs = {}
    seqs["TACo-Aptamer"] = best_seq_from_pool(
        load("taco_av2_pool_ens2noiface/8HZD_A.json")["pool"], natA)
    rd = []
    for t in ("0.1", "0.5", "1.0"):
        rd += [l.strip() for l in open(
            os.path.join(SUR, f"rhodesign_out_av2_retrained/8HZD_A.{t}.seqs"))
            if l.strip()]
    seqs["RhoDesign"] = best_seq_from_pool(rd, natA)
    gr = []
    for t in ("0.1", "0.5", "1.0"):
        gr += [l.strip() for l in open(
            os.path.join(SUR, f"grnade_out_v2/8HZD_A.{t}.seqs")) if l.strip()]
    seqs["gRNAde"] = best_seq_from_pool(gr, natA)
    nm_ = [l.strip() for l in open(os.path.join(SUR, "nampnn_out/8HZD_A.seqs"))
           if l.strip()]
    seqs["NA-MPNN"] = best_seq_from_pool(nm_, natA)
    al_ = []
    for line in open(os.path.join(SUR, "alignif_out/8HZD_A", "out.fasta")):
        if not line.startswith(">"):
            al_.append(line.strip().replace("-", ""))
    seqs["AlignIF"] = best_seq_from_pool(al_, natA)
    return seqs


SEQS = _load_seqs()

# ================= 回顾性审计数据 =================
def _audit():
    apt_clean = load("79_aptadb_clean.json")["entries"]
    apt_scored = load("79_aptadb_scored.json")["entries"]
    utx_clean = load("88_utexas_clean.json")["entries"]
    utx_scored = load("88_utexas_scored.json")

    d_apt = []
    for e in apt_clean:
        if e["leak"]["cluster_train"] or e["leak"]["exact"] is not None:
            continue
        s = apt_scored[e["key"]]
        d_apt.append(s["logp_self"] - np.mean(s["logp_rand_sameSS"]))
    d_utx = []
    for e in utx_clean:
        if e["leak"]["cluster_train"]:
            continue
        s = utx_scored[e["key"]]
        d_utx.append(s["logp_self"] - np.mean(s["logp_randA"]))

    fam_rho, fam_n = [], []
    byfam = defaultdict(list)
    for e in apt_clean:
        if e["leak"]["cluster_train"] or e["leak"]["exact"] is not None:
            continue
        if e.get("kd_nM") is None or e["kd_nM"] <= 0:
            continue
        byfam[e["target_name"]].append(
            (apt_scored[e["key"]]["logp_self"], -np.log10(e["kd_nM"] / 1e9)))
    for pairs in byfam.values():
        if len(pairs) >= 3:
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            if np.std(x) > 0 and np.std(y) > 0:
                r, _ = stats.spearmanr(x, y)
                if np.isfinite(r):
                    fam_rho.append(r)
                    fam_n.append(len(pairs))
    return np.array(d_apt), np.array(d_utx), np.array(fam_rho), fam_n


D_APT, D_UTX, FAM_RHO, FAM_N = _audit()
D_ALL = np.concatenate([D_APT, D_UTX])

# 筛选预算曲线
CS = load("84_consensus_screen.json")["curves"]

# 案例三链指标
CASE = ["8HZD_A", "8HZD_B", "8HZE_A"]
