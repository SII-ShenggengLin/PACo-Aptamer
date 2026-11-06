# -*- coding: utf-8 -*-
"""Fig 6 公共模块:统一画布几何 + 数据准备。

所有子图脚本(fig6_b..f)与组合脚本(fig6_assemble)共用:
- make_canvas():同一张 183×202mm 画布,返回 (fig, L);L 为各面板 axes 的
  精确 fig-fraction 矩形,全部锚定在画廊(panel a)的 4 列图像网格上,
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色/对齐天然统一。
- 字号(第五轮可读性放大,全图 +1 pt):刻度 8.0 / 轴题 8.5 / 图例 8.0,
  面板内文字 ≥7.0 pt,不加粗、一律黑色;编号字母 10 pt 粗体。
- 数据准备(配体=O2I/N618 荧光染料,非 5′-GDP 帽子):
  iface      链 A 中任一原子距任一 O2I 原子 <5 Å 的残基号
  seqs       各方法最优设计(best sample)及对 natA 的恢复率
  d_apt/d_utx/d_all/fam_* 回顾性审计数组;cs 筛选预算曲线

2026-10-23 版式(第二轮,应作者要求):panel a 画廊格题/注记与结构图的净空
加大(标题 pad 3.5 pt、注记下移 5 pt、行距 +0.12 in);b/c 行加高至 0.95 in,
b 两幅渲染上下各加一行与画廊格一致的格题/注记文字(渲染图内不再嵌字);
c 序列条跨画廊第 3–4 列图像;d ΔlogP 直方图跨第 1–2 列并收窄;e 家族内相关
森林图加宽(图例改为均值虚线旁的文字注记);f 筛选预算曲线加宽。d/e/f 的
图例统一移至左上。编号 d/e/f 仍锚定画廊列边界。
"""
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, ROOT, SUR, apply_style, load  # noqa: E402

# ================= 画布几何(所有面板唯一事实来源) =================
FIG_W_IN = 183 * MM                     # Nature Methods 双栏宽 183 mm
GL, GR = 0.010, 0.990                   # 画廊 4 列网格边界(fig fraction)
COLS = [GL + i * (GR - GL) / 4 for i in range(5)]
CELL_IN = (GR - GL) * FIG_W_IN / 4      # 列宽 1.765 in
GAL_IMG_FRAC = 0.72     # 格内图像宽占列宽比例(第五轮可读性放大:0.62→0.72,
                        # 相邻列图像净空仍 ~0.22 in;编号/子图左缘随 IMG_X 平移)
GAL_IMG_W_IN = CELL_IN * GAL_IMG_FRAC   # 格内图像宽 1.271 in
GAL_INSET = (CELL_IN - GAL_IMG_W_IN) / 2
IMG_X = [c * FIG_W_IN + GAL_INSET for c in COLS[:-1]]    # 各列图像左缘(in)
IMG_R = [c * FIG_W_IN - GAL_INSET for c in COLS[1:]]     # 各列图像右缘(in)
GAL_IMG_H_IN = GAL_IMG_W_IN * 800 / 1200
GAL_PITCH_IN = GAL_IMG_H_IN + 0.56      # 行距:格题 3.5 pt/注记 5 pt 净空 +
                                         # 8/8.5 pt 行高,注记底与下一行格题
                                         # 仍隔 ~5 mm
GAL_TOP_IN = 0.32                       # 页边 + 首行标题带
GAL_CAP_IN = 0.24                       # 末行格下注记带(注记底再留空)
GAL_BOTTOM_IN = GAL_TOP_IN + 2 * GAL_PITCH_IN + GAL_IMG_H_IN + GAL_CAP_IN
GAL_LEGEND_IN = 0.43                    # 画廊下方图例带(居中;注记→图例、
                                        # 图例→面板字母 b/c 净空各 ≥4 mm)

ROW2_TOP_IN = GAL_BOTTOM_IN + GAL_LEGEND_IN + 0.12   # b(结构)/ c(序列条)
B_IMG_FRAC = 0.92       # b 渲染显示宽 = 列宽的 0.92(0.62→0.75→0.85→0.92
                        # 四轮放大,已近左半区极限:两幅渲染净空仍 ~3.6 mm,
                        # b2 右缘距 c ≥2 mm);仍列心对齐,编号 b 仍锚定
                        # 画廊列 1 图像左缘
B_IMG_W_IN = CELL_IN * B_IMG_FRAC
ROW2_H_IN = 1.42        # 0.65 → 0.95 → 1.18 → 1.32 → 1.42(第六轮再加大):
                        # 容纳 b 渲染上下各一行文字 + 放大后的渲染,
                        # c 序列条行距同步放宽
ROW3_TOP_IN = ROW2_TOP_IN + ROW2_H_IN + 0.34         # d/e/f 统计图(行距放宽)
ROW3_H_IN = 1.70        # 第五轮 1.50 → 1.70:行 3 面板整体放大
FIG_H_IN = ROW3_TOP_IN + ROW3_H_IN + 0.21            # ≈ 212 mm(≤247 mm)

# 行 3 面板宽度重排(2026-10-23 第二/四轮应作者要求:d 收窄,e/f 加宽,
# e–f 间隔加大):d 左缘仍让位 y 刻度签(编号 d 留在画廊列 1 图像左缘);
# d 右缘自第 3 列边界收进 0.30 in;e 无 y 刻度,左缘越过第 3 列边界 0.18 in
# 吃进 d 让出的宽度(编号 e 仍锚定第 3 列边界,与编号 c 对齐);
# [e 右缘, f 左缘] 0.42 in:f 的 y 刻度签 + 轴标题共占 ~0.26 in,余
# ~0.16 in(≈4 mm)作 e 内容与 f 轴外文字的净空(第四轮自 0.29 加宽,
# f 的纵坐标不再贴住 e);f 右缘略越第 4 列边界,右页边仍 ≥1 mm
D_X0_IN = IMG_X[0] + 0.17               # 0.126→0.17:8/8.5 pt 刻度签+轴题更宽
D_X1_IN = COLS[2] * FIG_W_IN - 0.42     # 第六轮再收窄,让宽给 e–f 间隔
E_X0_IN = COLS[2] * FIG_W_IN - 0.28     # 第六轮再左移,e 宽度不减
F_X0_IN = COLS[3] * FIG_W_IN - 0.055
F_GAP_IN = 0.56                         # e 右缘 → f 左缘(第六轮 0.46→0.56):
                                        # f 纵坐标文字(刻度签+轴题)与 e 内容
                                        # 净空自 ~5 mm 增至 ~9 mm
E_X1_IN = F_X0_IN - F_GAP_IN
F_X1_IN = COLS[4] * FIG_W_IN + 0.028


def _rect(x0_in, y_top_in, w_in, h_in):
    """左上角 + 宽高(英寸,自页顶量)→ fig-fraction 矩形。"""
    return [x0_in / FIG_W_IN, (FIG_H_IN - y_top_in - h_in) / FIG_H_IN,
            w_in / FIG_W_IN, h_in / FIG_H_IN]


def make_canvas():
    """返回 (fig, L):L 给出 b/c/d/e/f 各面板的 axes 矩形(fig fraction)。"""
    apply_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "axes.labelsize": 8.5,
    })
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    L = {
        # b 两槽:axes 精确包裹渲染图(宽 = 画廊图像宽,高按渲染图实际宽高比,
        # 由 fig6_b 读图后自算),顶部自 ROW2_TOP 下移一个标题带;此处先给
        # 占位矩形(宽度正确),fig6_b.draw 内覆盖为精确矩形
        "b1": _rect(IMG_X[0], ROW2_TOP_IN, GAL_IMG_W_IN, ROW2_H_IN),
        "b2": _rect(IMG_X[1], ROW2_TOP_IN, GAL_IMG_W_IN, ROW2_H_IN),
        "c":  _rect(COLS[2] * FIG_W_IN + 0.015, ROW2_TOP_IN,
                    COLS[4] * FIG_W_IN - COLS[2] * FIG_W_IN - 0.015, ROW2_H_IN),
        "d":  _rect(D_X0_IN, ROW3_TOP_IN, D_X1_IN - D_X0_IN, ROW3_H_IN),
        "e":  _rect(E_X0_IN, ROW3_TOP_IN, E_X1_IN - E_X0_IN, ROW3_H_IN),
        "f":  _rect(F_X0_IN, ROW3_TOP_IN, F_X1_IN - F_X0_IN, ROW3_H_IN),
    }
    return fig, L


def panel_label_at(fig, letter, x_in, y_top_in, size=10):
    """按画廊网格的固定 x(英寸,自左)放面板字母,与 fig_style.panel_label
    同样式(粗体小写、axes 顶外 1.2 mm),用于左缘让位给刻度签的 d 面板。"""
    dx = (1.2 * MM) / FIG_W_IN
    dy = (1.2 * MM) / FIG_H_IN
    fig.text(x_in / FIG_W_IN - dx, 1 - y_top_in / FIG_H_IN + dy, letter,
             fontsize=size, fontweight="bold", va="bottom", ha="left",
             color="black")


# PyMOL 渲染图(由 fig6_pymol.py 生成,画幅比例与槽位一致)
PYMOL_OVERVIEW = os.path.join(ROOT, "figures", "out", "fig6_pymol_overview.png")
PYMOL_POCKET = os.path.join(ROOT, "figures", "out", "fig6_pymol_pocket.png")


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
    seqs["PACo-Aptamer"] = best_seq_from_pool(
        load("paco_av2_pool_ens2noiface/8HZD_A.json")["pool"], natA)
    rd = []
    for t in ("0.1", "0.5", "1.0"):
        rd += [l.strip() for l in open(
            os.path.join(SUR, f"rhodesign_out_av2_retrained/8HZD_A.{t}.seqs"))
            if l.strip()]
    seqs["RhoDesign"] = best_seq_from_pool(rd, natA)
    gr = []
    for t in ("0.1", "0.5", "1.0"):
        gr += [l.strip() for l in open(
            os.path.join(SUR, f"grnade_out_av2_retrained_cfix/8HZD_A.{t}.seqs")) if l.strip()]
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
