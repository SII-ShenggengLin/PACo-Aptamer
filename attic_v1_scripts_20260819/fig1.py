# -*- coding: utf-8 -*-
"""Fig 1 | PACo-Aptamer 架构与数据(overview)。

a 范式示意(现有方法 vs 靶标条件化)
b 数据漏斗(PDB → 纯 RNA/复合物 → 双重聚类 → 三重隔离划分)
c 模型架构数据流(GVP + 配体 token 混合序列 Transformer + 自回归解码)
d 两阶段训练(S1 预训练 → S2 复合物微调 → ens2 集成)
e 主结果预览 mini-panel
数据:survey/btest_v2_main_table.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (  # noqa: E402
    LIGAND, METHOD_COLOR, METHOD_ORDER, MM, PACO, PACO_D, apply_style, load,
    panel_label, save)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

main = load("btest_v2_main_table.json")["table"]

C_DATA, C_MODEL, C_MET, C_SS = "#F0EDE8", "#DCE6EF", "#E9E2D8", "#E4E9E1"
EC = "#8C8C88"


def box(ax, x, y, w, h, fc, txt, ec=EC, lw=0.5, fs=5.0, bold=False,
        style="round,pad=0.10", tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=lw))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc or "black", fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color="#545450", lw=0.7, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=8, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def rna_icon(ax, cx, cy, s=1.0, color="#545450", lw=0.9):
    """简化 RNA 折叠 icon(平滑路径)。"""
    t = np.linspace(0, 4 * np.pi, 120)
    x = cx + s * 0.09 * t * np.cos(t * 0.22)
    y = cy + s * 0.32 * np.sin(t)
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round")


def lig_icon(ax, cx, cy, s=1.0, color=LIGAND):
    """小分子 icon(六元环 + 取代基)。"""
    ang = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
    ax.plot(cx + s * 0.6 * np.cos(ang), cy + s * 0.6 * np.sin(ang),
            color=color, lw=1.0)
    for a in [0.3, 2.4, 4.5]:
        ax.plot([cx + s * 0.6 * np.cos(a), cx + s * 1.1 * np.cos(a)],
                [cy + s * 0.6 * np.sin(a), cy + s * 1.1 * np.sin(a)],
                color=color, lw=0.8)


# ---------------- 画布 ----------------
fig = plt.figure(figsize=(183 * MM, 138 * MM))
gs = gridspec.GridSpec(2, 8, figure=fig, hspace=0.42, wspace=0.9,
                       height_ratios=[1.0, 1.12])

# ============ a 范式示意 ============
axa = fig.add_subplot(gs[0, 0:3])
axa.set_xlim(0, 10); axa.set_ylim(0, 10); axa.axis("off")
panel_label(axa, "a", x=-0.045, y=1.10)
# 左半:现有方法
box(axa, 0.1, 5.6, 4.4, 3.3, "white", "", ec="#C4C4C0", lw=0.6)
axa.text(2.3, 8.55, "Existing inverse folding", ha="center", fontsize=5.6,
         style="italic", color="#545450")
rna_icon(axa, 1.15, 7.3, s=1.0)
axa.text(1.15, 5.95, "RNA backbone", ha="center", fontsize=4.7,
         color="#545450")
arrow(axa, 1.9, 7.3, 2.75, 7.3)
box(axa, 2.8, 6.5, 1.5, 1.6, C_MODEL, "seq\ndesign", fs=4.7)
axa.text(3.55, 5.95, "sequence", ha="center", fontsize=4.7, color="#545450")
# 靶标配体(在模型之外,虚线断开)
lig_icon(axa, 3.7, 9.5, s=0.5)
axa.plot([3.7, 3.7], [8.95, 9.1], color="#B07060", lw=0.8, ls=":")
axa.text(4.35, 9.5, "target ligand not in model", fontsize=4.6,
         color="#B07060", ha="center", va="center")
axa.plot([5.15, 5.5], [9.3, 9.7], color="#B07060", lw=1.1)
axa.plot([5.15, 5.5], [9.7, 9.3], color="#B07060", lw=1.1)
# 右半:PACo
box(axa, 5.0, 5.6, 4.9, 3.3, "#F3F7FA", "", ec=PACO, lw=0.8)
axa.text(7.45, 8.55, "PACo-Aptamer (this work)", ha="center", fontsize=5.6,
         style="italic", color=PACO_D)
rna_icon(axa, 5.95, 7.3, s=0.95, color=PACO_D)
axa.text(5.7, 6.2, "backbone", ha="center", fontsize=4.7, color="#545450")
lig_icon(axa, 6.6, 8.15, s=0.7)
axa.text(7.5, 6.2, "ligand context", fontsize=4.7,
         color="#B0755F", ha="center", va="center")
# SS 弧线
for dy in (0.0, 0.24):
    tt = np.linspace(-0.42, 0.42, 20)
    axa.plot(7.35 + tt, 6.35 + dy + 0.08 * np.cos(tt * 7),
             color="#7FA88B", lw=0.7)
axa.text(6.5, 5.62, "SS pairs", ha="center", fontsize=4.7, color="#5A7A62")
arrow(axa, 6.95, 7.25, 7.72, 7.25, color=PACO_D)
box(axa, 7.8, 6.5, 1.6, 1.6, C_MODEL, "seq\ndesign", fs=4.7, ec=PACO)
axa.text(8.75, 5.62, "sequence", ha="center", fontsize=4.7, color="#545450")
axa.text(4.98, 7.3, "vs", ha="center", va="center", fontsize=5.5,
         color="#8C8C88")
# 下方对比注释
box(axa, 0.1, 0.4, 4.4, 4.4, C_DATA,
    "folding-centric:\nsequence must fold back\nto the given structure",
    fs=4.8, ec="#C4C4C0")
box(axa, 5.0, 0.4, 4.9, 4.4, "#F6E3DC",
    "binding-centric:\nligand pocket atoms condition\nthe design (training-time\ncalibration of interface)",
    fs=4.8, ec="#E8A088", tc="#6B4A3E")

# ============ b 数据漏斗 ============
axb = fig.add_subplot(gs[0, 3:8])
axb.set_xlim(0, 20); axb.set_ylim(0, 10); axb.axis("off")
panel_label(axb, "b", x=-0.028, y=1.10)
# 顶层
box(axb, 0.2, 8.35, 5.4, 1.3, C_DATA, "PDB RNA entries\n24,252", fs=5.0)
# 两流
arrow(axb, 2.9, 8.35, 2.9, 7.75, rad=0.0)
box(axb, 0.2, 6.25, 5.4, 1.35, "white",
    "RNA-only chains 6,988\n→ S1 pretraining", fs=4.8, ec="#C4C4C0")
arrow(axb, 5.6, 9.0, 8.2, 9.0)
box(axb, 8.2, 8.35, 5.2, 1.3, C_DATA, "RNA–ligand complexes\n868", fs=5.0)
arrow(axb, 10.8, 8.35, 10.8, 7.75)
box(axb, 8.2, 6.25, 5.2, 1.35, "white",
    "→ S2 finetuning\n(canonical 6,346 merged set)", fs=4.8, ec="#C4C4C0")
# 聚类
arrow(axb, 2.9, 6.25, 6.3, 5.35, rad=-0.15)
arrow(axb, 10.8, 6.25, 7.5, 5.35, rad=0.15)
box(axb, 3.6, 3.95, 6.6, 1.35, C_SS,
    "CD-HIT 0.8 + TM ≥ 0.45\ntransitive closure → 300 clusters", fs=4.8)
arrow(axb, 6.9, 3.95, 6.9, 3.3)
# 划分行
splits = [("train", "5,773", C_MODEL), ("val", "96", C_MODEL),
          ("test", "477", C_MET), ("indep", "235", C_MET),
          ("zs", "93", "#F6E3DC")]
xw = 3.35
for i, (nm, cnt, fc) in enumerate(splits):
    x0 = 0.55 + i * (xw + 0.28)
    ec = "#E8A088" if nm == "zs" else EC
    box(axb, x0, 1.45, xw, 1.5, fc, f"{nm}\n{cnt}", fs=4.9,
        ec=ec, lw=0.9 if nm == "zs" else 0.5,
        bold=(nm == "zs"))
    arrow(axb, 6.9, 3.3, x0 + xw / 2, 3.05, rad=0.0 if i == 2 else
          (0.12 if i < 2 else -0.12), lw=0.5)
axb.text(9.2, 0.55, "cluster-wise 8 : 1 : 1 · 511 chains discarded",
         fontsize=4.7, color="#545450", ha="center", va="center")
axb.annotate("ligand-level isolation", xy=(16.7, 1.45), xytext=(16.7, 0.55),
             fontsize=4.7, color="#B0755F", ha="center", va="center",
             arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#B0755F"))
box(axb, 14.0, 5.3, 5.8, 4.35, "white",
    "Quasi-blind protocol\n\n• cluster holdout (TM ≥ 0.45)\n"
    "• temporal split (2024+)\n• ligand-level isolation\n"
    "  (test ligands unseen)", fs=4.7, ec="#C4C4C0")

# ============ c 架构图 ============
axc = fig.add_subplot(gs[1, 0:4])
axc.set_xlim(0, 20); axc.set_ylim(0, 10); axc.axis("off")
panel_label(axc, "c", x=-0.035, y=1.12)
# 输入三 chips
box(axc, 0.2, 7.6, 4.3, 2.0, C_DATA,
    "RNA backbone\n9 atoms / nt (C1'-centered\nframes, dihedrals)", fs=4.6)
box(axc, 0.2, 4.9, 4.3, 2.0, C_SS, "SS pairing map\nRNAfold MFE base pairs\n(edge features)", fs=4.6,
    tc="#3D5A45")
box(axc, 0.2, 2.2, 4.3, 2.0, "#F6E3DC",
    "Ligand pocket atoms\nelement one-hot + dist. RBF\n(token per atom)", fs=4.6,
    ec="#E8A088")
# GVP
for y0 in (7.6, 4.9):
    arrow(axc, 4.5, y0 + 1.0, 5.5, 6.3, rad=0.1)
arrow(axc, 4.5, 3.2, 5.5, 4.0, rad=-0.1, color="#B0755F")
box(axc, 5.5, 5.15, 3.4, 2.6, C_MODEL,
    "GVP encoder ×3\nscalar 512 · vector 256\nSS pair edges", fs=4.6)
box(axc, 5.5, 1.8, 3.4, 2.2, "#D8C8B0",
    "ligand embedding\n2-layer MLP (512)", fs=4.6, ec="#E8A088")
arrow(axc, 8.9, 6.45, 9.9, 6.45)
arrow(axc, 8.9, 2.9, 9.9, 5.0, color="#B0755F")
box(axc, 9.9, 3.9, 4.6, 4.2, C_MODEL,
    "Transformer encoder ×3\n512 d · 4 heads\n\nmixed sequence:\nRNA tokens + ligand tokens\n(joint attention)", fs=4.6)
arrow(axc, 14.5, 6.0, 15.5, 6.0)
box(axc, 15.5, 3.9, 4.3, 4.2, C_MODEL,
    "Transformer\ndecoder ×3\n\nautoregressive\nA/C/G/U\n(temperature\nsampling)", fs=4.6)
axc.text(17.65, 3.35, "sequence", ha="center", fontsize=4.8, color=PACO_D,
         fontweight="bold")
axc.text(10, 1.4, "cross-attention carries ligand context to every decoding step",
         fontsize=4.5, color="#B0755F", ha="center", style="italic")

# ============ d 两阶段训练 ============
axd = fig.add_subplot(gs[1, 4:6])
axd.set_xlim(0, 10); axd.set_ylim(0, 10); axd.axis("off")
panel_label(axd, "d", x=-0.17, y=1.12)
box(axd, 0.2, 6.3, 4.1, 2.6, C_MODEL,
    "S1 pretraining\n6,988 RNA-only chains\nstructure → sequence", fs=4.6)
arrow(axd, 2.25, 6.3, 2.25, 5.5)
box(axd, 0.2, 2.8, 4.1, 2.6, C_MODEL,
    "S2 finetuning\n+ ligand context\n(complex chains)", fs=4.6)
arrow(axd, 4.3, 4.1, 5.3, 4.1)
box(axd, 5.3, 2.8, 4.5, 2.6, "#DCE6EF",
    "ens2 ensemble\n2 S2 checkpoints\nstepwise softmax\nprobability averaging", fs=4.6,
    ec=PACO, lw=0.8)
axd.text(2.45, 5.9, "60 epochs", fontsize=4.3, ha="left", color="#545450")
axd.text(2.25, 2.4, "8 epochs", fontsize=4.3, ha="center", color="#545450")
axd.text(5.0, 9.2, "two-stage training", fontsize=5.6, ha="center",
         style="italic", color=PACO_D)
axd.text(7.55, 2.3, "val-selected,\npool size unchanged", fontsize=4.3,
         ha="center", color="#545450")

# ============ e 主结果预览 ============
axe = fig.add_subplot(gs[1, 6:8])
panel_label(axe, "e", x=-0.24, y=1.12)
tk = lambda m: main["PACoAptamer" if m == "PACo-Aptamer" else m]
vb = [100 * tk(m)["recovery_best_mean"] for m in METHOD_ORDER]
xe = np.arange(len(METHOD_ORDER))
axe.bar(xe, vb, 0.62, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
        ec="white", lw=0.3)
for x0, v in zip(xe, vb):
    axe.text(x0, v + 1.4, f"{v:.1f}", ha="center", fontsize=4.8,
             fontweight="bold" if v == max(vb) else "normal")
axe.set_xticks(xe)
axe.set_xticklabels([m.replace("-Aptamer", "") for m in METHOD_ORDER],
                    rotation=32, ha="right")
axe.set_ylabel("Recovery, pool-in-best (%)")
axe.set_title("splits_v2 test (n = 477)", fontsize=5.2, pad=3)
axe.set_ylim(0, 92)
axe.text(0.97, 0.03, "preview of Fig. 2", transform=axe.transAxes,
         fontsize=4.4, ha="right", color="#8C8C88", style="italic")

save(fig, "fig1_overview")
