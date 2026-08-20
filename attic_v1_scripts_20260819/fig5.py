# -*- coding: utf-8 -*-
"""Fig 5 | 生成质量:多样性、新颖性、进化保真。

a 温度扫描 recovery–diversity 前沿(温度渐变着色)
b 新颖性 rank-ordered 最大相似度曲线(vs 训练库)
c 生成池 4-mer 组成 t-SNE(60 链 × 5 方法 × 30 设计 + native)
d 家族组成保真 KL(LOO TM 簇 profile,含随机参照锚)
e 序列可行性 GC–MFE 2D 图(native 五角星参照)
数据:survey/{82_temp_diversity, 83_novelty, 85_tsne_points,
85_family_kl, 82_gc_mfe}.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (  # noqa: E402
    GRNADE, METHOD_COLOR, METHOD_ORDER, MM, NATIVE, RHODESIGN, TACO,
    apply_style, load, panel_label, save)


def _lighten(hex_color, frac):
    """把 hex 颜色向白色混合 frac(0-1),用于温度渐变。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r += int((255 - r) * frac); g += int((255 - g) * frac); b += int((255 - b) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------- 数据 ----------------
td = load("82_temp_diversity.json")
nov = load("83_novelty.json")
ts = load("85_tsne_points.json")
kl = load("85_family_kl.json")
gc = load("82_gc_mfe.json")

# ---------------- 画布 ----------------
fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.52, wspace=1.15,
                       height_ratios=[1.0, 1.05])

# ============ a 温度权衡 ============
axa = fig.add_subplot(gs[0, 0:2])
panel_label(axa, "a", x=-0.19, y=1.12)
temps = ["t=0.1", "t=0.5", "t=1.0"]
for m in ["TACo", "RhoDesign", "gRNAde"]:
    xs = [100 * td[m][t]["div"]["mean"] for t in temps]
    ys = [100 * td[m][t]["rec"]["mean"] for t in temps]
    base = METHOD_COLOR[m]
    shades = [base, _lighten(base, 0.35), _lighten(base, 0.62)]
    axa.plot(xs, ys, color=base, lw=0.7, alpha=0.7, zorder=2)
    for x, y, c, t in zip(xs, ys, shades, temps):
        axa.scatter(x, y, s=16, c=[c], edgecolors=base, linewidths=0.6, zorder=3)
axa.annotate("t = 0.1", (23.9, 59.6), fontsize=5.0, color=TACO,
             xytext=(25.8, 64.6), textcoords="data",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=TACO))
axa.annotate("t = 1.0", (31.5, 58.4), fontsize=5.0, color=TACO,
             xytext=(33.8, 62.5), textcoords="data", ha="right",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=TACO))
axa.annotate("t = 0.1, uniq 0.88", (8.2, 54.6), fontsize=4.8, color=RHODESIGN,
             xytext=(11.0, 58.8), textcoords="data",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=RHODESIGN))
axa.annotate("mode collapse:\n70% duplicate sequences", (8.2, 54.6),
             fontsize=4.8, color=RHODESIGN, xytext=(21.0, 30.5),
             textcoords="data", ha="center",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=RHODESIGN,
                             ls=":", connectionstyle="arc3,rad=-0.25"))
axa.axhline(25, color="#B07060", lw=0.5, ls=":")
axa.text(5.0, 26.0, "random", fontsize=4.6, color="#B07060", ha="left")
axa.set_xlabel("Pool diversity (mean pairwise distance)")
axa.set_ylabel("Recovery, t-mean (%)")
handles = [Line2D([], [], marker="o", ls="-", color=METHOD_COLOR[m],
                  label=m if m != "TACo" else "TACo-Aptamer",
                  markersize=3.5, markerfacecolor=_lighten(METHOD_COLOR[m], 0.35))
           for m in ["TACo", "RhoDesign", "gRNAde"]]
axa.legend(handles=handles, loc="lower left", fontsize=5.0)
axa.set_xlim(4, 36)

# ============ b 新颖性 rank 曲线 ============
axb = fig.add_subplot(gs[0, 2:5])
panel_label(axb, "b", x=-0.09, y=1.12)
for m in METHOD_ORDER:
    k = "TACo" if m == "TACo-Aptamer" else m
    rc = np.array(nov["methods"][k]["train"]["rank_curve"])
    xq = np.linspace(0, 1, len(rc))
    axb.plot(xq, rc, lw=0.9, color=METHOD_COLOR[m],
             label=f"{m} (median {nov['methods'][k]['train']['median']:.2f}, "
                   f"{100*nov['methods'][k]['train']['frac_gt_0.8']:.1f}% > 0.8)")
axb.axhline(0.8, color="#B07060", lw=0.5, ls=":")
axb.text(0.99, 0.815, "identity = 0.8", fontsize=4.6, color="#B07060",
         ha="right", va="bottom")
med_apt = {m: nov["methods"]["TACo" if m == "TACo-Aptamer" else m]["aptadb"]["median"] for m in METHOD_ORDER}
for i, m in enumerate(METHOD_ORDER):
    axb.scatter([1.012], [med_apt[m]], marker="|", s=30, lw=1.0,
                color=METHOD_COLOR[m], clip_on=False)
axb.text(1.022, 0.30, "vs\nAptaDB\n(medians)", fontsize=4.4, va="center",
         ha="left", color="#545450", transform=axb.get_yaxis_transform())
axb.set_xlim(0, 1); axb.set_ylim(0, 1.02)
axb.set_xlabel("Designs ranked by max identity to training set")
axb.set_ylabel("Max sequence identity")
axb.legend(loc="upper left", fontsize=4.9, handlelength=1.4,
           labelspacing=0.25)

# ============ c t-SNE ============
axc = fig.add_subplot(gs[1, 0:3])
panel_label(axc, "c", x=-0.13, y=1.12)
lab = np.array(ts["label"]); x = np.array(ts["x"]); y = np.array(ts["y"])
for m in METHOD_ORDER:
    key = "TACo" if m == "TACo-Aptamer" else m
    sel = lab == key
    axc.scatter(x[sel], y[sel], s=1.2, c=METHOD_COLOR[m], alpha=0.30, lw=0)
sel = lab == "native"
axc.scatter(x[sel], y[sel], s=14, marker="x", c="#3D5A45", lw=0.9,
            label="native", zorder=5)
axc.set_xticks([]); axc.set_yticks([])
axc.set_xlabel("t-SNE 1"); axc.set_ylabel("t-SNE 2")
handles = [Line2D([], [], marker="o", ls="", color=METHOD_COLOR[m],
                  markersize=3, alpha=0.9,
                  label=m if m != "TACo-Aptamer" else "TACo-Aptamer")
           for m in METHOD_ORDER]
handles.append(Line2D([], [], marker="x", ls="", color="#3D5A45",
                      markersize=4, label="native"))
axc.legend(handles=handles, ncol=3, loc="upper right", fontsize=4.8,
           handletextpad=0.2, columnspacing=0.7)
axc.text(0.015, 0.02, "60 test chains × 30 designs per method (t = 0.1 pools), 4-mer composition",
         transform=axc.transAxes, fontsize=4.6, color="#545450")

# ============ d 家族 KL ============
axd = fig.add_subplot(gs[1, 3])
panel_label(axd, "d", x=-0.30, y=1.12)
mm = METHOD_ORDER
kvm = [kl["methods"][("TACo" if m == "TACo-Aptamer" else m)]["kl_mean"]
       for m in mm]
kci = [kl["methods"][("TACo" if m == "TACo-Aptamer" else m)]["kl_mean"] -
       kl["methods"][("TACo" if m == "TACo-Aptamer" else m)]["ci95"][0]
       for m in mm]
xd = np.arange(len(mm))
axd.bar(xd, kvm, 0.62, yerr=kci, color=[METHOD_COLOR[m] for m in mm],
        error_kw=dict(lw=0.5, capsize=1.3), ec="white", lw=0.3)
for x0, v in zip(xd, kvm):
    axd.text(x0, v + 0.10, f"{v:.2f}", ha="center", fontsize=4.8,
             fontweight="bold" if v == min(kvm) else "normal")
rr = kl["random_ref"]
axd.axhline(rr["kl_mean"], color="#B07060", lw=0.7, ls="--")
axd.text(0.02, rr["kl_mean"] - 0.16, f"random ref {rr['kl_mean']:.2f}",
         fontsize=4.7, color="#B07060", va="top", transform=axd.get_yaxis_transform())
axd.set_xticks(xd); axd.set_xticklabels(
    [m.replace("-Aptamer", "") for m in mm], rotation=32, ha="right")
axd.set_ylabel("KL(designs $\\|$ family profile)")
axd.set_ylim(3.0, 5.3)

# ============ e GC–MFE ============
axe = fig.add_subplot(gs[1, 4])
panel_label(axe, "e", x=-0.30, y=1.12)
nat = gc["native_ref"]
axd0 = None
LBL = {  # (文字锚点偏移, ha)
    "TACo": ((2.2, 0.018), "left"), "RhoDesign": ((-2.2, -0.030), "right"),
    "AlignIF": ((2.2, -0.016), "left"), "NA-MPNN": ((-2.2, 0.020), "right"),
    "gRNAde": ((2.2, 0.018), "left")}
for m in METHOD_ORDER:
    key = "TACo" if m == "TACo-Aptamer" else m
    g = gc["methods"][key]
    gx, gy = 100 * g["gc_mean"], g["mfe_nt_mean"]
    axe.scatter(gx, gy, s=22, c=METHOD_COLOR[m], zorder=4, ec="white", lw=0.4)
    off, ha = LBL[key]
    axe.annotate(key, (gx, gy), xytext=(gx + off[0], gy + off[1]),
                 fontsize=4.8, color=METHOD_COLOR[m], ha=ha, va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.4,
                                 color=METHOD_COLOR[m], shrinkA=0, shrinkB=3))
axe.scatter(100 * nat["gc_mean"], nat["mfe_nt_mean"], marker="*", s=120,
            c=NATIVE, zorder=6, ec="#3D5A45", lw=0.6)
axe.annotate("native", (100 * nat["gc_mean"], nat["mfe_nt_mean"]),
             xytext=(100 * nat["gc_mean"] - 2.4, nat["mfe_nt_mean"] + 0.026),
             fontsize=5.2, color="#3D5A45", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="-", lw=0.4, color="#3D5A45",
                             shrinkA=0, shrinkB=5))
axe.set_xlabel("GC content (%)")
axe.set_ylabel("MFE per nucleotide\n(kcal mol$^{-1}$ nt$^{-1}$)")
axe.set_xlim(42, 64); axe.set_ylim(-0.48, -0.17)

save(fig, "fig5_generation_quality")
