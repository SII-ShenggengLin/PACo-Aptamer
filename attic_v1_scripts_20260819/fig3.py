# -*- coding: utf-8 -*-
"""Fig 3 | Zero-shot 配体级隔离泛化(协议贡献独立成节)。

a 三重隔离协议总览(结构簇 8:1:1 + 时间切分 + 配体级隔离层级图解)
b,c zs 93 / indep 235 结果(柱状 + head-to-head vs RhoDesign)
d 逐链 Δ(PACo − RhoDesign)分布(zs / indep 双小提琴)
数据:survey/alayer_v2_retrained.json(per_chain + views)、
81_wilcoxon.json、data/splits_v2/splits_v2.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (  # noqa: E402
    MM, METHOD_COLOR, METHOD_ORDER, ROOT, SUR, PACO, apply_style, load,
    panel_label, save)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

rng = np.random.default_rng(11)


def pfmt_p(p):
    if p < 1e-4:
        m = int(np.floor(np.log10(p)))
        return f"P = {p/10**m:.1f} × 10$^{{{m}}}$"
    return f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"


al = load("alayer_v2_retrained.json")
pc = al["per_chain"]
wil = load("81_wilcoxon.json")
sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
views = al["views"]
MKEY = {m: ("PACoAptamer" if m == "PACo-Aptamer" else m) for m in METHOD_ORDER}

fig = plt.figure(figsize=(183 * MM, 122 * MM))
gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.52, wspace=1.9)

# ============ a 三重隔离图解(跨两行) ============
axa = fig.add_subplot(gs[:, 0:3])
axa.set_xlim(0, 10); axa.set_ylim(0, 12); axa.axis("off")
panel_label(axa, "a", x=-0.045, y=1.045)

# 层1:结构簇
axa.add_patch(FancyBboxPatch((0.4, 7.6), 9.2, 3.6,
                             boxstyle="round,pad=0.15", fc="#F0EDE8",
                             ec="#8C8C88", lw=0.6))
axa.text(5, 10.75, "1 · Cluster-level holdout", ha="center", fontsize=6.0,
         fontweight="bold", color="#2E5F8A")
axa.text(5, 9.5, "300 clusters\nCD-HIT 0.8 + TM ≥ 0.45\nclosure → 8 : 1 : 1",
         ha="center", va="center", fontsize=5.2)
axa.text(0.85, 8.05, "train 5,773 · val 96 · test 477", fontsize=4.9,
         color="#545450")

# 层2:时间
axa.add_patch(FancyBboxPatch((0.9, 4.4), 8.2, 2.6,
                             boxstyle="round,pad=0.15", fc="#DCE6EF",
                             ec="#8C8C88", lw=0.6))
axa.text(5, 6.55, "2 · Temporal split", ha="center", fontsize=6.0,
         fontweight="bold", color="#2E5F8A")
axa.text(5, 5.35, "independent test, n = 235\nchains released 2024+",
         ha="center", va="center", fontsize=5.2)

# 层3:配体级(高亮橙)
axa.add_patch(FancyBboxPatch((1.4, 1.2), 7.2, 2.6,
                             boxstyle="round,pad=0.15", fc="#F6E3DC",
                             ec="#E8A088", lw=0.9))
axa.text(5, 3.35, "3 · Ligand-level isolation", ha="center", fontsize=6.0,
         fontweight="bold", color="#B0755F")
axa.text(5, 2.15, "zero-shot targets, n = 93\nligands unseen during training",
         ha="center", va="center", fontsize=5.2)
axa.text(5, 0.45, "ligand sets: S2 \u2229 zero-shot = empty  ·  chains: S2 \u2229 all test = empty",
         ha="center", fontsize=4.9, color="#B0755F", style="italic")

for y0, y1 in [(7.6, 7.0), (4.4, 3.8)]:
    axa.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>",
                                  mutation_scale=8, color="#545450", lw=0.7))

# ============ b/c 柱状 + h2h + Δ 小提琴 ============
def draw_view(view, wilkey, row, letter_bar, letter_h2h, letter_vio, title):
    # 柱状
    axb = fig.add_subplot(gs[row, 3:6])
    panel_label(axb, letter_bar, x=-0.145, y=1.14)
    vb = [100 * views[view][MKEY[m]]["best"] for m in METHOD_ORDER]
    vt = [100 * views[view][MKEY[m]]["t01"] for m in METHOD_ORDER]
    xs = np.arange(len(METHOD_ORDER)); w = 0.38
    axb.bar(xs - w / 2, vb, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
            ec="white", lw=0.3)
    axb.bar(xs + w / 2, vt, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
            alpha=0.45, ec="white", lw=0.3)
    for x, v in zip(xs - w / 2, vb):
        axb.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=4.8,
                 fontweight="bold" if v == max(vb) else "normal")
    for x, v in zip(xs + w / 2, vt):
        axb.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=4.8, alpha=0.75)
    wj = wil[wilkey]["RhoDesign"]["recovery_best"]
    wjt = wil[wilkey]["RhoDesign"]["recovery_t01_mean"]
    axb.text(0.98, 0.985, (f"vs RhoDesign: Δ = +{100*wj['delta']:.1f} pp, "
            f"{pfmt_p(wj['p_two_sided'])}\n(t = 0.1: Δ = +{100*wjt['delta']:.1f} pp, "
            f"{pfmt_p(wjt['p_two_sided'])})"),
            transform=axb.transAxes, fontsize=4.7, ha="right", va="top",
            bbox=dict(fc="white", ec="#C4C4C0", lw=0.4, pad=1.5))
    axb.set_xticks(xs)
    axb.set_xticklabels([m.replace("-Aptamer", "") for m in METHOD_ORDER],
                        rotation=30, ha="right")
    axb.set_ylabel("Sequence recovery (%)")
    axb.set_title(title, fontsize=5.6, pad=3.0)
    axb.set_ylim(0, 116)

    # head-to-head
    axh = fig.add_subplot(gs[row, 6:8])
    panel_label(axh, letter_h2h, x=-0.21, y=1.14)
    stems = sorted(sp[view])
    x = np.array([pc[s]["RhoDesign"]["recovery_best"] for s in stems]) * 100
    y = np.array([pc[s]["PACoAptamer"]["recovery_best"] for s in stems]) * 100
    lim = (20, 100)
    axh.fill_between([0, 100], [0, 100], [100, 100], color="#EDF2F7", zorder=0)
    axh.plot(lim, lim, color="#8C8C88", lw=0.6, ls="--", zorder=1)
    axh.scatter(x[y <= x], y[y <= x], s=4, c=PACO, alpha=0.6, lw=0, zorder=2)
    axh.scatter(x[y > x], y[y > x], s=4, c="#B8B8B4", alpha=0.6, lw=0, zorder=2)
    win = float(np.mean(y > x))
    axh.text(0.06, 0.94, f"PACo-Aptamer wins {100*win:.0f}%\nof chains",
             transform=axh.transAxes, va="top", fontsize=5.2)
    axh.set_xlim(lim); axh.set_ylim(lim); axh.set_aspect("equal")
    axh.set_xlabel("RhoDesign recovery (%)")
    axh.set_ylabel("PACo-Aptamer recovery (%)")

    # Δ 小提琴
    axv = fig.add_subplot(gs[row, 8:10])
    panel_label(axv, letter_vio, x=-0.21, y=1.14)
    dv = (y - x)
    vp = axv.violinplot([dv], widths=0.75, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor("#DCE6EF"); b.set_edgecolor(PACO); b.set_alpha(0.9)
    axv.scatter(rng.normal(0, 0.05, len(dv)), dv, s=2.5, c=PACO, alpha=0.45,
                lw=0)
    axv.axhline(0, color="#545450", lw=0.6)
    axv.axhline(np.mean(dv), color="black", lw=0.9)
    axv.text(0.97, 0.955, f"mean Δ = {np.mean(dv):+.1f} pp\nmedian {np.median(dv):+.1f}",
             transform=axv.transAxes, ha="right", va="top", fontsize=5.0,
             bbox=dict(fc="white", ec="none", pad=1.0))
    axv.set_xticks([]); axv.set_xlim(-0.65, 0.65)
    axv.set_ylabel("Δ recovery vs RhoDesign (pp)")


draw_view("zero_shot_targets", "A_zs_93", 0, "b", "c", "d",
          "zero-shot targets (n = 93, ligand-level isolation)")
draw_view("independent_test", "A_indep_235", 1, "e", "f", "g",
          "independent test (n = 235, 2024+ temporal split)")

save(fig, "fig3_zeroshot_protocol")
