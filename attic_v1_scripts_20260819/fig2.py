# -*- coding: utf-8 -*-
"""Fig 2 | 严格防泄漏基准下的全面领先(主结果图)。

a 评估协议流程  b 六方法主表  c 恢复率柱状(best/t01 双口径+Wilcoxon P)
d,e head-to-head 逐链散点(vs RhoDesign / vs AlignIF)  f 逐靶 Δ 双向条形
g 界面恢复率(绝对+归一化)  h 置信度-恢复率相关(链间/链内/富集)
i 长度分层箱线
数据:survey/{btest_v2_main_table, 81_wilcoxon, 77_p02_distance_gradient,
77_p04_confidence_recovery, 77_ens2_pool_btest/, metrics_*_btest_v2}.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (  # noqa: E402
    MM, METHOD_COLOR, METHOD_ORDER, SUR, PACO, apply_style, load, panel_label,
    save)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

rng = np.random.default_rng(42)


def pfmt_p(p):
    if p < 1e-4:
        m = int(np.floor(np.log10(p)))
        return f"P = {p/10**m:.1f} × 10$^{{{m}}}$"
    return f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"


# ---------------- 数据 ----------------
main = load("btest_v2_main_table.json")["table"]
wil = load("81_wilcoxon.json")
p04 = load("77_p04_confidence_recovery.json")
p02 = load("77_p02_distance_gradient.json")["methods"]
IFACE_KEY = "≤5Å(界面)"

pc_paco = load("metrics_PACo_v2_ens2noiface_full.json")["per_chain"]
pc = {"PACo-Aptamer": {k: v["recovery_best"] for k, v in pc_paco.items()}}
for m, f, key in [("RhoDesign", "metrics_RhoDesign_btest_v2.json", "recovery_best_overall"),
                  ("gRNAde", "metrics_gRNAde_btest_v2.json", "recovery_best_overall"),
                  ("NA-MPNN", "metrics_NAMPNN_btest_v2.json", "recovery_best"),
                  ("AlignIF", "metrics_AlignIF_btest_v2.json", "recovery_best")]:
    pc[m] = {k: v[key] for k, v in load(f)["per_chain"].items()}
stems = sorted(pc_paco)
chain_len = {s: pc_paco[s]["len"] for s in stems}

# ens2 池一次性读入(链级散点 + 链内相关共用,77 号同口径:
# x=全池 300 条 mean logP,y=文件内 recovery_t01_mean,N 位计分母)
pool_dir = os.path.join(SUR, "77_ens2_pool_btest")
conf_x, conf_y, within = [], [], []
for s in stems:
    pf = os.path.join(pool_dir, f"{s}.json")
    if not os.path.exists(pf):
        continue
    d = json.load(open(pf))
    nt, pool, lps = d["native"], d["pool"], d["logp"]
    conf_x.append(float(np.mean(lps)))
    conf_y.append(100 * d["recovery_t01_mean"])
    if len(lps) >= 30 and np.std(lps) > 0:
        rc = [sum(1 for a, b in zip(sq, nt) if a == b) / len(nt) for sq in pool]
        if np.std(rc) > 0:
            r, _ = stats.spearmanr(lps, rc)
            if np.isfinite(r):
                within.append(float(r))
conf_x, conf_y, within = map(np.array, (conf_x, conf_y, within))
rho_chain, _ = stats.spearmanr(conf_x, conf_y)
# ρ 的 95% CI(bootstrap,链级重采样)
_rng = np.random.default_rng(0)
_rhos = []
for _ in range(2000):
    ii = _rng.integers(0, len(conf_x), len(conf_x))
    if np.std(conf_x[ii]) > 0 and np.std(conf_y[ii]) > 0:
        _rhos.append(stats.spearmanr(conf_x[ii], conf_y[ii]).statistic)
rho_ci = np.percentile(_rhos, [2.5, 97.5])

# ---------------- 画布 8 列网格 ----------------
fig = plt.figure(figsize=(183 * MM, 175 * MM))
gs = gridspec.GridSpec(4, 8, figure=fig, hspace=0.68, wspace=1.35,
                       height_ratios=[1.0, 1.3, 1.15, 1.2])

# ============ a 协议流程 ============
axa = fig.add_subplot(gs[0, 0:3]); axa.set_xlim(0, 10); axa.set_ylim(0, 6.4)
axa.axis("off")
panel_label(axa, "a", x=-0.045, y=1.14)
C_DATA, C_MODEL, C_MET = "#F0EDE8", "#DCE6EF", "#E9E2D8"
boxes = [
    (0.1, 4.0, 3.0, 1.85, C_DATA, "Conditioning input:\nRNA backbone + SS\n(backbone-only parity)"),
    (3.55, 4.0, 3.0, 1.85, C_MODEL, "Sequence pool\n3 T × 100\nper method"),
    (7.0, 4.0, 2.9, 1.85, C_MET, "Recovery:\nbest / t = 0.1\n+ interface (<5 Å)"),
    (0.1, 0.7, 3.0, 1.85, C_MODEL, "Same-data\nretraining\n(5 methods)"),
    (3.55, 0.7, 6.35, 1.85, C_MET, "Per-chain paired Wilcoxon + 95% CI bootstrap\nvs strongest baseline"),
]
for x, y, w, h, fc, txt in boxes:
    axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                 fc=fc, ec="#8C8C88", lw=0.5))
    axa.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=5.2)
for (x0, y0, x1, y1) in [(3.1, 4.9, 3.55, 4.9), (6.55, 4.9, 7.0, 4.9),
                         (1.6, 4.0, 1.6, 2.55), (3.1, 1.6, 3.55, 4.0)]:
    axa.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                  mutation_scale=7, color="#545450", lw=0.6))
axa.text(5, 6.15, "Quasi-blind evaluation protocol", ha="center",
         fontsize=6.4, style="italic", color="#2E5F8A")

# ============ b 主表 ============
axb = fig.add_subplot(gs[0, 3:8]); axb.axis("off")
panel_label(axb, "b", x=-0.052, y=1.05)
hdr = ["Method", "Pool-in-best (%)", "t = 0.1 mean (%)", "Coverage", "Pool size"]
cell = []
for m in METHOD_ORDER:
    t = main["PACoAptamer" if m == "PACo-Aptamer" else m]
    pool = {"PACo-Aptamer": "3T × 100 × 2", "RhoDesign": "3T × 100",
            "AlignIF": "T0.1 × 20", "NA-MPNN": "T0.1 × 10",
            "gRNAde": "3T × 100"}[m]
    cell.append([m, f"{100*t['recovery_best_mean']:.2f}",
                 f"{100*t['recovery_t01_mean']:.2f}", t["coverage"], pool])
tb = axb.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(5.6); tb.scale(1, 1.7)
best_b = max(float(r[1]) for r in cell); best_t = max(float(r[2]) for r in cell)
for (i, j), c in tb.get_celld().items():
    c.set_linewidth(0.4); c.set_edgecolor("#C4C4C0")
    if i == 0:
        c.set_facecolor("#EFEDEA"); c.set_text_props(fontweight="bold")
    else:
        m = cell[i - 1][0]
        c.set_facecolor("#EDF2F7" if m == "PACo-Aptamer" else "white")
        if j in (1, 2):
            c.set_text_props(color=METHOD_COLOR[m])
            if float(cell[i - 1][j]) == (best_b if j == 1 else best_t):
                c.set_text_props(color=METHOD_COLOR[m], fontweight="bold")
axb.text(0.5, 0.015, "splits_v2 test, n = 477 chains · canonical native sequences · random baseline 25%",
         transform=axb.transAxes, ha="center", fontsize=5.0, color="#545450")

# ============ c 双口径柱状 ============
axc = fig.add_subplot(gs[1, 0:2])
panel_label(axc, "c", x=-0.24, y=1.1)
xs = np.arange(len(METHOD_ORDER)); w = 0.38
_tk = lambda m: main["PACoAptamer" if m == "PACo-Aptamer" else m]
vb = [100 * _tk(m)["recovery_best_mean"] for m in METHOD_ORDER]
vt = [100 * _tk(m)["recovery_t01_mean"] for m in METHOD_ORDER]
axc.bar(xs - w / 2, vb, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
        ec="white", lw=0.3)
axc.bar(xs + w / 2, vt, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
        alpha=0.45, ec="white", lw=0.3)
for x, v in zip(xs - w / 2, vb):
    axc.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=4.8,
             fontweight="bold" if v == max(vb) else "normal")
for x, v in zip(xs + w / 2, vt):
    axc.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=4.8, alpha=0.75)
axc.axhline(25, color="#B07060", lw=0.6, ls=":")
axc.text(4.4, 26.5, "random", fontsize=4.6, color="#B07060", ha="right")
pR = wil["B_test_477"]["RhoDesign"]["recovery_best"]
pRt = wil["B_test_477"]["RhoDesign"]["recovery_t01_mean"]
axc.text(0.98, 0.985, (f"vs RhoDesign (paired Wilcoxon, n = 477)\n"
        f"best: Δ = +{100*pR['delta']:.1f} pp, {pfmt_p(pR['p_two_sided'])}\n"
        f"t = 0.1: Δ = +{100*pRt['delta']:.1f} pp, {pfmt_p(pRt['p_two_sided'])}"),
         transform=axc.transAxes, fontsize=4.9, ha="right", va="top",
         bbox=dict(fc="white", ec="#C4C4C0", lw=0.4, pad=1.6))
axc.set_xticks(xs); axc.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
axc.set_ylabel("Sequence recovery (%)"); axc.set_ylim(0, 122)

# ============ d,e head-to-head ============
for ax, opp, letter in [
        (fig.add_subplot(gs[1, 3:5]), "RhoDesign", "d"),
        (fig.add_subplot(gs[1, 5:8]), "AlignIF", "e")]:
    panel_label(ax, letter, x=-0.21, y=1.1)
    x = np.array([pc[opp][s] for s in stems]) * 100
    y = np.array([pc["PACo-Aptamer"][s] for s in stems]) * 100
    lim = (5, 100)
    ax.fill_between([0, 100], [0, 100], [100, 100], color="#EDF2F7", zorder=0)
    ax.plot(lim, lim, color="#8C8C88", lw=0.6, ls="--", zorder=1)
    win = float(np.mean(y > x))
    ax.scatter(x[y <= x], y[y <= x], s=3.5, c=PACO, alpha=0.55, lw=0, zorder=2)
    ax.scatter(x[y > x], y[y > x], s=3.5, c="#B8B8B4", alpha=0.55, lw=0, zorder=2)
    wj = wil["B_test_477"][opp]["recovery_best"]
    ax.text(0.06, 0.94, (f"PACo-Aptamer wins {100*win:.0f}% of chains\n"
            f"Δ = +{100*wj['delta']:.1f} pp, {pfmt_p(wj['p_two_sided'])}"),
            transform=ax.transAxes, va="top", fontsize=5.4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{opp} recovery (%)")
    ax.set_ylabel("PACo-Aptamer recovery (%)")

# ============ f 逐靶 Δ 双向条形 ============
axf = fig.add_subplot(gs[2, 0:5])
panel_label(axf, "f", x=-0.075, y=1.12)
delta = np.array([pc["PACo-Aptamer"][s] - pc["RhoDesign"][s] for s in stems]) * 100
order = np.argsort(delta)
cc = load("74_chain_ccd.json")
iface_flag = np.array([s in cc for s in stems])[order]
dv = delta[order]
axf.bar(np.arange(len(dv))[dv >= 0], dv[dv >= 0], width=1.0, color=PACO,
        alpha=0.85, lw=0)
axf.bar(np.arange(len(dv))[dv < 0], dv[dv < 0], width=1.0, color="#A8A8A4",
        alpha=0.9, lw=0)
axf.scatter(np.flatnonzero(iface_flag), np.full(int(iface_flag.sum()), -37.5),
            marker="s", s=1.6, c="#E8A088", lw=0, clip_on=False)
axf.axhline(0, color="#545450", lw=0.6)
med = float(np.median(delta))
axf.axhline(med, color=PACO, lw=0.7, ls=":")
axf.text(len(dv) * 0.998, med + 1.8, f"median {med:+.1f} pp", ha="right",
         fontsize=5.0, color=PACO)
axf.set_xlim(-5, len(dv) + 5); axf.set_ylim(-45, 64)
axf.set_xlabel(f"Chains ranked by per-chain Δ recovery (n = {len(dv)})")
axf.set_ylabel("Δ recovery vs\nRhoDesign (pp)")
axf.text(0.015, 0.035,
         f"{100*np.mean(delta > 0):.0f}% of chains positive · orange squares = ligand-interface chains",
         transform=axf.transAxes, fontsize=5.0, color="#545450",
         bbox=dict(fc="white", ec="none", pad=1.0))

# ============ g 界面恢复率 ============
axg1 = fig.add_subplot(gs[2, 5:8])
panel_label(axg1, "g", x=-0.135, y=1.12)
mrows = ["PACo_ens2", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]
mlab = ["PACo", "RhoDesign", "AlignIF", "NA-MPNN", "gRNAde"]
iv = [100 * p02[m][IFACE_KEY]["mean"] for m in mrows]
err = [100 * (p02[m][IFACE_KEY]["mean"] - p02[m][IFACE_KEY]["ci95"][0])
       for m in mrows]
nm = [100 * p02[m][IFACE_KEY]["norm_mean"] for m in mrows]
nme = [100 * (p02[m][IFACE_KEY]["norm_mean"] - p02[m][IFACE_KEY]["norm_ci95"][0])
       for m in mrows]
xg = np.arange(5); wg = 0.38
cols = [METHOD_COLOR[m] for m in mlab]
axg1.bar(xg - wg / 2, iv, wg, yerr=err, error_kw=dict(lw=0.5, capsize=1.2),
         color=cols, ec="white", lw=0.3)
for x, v in zip(xg - wg / 2, iv):
    axg1.text(x, v + 4.5, f"{v:.1f}", ha="center", fontsize=4.7,
              fontweight="bold" if v == max(iv) else "normal")
axg1.set_xticks(xg); axg1.set_xticklabels(mlab, rotation=30, ha="right")
axg1.set_ylabel("Interface recovery (%)"); axg1.set_ylim(0, 95)
axg2 = axg1.twinx()
axg2.bar(xg + wg / 2, nm, wg, yerr=nme, error_kw=dict(lw=0.5, capsize=1.2),
         color=cols, alpha=0.45, ec="white", lw=0.3)
axg2.axhline(0, color="#545450", lw=0.5)
axg2.set_ylabel("Δ (interface − chain mean, pp)", fontsize=5.6)
axg2.set_ylim(-26, 13)
axg2.spines["right"].set_visible(True); axg2.spines["right"].set_linewidth(0.5)
axg2.spines["top"].set_visible(False)
axg1.text(0.985, 0.97, "solid: absolute (left axis)\nfaded: normalized (right axis)",
          transform=axg1.transAxes, fontsize=4.8, ha="right", va="top",
          color="#545450",
          bbox=dict(fc="white", ec="#C4C4C0", lw=0.4, pad=1.5))
_ifaceW = load("19_iface_wilcoxon.json")
axg1.annotate((f"PACo vs RhoDesign: Δ = +{100*(_ifaceW['paco_mean']-_ifaceW['rho_mean']):.1f} pp, "
               f"P = {_ifaceW['p']:.2f} (n.s.)\n"
               "(paired Wilcoxon, 110 shared interface chains)"),
              xy=(0.5, 0.10), xycoords="axes fraction", fontsize=4.8,
              ha="center", color="#2E5F8A")

# ============ h 置信度-恢复率(3 子图) ============
axh1 = fig.add_subplot(gs[3, 0:2])
panel_label(axh1, "h", x=-0.24, y=1.1)
hb = axh1.hexbin(conf_x, 100 * conf_y, gridsize=26, cmap="Blues", mincnt=1,
                 linewidths=0.1)
axh1.text(0.05, 0.95, f"ρ = {rho_chain:.3f}\n95% CI [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]",
          transform=axh1.transAxes, va="top", fontsize=5.4, fontweight="bold",
          color="#2E5F8A")
axh1.set_xlabel("Chain-mean log-probability (300 designs)")
axh1.set_ylabel("Recovery, t = 0.1 mean (%)")
axcb = fig.colorbar(hb, ax=axh1, shrink=0.78, pad=0.02)
axcb.ax.tick_params(labelsize=4.5); axcb.outline.set_linewidth(0.4)

axh2 = fig.add_subplot(gs[3, 2:4])
vp = axh2.violinplot([within], widths=0.75, showextrema=False)
for b in vp["bodies"]:
    b.set_facecolor("#DCE6EF"); b.set_edgecolor(PACO); b.set_alpha(0.9)
axh2.scatter(rng.normal(0, 0.035, len(within)), within, s=2, c=PACO, alpha=0.35,
             lw=0)
axh2.axhline(float(np.mean(within)), color="black", lw=0.9)
axh2.axhline(0, color="#B07060", lw=0.5, ls=":")
axh2.text(0.03, 0.95, (f"mean ρ = {np.mean(within):.3f}\n"
          f"{100*p04['within_chain_spearman']['frac_positive']:.0f}% of chains ρ > 0"),
          transform=axh2.transAxes, va="top", fontsize=5.2)
axh2.set_xticks([]); axh2.set_xlim(-0.6, 0.6)
axh2.set_ylabel("Within-chain Spearman(logP, recovery)")

axh3 = fig.add_subplot(gs[3, 4:5])
er10 = p04["logp_rank_enrichment"]["top10pct_minus_pool_mean"]
er1 = p04["logp_rank_enrichment"]["top1_minus_pool_mean"]
enr = [100 * er10["mean"], 100 * er1["mean"]]
eci = [100 * (er10["mean"] - er10["ci95"][0]), 100 * (er1["mean"] - er1["ci95"][0])]
axh3.bar([0, 1], enr, 0.55, yerr=eci, color=[PACO, "#2E5F8A"],
         error_kw=dict(lw=0.5, capsize=1.5), ec="white")
for x, v in zip([0, 1], enr):
    axh3.text(x, v + 0.14, f"+{v:.2f}", ha="center", fontsize=5.4,
              fontweight="bold")
axh3.axhline(0, color="#545450", lw=0.6)
axh3.set_xticks([0, 1]); axh3.set_xticklabels(["top-10%", "top-1"], fontsize=5.0)
axh3.set_ylabel("Recovery − pool mean (pp)")
axh3.set_xlabel("logP-ranked selection", fontsize=5.4)
axh3.set_ylim(0, 2.0)

# ============ i 长度分层箱线 ============
axi = fig.add_subplot(gs[3, 5:8])
panel_label(axi, "i", x=-0.135, y=1.1)
lbins = [(0, 40, "≤40 nt"), (40, 80, "40–80 nt"), (80, 10000, ">80 nt")]
pos, data, colors = [], [], []
for mi, m in enumerate(METHOD_ORDER):
    for bi, (lo, hi, _) in enumerate(lbins):
        ss = [pc[m][s] for s in stems if lo <= chain_len[s] < hi]
        data.append([100 * v for v in ss]); pos.append(bi + (mi - 2) * 0.16)
        colors.append(METHOD_COLOR[m])
bp = axi.boxplot(data, positions=pos, widths=0.13, whis=1.5, patch_artist=True,
                 showfliers=False, medianprops=dict(color="black", lw=0.7),
                 boxprops=dict(lw=0.4), whiskerprops=dict(lw=0.4),
                 capprops=dict(lw=0.4))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.9)
for mi in range(len(METHOD_ORDER)):
    idx = [3 * mi + bi for bi in range(3)]
    means = [np.mean(data[i]) for i in idx]
    axi.scatter([pos[i] for i in idx], means, s=4, color="black", zorder=5, lw=0)
    axi.plot([pos[i] for i in idx], means, color="black", lw=0.5, alpha=0.5)
axi.set_xticks(range(3)); axi.set_xticklabels([b[2] for b in lbins])
axi.set_ylabel("Recovery, pool-in-best (%)")
axi.set_xlabel("Native chain length")
handles = [Line2D([], [], marker="s", ls="", color=METHOD_COLOR[m], markersize=3.5,
                  label=m) for m in METHOD_ORDER]
axi.legend(handles=handles, ncol=2, loc="lower left", fontsize=4.6,
           handletextpad=0.3, columnspacing=0.8)
axi.text(0.985, 0.97, "dots = method means", transform=axi.transAxes, ha="right",
         va="top", fontsize=4.6, color="#545450")

save(fig, "fig2_main_results")
print("chain-level rho check:", round(rho_chain, 4),
      "| within n =", len(within))
