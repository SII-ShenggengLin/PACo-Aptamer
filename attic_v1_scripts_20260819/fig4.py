# -*- coding: utf-8 -*-
"""Fig 4 | 配体条件化的价值("从折叠到结合"核心证据)。

a 消融双口径表(全局 best / 界面 best / Δ(界面-整体) / ≥3残基子集 / Wilcoxon p)
b 每核苷酸恢复率 vs 距配体距离(绝对 + 归一化)
c SS 条件保真 INF(多裁判框架:native/random 边界 + 相似度分箱机制)
f 自洽结构 scTM(native 上界 + 5 方法 + recovery-matched 负对照)
g recovery-matched 负对照逐链 Δ 分布
数据:survey/{74_p00_iface_ablation, 77_p02_distance_gradient,
77_p03_ss_fidelity, 86_sctm}.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (  # noqa: E402
    METHOD_COLOR, MM, NATIVE, NEG, TACO, apply_style, load, panel_label, save)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

# ---------------- 数据 ----------------
p00 = load("74_p00_iface_ablation.json")["phase1_ablation"]
p02 = load("77_p02_distance_gradient.json")["methods"]
p03 = load("77_p03_ss_fidelity.json")["methods"]
sctm = load("86_sctm.json")

BIN_NAMES = ["≤5Å(界面)", "5-8Å", "8-12Å", "12-20Å", ">20Å"]
BIN_LAB = ["≤5", "5–8", "8–12", "12–20", ">20"]
BIN_X = [2.5, 6.5, 10, 16, 25]

MKEY = [("TACo-Aptamer", "TACo_ens2"), ("RhoDesign", "RhoDesign"),
        ("AlignIF", "AlignIF"), ("NA-MPNN", "NA-MPNN"), ("gRNAde", "gRNAde")]

# ---------------- 画布 ----------------
fig = plt.figure(figsize=(183 * MM, 128 * MM))
gs = gridspec.GridSpec(2, 8, figure=fig, hspace=0.55, wspace=1.15,
                       height_ratios=[1.05, 1.15])

# ============ a 消融双口径表 ============
axa = fig.add_subplot(gs[0, 0:3]); axa.axis("off")
panel_label(axa, "a", x=-0.062, y=1.06)
rows_abl = [("Full model (ens2)", "baseline"), ("– ensemble (single)", "no_ensemble"),
            ("– SS conditioning", "no_ss"), ("– data augmentation", "no_aug"),
            ("– ligand context", "no_ligand")]
hdr = ["Model variant", "Global\nbest (%)", "Interface\nbest (%)",
       "Δ (iface–\noverall, pp)", "Iface Δ vs full\n(pp, P)"]
cell = []
for lab, key in rows_abl:
    r = p00[key]
    d_glob = 100 * (r["global_best_477"] - p00["baseline"]["global_best_477"])
    iface_b = 100 * r["iface"]["iface_best"]
    delta_io = 100 * r["iface"]["iface_minus_overall"]
    ge3 = 100 * r["iface_ge3"]["iface_minus_overall"]
    if key == "baseline":
        pcell = "—"
    else:
        vb = r["vs_baseline_iface"]
        # vb.delta_iface_best = baseline − variant(界面口径)
        dvar = -100 * vb["delta_iface_best"]
        pcell = f"{dvar:+.2f} (P={vb['wilcoxon_p_one_sided']:.3f})" \
            if vb["wilcoxon_p_one_sided"] >= 0.001 else \
            f"{dvar:+.2f} (P={vb['wilcoxon_p_one_sided']:.4f})"
    cell.append([lab, f"{100*r['global_best_477']:.2f}", f"{iface_b:.2f}",
                 f"{delta_io:+.2f}", pcell])
tb = axa.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(4.9); tb.scale(1, 1.62)
for (i, j), c in tb.get_celld().items():
    c.set_linewidth(0.4); c.set_edgecolor("#C4C4C0")
    if i == 0:
        c.set_facecolor("#EFEDEA"); c.set_text_props(fontweight="bold")
    elif i == 1:
        c.set_facecolor("#EDF2F7")
axa.text(0.5, 0.010,
         "300-sample pools (3 T × 100) · final noiface recipe · global = test 477 chains\n"
         "interface = 115 ligand-contact chains (<5 Å; 100 with ≥3 interface residues);\n"
         "rows 2–5 are single models — within that family, component removals cost\n"
         "0.6–0.8 pp globally (interface effects n.s. at this pool size)",
         transform=axa.transAxes, ha="center", fontsize=4.5, color="#545450")

# ============ b 距离梯度(双子图) ============
gsb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 3:8],
                                       wspace=1.0)
for sub, (mode, ylab, letter_off) in enumerate([
        ("abs", "Recovery (%)", ""), ("norm", "Δ recovery vs chain mean (pp)", "")]):
    axb = fig.add_subplot(gsb[0, sub])
    if sub == 0:
        panel_label(axb, "b", x=-0.115, y=1.12)
    for mlab, mkey in MKEY:
        vals, lo, hi = [], [], []
        for bn in BIN_NAMES:
            d = p02[mkey][bn]
            vals.append(100 * d["mean"])
            lo.append(100 * (d["mean"] - d["ci95"][0]))
            hi.append(100 * (d["ci95"][1] - d["mean"]))
        if mode == "norm":
            vals = [100 * p02[mkey][bn]["norm_mean"] for bn in BIN_NAMES]
            lo = [100 * (p02[mkey][bn]["norm_mean"] - p02[mkey][bn]["norm_ci95"][0])
                  for bn in BIN_NAMES]
            hi = [100 * (p02[mkey][bn]["norm_ci95"][1] - p02[mkey][bn]["norm_mean"])
                  for bn in BIN_NAMES]
        axb.errorbar(BIN_X, vals, yerr=[lo, hi], marker="o", ms=2.4,
                     lw=0.9, capsize=1.2, elinewidth=0.5,
                     color=METHOD_COLOR[mlab],
                     label=mlab if sub == 0 else None)
    axb.axvspan(0, 5, color="#E8A088", alpha=0.14, lw=0)
    if sub == 0:
        axb.text(2.5, 88, "pocket", fontsize=4.8, ha="center", color="#B0755F")
        axb.set_ylim(0, 92)
    else:
        axb.axhline(0, color="#545450", lw=0.5)
        axb.set_ylim(-20, 14)
        axb.text(2.5, 11.5, "pocket", fontsize=4.8, ha="center", color="#B0755F")
    axb.set_xticks(BIN_X); axb.set_xticklabels(BIN_LAB)
    axb.set_xlabel("Distance to nearest ligand atom (Å)")
    axb.set_ylabel(ylab)
    if sub == 0:
        axb.legend(fontsize=4.9, loc="lower left", handlelength=1.2,
                   labelspacing=0.25, ncol=2, columnspacing=0.8,
                   frameon=True, facecolor="white", framealpha=0.85,
                   edgecolor="none")   # gRNAde 低曲线穿图例区 → 白底衬
    if sub == 1:
        axb.text(0.97, 0.05, "absolute (left) · normalized (right)\nmethod level removed",
                 transform=axb.transAxes, fontsize=4.4, color="#545450",
                 ha="right", va="bottom")

# ============ c SS 保真 ============
gsc = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 0:4],
                                       wspace=2.2)
axc = fig.add_subplot(gsc[0, 0:3])
panel_label(axc, "c", x=-0.16, y=1.12)
crows = [("native", "native", NATIVE), ("gRNAde", "gRNAde", METHOD_COLOR["gRNAde"]),
         ("NA-MPNN", "NA-MPNN", METHOD_COLOR["NA-MPNN"]),
         ("AlignIF", "AlignIF", METHOD_COLOR["AlignIF"]),
         ("TACo-Aptamer", "TACo_ens2", METHOD_COLOR["TACo-Aptamer"]),
         ("RhoDesign", "RhoDesign", METHOD_COLOR["RhoDesign"]),
         ("random", "random", NEG)]
names, vals, cis, cols = [], [], [], []
for lab, key, col in crows:
    d = p03[key]["inf"]
    names.append(lab); vals.append(d["mean"])
    cis.append(d["mean"] - d["ci95"][0]); cols.append(col)
xc = np.arange(len(names))
axc.bar(xc, vals, 0.62, yerr=cis, color=cols,
        error_kw=dict(lw=0.5, capsize=1.3), ec="white", lw=0.3)
for x0, v in zip(xc, vals):
    axc.text(x0, v + 0.03, f"{v:.3f}", ha="center", fontsize=4.8,
             fontweight="bold" if names[int(x0)] in ("native", "random") else "normal")
axc.set_xticks(xc); axc.set_xticklabels(
    [n.replace("-Aptamer", "") for n in names], rotation=32, ha="right")
axc.set_ylabel("SS-fidelity INF (input SS vs refolded design)")
axc.set_ylim(0, 1.14)
axc.text(0.985, 0.955, "upper bound: native = 1.0\nlower bound: random = 0.03",
         transform=axc.transAxes, ha="right", va="top", fontsize=4.8,
         color="#545450")

axc2 = fig.add_subplot(gsc[0, 3])
BIN_KEYS = ["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"]
BIN_POS = [0.2, 0.45, 0.55, 0.65, 0.75, 0.9]
for mlab, mkey in MKEY:
    d = p03[mkey]["inf_by_identity"]
    ys, xs = [], []
    for bk, bp in zip(BIN_KEYS, BIN_POS):
        if bk in d:
            xs.append(bp); ys.append(d[bk]["inf_mean"])
    axc2.plot(xs, ys, marker="o", ms=2.2, lw=0.8,
              color=METHOD_COLOR[mlab])
axc2.set_xlabel("Design identity to native")
axc2.set_ylabel("INF")
axc2.set_xlim(0.05, 1.0); axc2.set_ylim(0, 1.02)
axc2.text(0.08, 0.06, "compensatory mutations:\nhigh INF at low identity",
          fontsize=4.4, color="#545450", transform=axc2.transAxes)

# ============ f scTM ============
axf = fig.add_subplot(gs[1, 4:6])
panel_label(axf, "f", x=-0.28, y=1.12)
frows = [("native", "native", NATIVE), ("NA-MPNN", "NA-MPNN", METHOD_COLOR["NA-MPNN"]),
         ("TACo-Aptamer", "TACo", METHOD_COLOR["TACo-Aptamer"]),
         ("RhoDesign", "RhoDesign", METHOD_COLOR["RhoDesign"]),
         ("random-matched", "randmatch", NEG),
         ("gRNAde", "gRNAde", METHOD_COLOR["gRNAde"])]
rng = np.random.default_rng(7)
for i, (lab, key, col) in enumerate(frows):
    vals = np.array([v[f"sctm_{key}"] for v in sctm["per_chain"].values()
                     if f"sctm_{key}" in v])
    yp = len(frows) - 1 - i
    bp = axf.boxplot([vals], positions=[yp], vert=False, widths=0.55,
                     whis=1.5, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=0.7),
                     boxprops=dict(lw=0.4), whiskerprops=dict(lw=0.4),
                     capprops=dict(lw=0.4))
    bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.85)
    axf.scatter([np.mean(vals)], [yp], color="black", s=5, lw=0, zorder=5)
    sm = sctm["summary"][key]
    axf.text(1.02, yp, f"{sm['mean']:.3f} · {100*sm['pass_045']:.0f}%",
             va="center", fontsize=4.7, transform=axf.get_yaxis_transform(),
             fontweight="bold" if key in ("native", "TACo") else "normal")
axf.set_yticks(range(len(frows)))
axf.set_yticklabels([r[0].replace("-Aptamer", "") for r in frows][::-1],
                    fontsize=5.0)
axf.axvline(0.45, color="#8C8C88", lw=0.5, ls=":")
axf.text(0.455, 5.42, "TM = 0.45", fontsize=4.4, color="#8C8C88")
axf.set_xlabel("scTM (RhoFold+ refold, vs native)")
axf.set_xlim(0, 1.0)
axf.text(1.02, 5.48, "mean · %pass@0.45", fontsize=4.4,
         transform=axf.get_yaxis_transform(), color="#545450")

# ============ g recovery-matched 负对照 ============
axg = fig.add_subplot(gs[1, 6:8])
panel_label(axg, "g", x=-0.19, y=1.12)
diff = np.array([v["sctm_TACo"] - v["sctm_randmatch"]
                 for v in sctm["per_chain"].values()
                 if "sctm_TACo" in v and "sctm_randmatch" in v]) * 100
axg.hist(diff, bins=36, color=TACO, alpha=0.85, ec="white", lw=0.2)
axg.axvline(0, color="#545450", lw=0.6)
axg.axvline(float(np.mean(diff)), color="#2E5F8A", lw=0.9, ls="--")
axg.text(0.955, 0.92,
         f"mean Δ = {np.mean(diff):+.1f} pp\n95% CI [2.4, 5.1]\n"
         "P = 4 × 10$^{{-6}}$ (one-sided Wilcoxon)\n\n"
         "recovery-matched random collapses to\ngRNAde level (0.226 vs 0.219)",
         transform=axg.transAxes, ha="right", va="top", fontsize=5.0,
         color="#2E5F8A")
axg.set_xlabel("scTM(TACo) − scTM(recovery-matched random), pp")
axg.set_ylabel("Chains")

save(fig, "fig4_ligand_conditioning")
print("randmatch diff mean:", np.mean(diff).round(2))
