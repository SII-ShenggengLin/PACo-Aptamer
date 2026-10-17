# -*- coding: utf-8 -*-
"""SI Fig. | Conditioning-channel diagnostics(二级结构扰动 + 配体注意力)。

(a) 二级结构输入扰动(477 链,TACo):zero / shuffle / centroid 相对 standard
    的 Δ recovery(best 与 t=0.1 两口径,pp),误差棒为 bootstrap 95% CI。
    数据源 data/survey/97_ss_robust.json summary(逐值断言:
    zero −0.28/−0.46pp、shuffle −0.26/−0.59pp、centroid +0.17/+0.00pp)。
(b) 配体 token 注意力质量(115 条界面链):界面 vs 非界面逐链差
    (ens2),shuffled-ligand 与单模型(seed43)对照;阴性(Wilcoxon
    P = 0.88 / 0.89 / ≈1.0)。数据源 data/survey/96_attention.json。
单独运行:python figures/si/si_ss_attention.py → figures/out/si/si_ss_attention.{pdf,png}
"""
import numpy as np

import si_common as C
from fig_style import panel_label

EXPECT = {"zero": (-0.2753, -0.4559), "shuffle": (-0.2627, -0.5942),
          "centroid": (0.1668, 0.0024)}


def draw_a(ax):
    panel_label(ax, "a", x=-0.16, y=1.08)
    d = C.load("97_ss_robust.json")["summary"]
    assert d["standard"]["n"] == 477
    conds = ["zero", "shuffle", "centroid"]
    xg = np.arange(3); wg = 0.36
    for j, (key, lab, col) in enumerate([("best", "best sample", C.TACO),
                                         ("t01", "t = 0.1 mean", C.TACO_D)]):
        vals, los, his = [], [], []
        for c in conds:
            dd = d[c][f"delta_{key}_vs_standard"]
            vals.append(100 * dd["mean"])
            los.append(100 * (dd["mean"] - dd["ci95"][0]))
            his.append(100 * (dd["ci95"][1] - dd["mean"]))
        assert all(abs(vals[i] - EXPECT[c][j]) < 0.02
                   for i, c in enumerate(conds)), (key, vals)
        ax.bar(xg + (j - 0.5) * wg, vals, wg * 0.9, color=col, ec="white",
               lw=0.3, label=lab, yerr=[los, his],
               error_kw=dict(lw=0.6, capsize=1.5), zorder=3)
        for xi, v, lo, hi in zip(xg + (j - 0.5) * wg, vals, los, his):
            ax.text(xi, v - lo - 0.06 if v < 0 else v + hi + 0.06,
                    f"{v:+.2f}", ha="center",
                    va="top" if v < 0 else "bottom", fontsize=4.8)
    ax.axhline(0, color="#545450", lw=0.6)
    ax.set_xticks(xg)
    ax.set_xticklabels(["zeroed SS", "shuffled SS", "centroid SS"],
                       fontsize=5.4)
    ax.set_ylabel("Δ recovery vs standard (pp)")
    ax.set_ylim(-1.05, 0.75)
    ax.legend(loc="lower left", fontsize=4.8, borderaxespad=0.1)
    ax.text(0.98, 0.03, "n = 477 chains; error bars = bootstrap 95% CI",
            transform=ax.transAxes, fontsize=4.4, ha="right", color="#777770")


def draw_b(ax):
    panel_label(ax, "b", x=-0.16, y=1.08)
    d = C.load("96_attention.json")
    pc = d["per_chain"]; sm = d["summary"]
    assert sm["n_chains"] == 115
    series = [("ens2\n(two-model)", "delta_iface_minus_noniface",
               "ens2_delta_iface_minus_noniface", C.TACO),
              ("shuffled\nligand ctrl", "delta_shuffle", "shuffle_delta",
               C.AUX_SAND),
              ("single model\n(seed43 ctrl)", "delta_seed43", "seed43_delta",
               C.TACO_D)]
    rng = np.random.default_rng(3)
    for i, (lab, key, skey, col) in enumerate(series):
        v = np.array([pc[k][key] for k in pc]) * 100
        s = sm[skey]
        assert abs(v.mean() - 100 * s["mean"]) < 1e-9
        vp = ax.violinplot(v, positions=[i], widths=0.58, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col); b.set_alpha(0.45); b.set_lw(0)
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=2.2, color=col,
                   alpha=0.35, lw=0)
        ax.plot([i - 0.24, i + 0.24], [v.mean(), v.mean()], color="black",
                lw=1.0, zorder=5)
        ax.text(i, 1.32, f"mean {100 * s['mean']:+.3f}\n"
                         f"{C.pfmt_sci(s['wilcoxon_p_greater'])}",
                ha="center", va="bottom", fontsize=4.6, color="#2E5F8A")
    ax.axhline(0, color="#545450", lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[0] for s in series], fontsize=5.2)
    ax.set_ylabel("Ligand-token attention mass\ninterface − non-interface (pp)")
    ax.set_ylim(-1.5, 2.6)
    ax.text(0.02, 0.03, "n = 115 interface chains; P = one-sided Wilcoxon "
            "(H1: interface > non-interface)", transform=ax.transAxes,
            fontsize=4.4, color="#777770")


def main():
    fig, gs = C.make_canvas(62, 1, 2, width_ratios=[1.0, 1.0],
                            wspace=0.36, left=0.085, right=0.99,
                            top=0.88, bottom=0.17)
    draw_a(fig.add_subplot(gs[0, 0]))
    draw_b(fig.add_subplot(gs[0, 1]))
    C.save_si(fig, "ss_attention")


if __name__ == "__main__":
    main()
