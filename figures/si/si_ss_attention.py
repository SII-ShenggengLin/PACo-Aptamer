# -*- coding: utf-8 -*-
"""SI Fig. | Conditioning-input diagnostics(二级结构扰动 + 配体注意力)。

(a) 二级结构输入扰动(477 链,PACo-Aptamer):zero / shuffle / centroid 相对
    standard 的 Δ recovery(best 与 t=0.1 两口径,pp),误差棒为 bootstrap
    95% CI。数据源 data/survey/97_ss_robust.json summary(逐值断言:
    zero −0.28/−0.46pp、shuffle −0.26/−0.59pp、centroid +0.17/+0.00pp)。
(b) 配体 token 注意力(115 条界面链,decoder cross-attention 质量,
    analysis/23 任务 B 口径):界面 − 非界面逐链差(发布集成),shuffled-
    ligand 与标准变体单模型对照;阴性(Wilcoxon 单侧 P = 0.88 / 0.89 /
    ≈1.0)。数据源 data/survey/96_attention.json。
单独运行:python figures/si/si_ss_attention.py → figures/out/si/si_ss_attention.{pdf,png}
"""
import numpy as np

import si_common as C
from fig_style import panel_label

EXPECT = {"zero": (-0.2753, -0.4559), "shuffle": (-0.2627, -0.5942),
          "centroid": (0.1668, 0.0024)}


def draw_a(ax):
    panel_label(ax, "a")
    d = C.load("97_ss_robust.json")["summary"]
    assert d["standard"]["n"] == 477
    conds = ["zero", "shuffle", "centroid"]
    xg = np.arange(3); wg = 0.36
    for j, (key, lab, col) in enumerate([("best", "best sample", C.PACO),
                                         ("t01", "t = 0.1 mean", C.PACO_D)]):
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
                    va="top" if v < 0 else "bottom", fontsize=5.0)
    ax.axhline(0, color="#7A7672", lw=0.6)
    ax.set_xticks(xg)
    ax.set_xticklabels(["zeroed", "shuffled", "centroid"], fontsize=5.6)
    ax.set_xlabel("Secondary-structure input")
    ax.set_ylabel("Δ recovery relative to standard (pp)")
    ax.set_ylim(-1.05, 0.75)
    ax.tick_params(axis="y", labelsize=5.6)
    ax.legend(loc="lower right", fontsize=5.2, borderaxespad=0.2)


def draw_b(ax):
    panel_label(ax, "b")
    d = C.load("96_attention.json")
    pc = d["per_chain"]; sm = d["summary"]
    assert sm["n_chains"] == 115
    series = [("published\nensemble", "delta_iface_minus_noniface",
               "ens2_delta_iface_minus_noniface", C.PACO),
              ("shuffled ligand\n(control)", "delta_shuffle", "shuffle_delta",
               C.NEG),
              ("standard variant\n(control)", "delta_seed43", "seed43_delta",
               C.PACO_D)]
    rng = np.random.default_rng(3)
    all_max = max(max(np.array([pc[k][s[1]] for k in pc]) * 100)
                  for s in series)
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
        ax.text(i, v.max() + 0.06, f"mean {100 * s['mean']:+.3f}\n"
                 f"{C.pfmt_sci(s['wilcoxon_p_greater'])}",
                ha="center", va="bottom", fontsize=4.8)
    ax.axhline(0, color="#7A7672", lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[0] for s in series], fontsize=5.2)
    ax.set_ylabel("Attention on ligand tokens\n"
                  "interface − non-interface (pp)")
    ax.set_ylim(-1.8, all_max + 0.5)
    ax.tick_params(axis="y", labelsize=5.6)


def main():
    fig, gs = C.make_canvas(62, 1, 2, width_ratios=[1.0, 1.0],
                            wspace=0.30, left=0.085, right=0.99,
                            top=0.88, bottom=0.20)
    draw_a(fig.add_subplot(gs[0, 0]))
    draw_b(fig.add_subplot(gs[0, 1]))
    C.save_si(fig, "ss_attention")


if __name__ == "__main__":
    main()
