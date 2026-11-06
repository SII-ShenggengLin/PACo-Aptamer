# -*- coding: utf-8 -*-
"""ED Fig. 4 | Conditioning-input diagnostics(条件输入诊断)。

(a) SS 推理期扰动:zero / shuffle / centroid 相对 standard 的变化
    (best/t01 两口径,误差棒 = bootstrap 95% CI;standard 为参考基准,
    Δ 恒为 0,故不画组,绝对值 74.6/59.7 由图注报告;477 链)。
    轴标签与 SI 图 12A 同款("Secondary-structure input",三组扰动)。
(b) 配体注意力质量:界面位 vs 非界面位的逐链差(pp)分布小提琴,
    发布集成与乱序配体、标准变体单模对照并排;颜色/单位/口径与
    SI 图 12B 一致(PACO/灰玫红/浅石板蓝;P = Wilcoxon 单侧 greater)。
面板内角落说明删除(n、CI、P 口径进图注),Δ 记号不进图。
单独运行:python figures/ed/ed_fig4.py → figures/out/ed_fig4.{pdf,png}
"""
import numpy as np

import ed_common as C
from fig_style import NEG, PACO, PACO_D, panel_label, save

COND_LAB = {"zero": "zeroed", "shuffle": "shuffled", "centroid": "centroid"}


def draw_a(ax):
    panel_label(ax, "a")
    sm = C.SS97["summary"]
    conds = ["zero", "shuffle", "centroid"]
    xg = np.arange(len(conds))
    wg = 0.36
    for i, c in enumerate(conds):
        s = sm[c]
        for j, (metric, off) in enumerate([("best", -wg / 2), ("t01", wg / 2)]):
            d = s[f"delta_{metric}_vs_standard"]
            v = 100 * d["mean"]
            lo, hi = 100 * d["ci95"][0], 100 * d["ci95"][1]
            ax.bar(i + off, v, wg * 0.94, color=PACO if j == 0 else PACO_D,
                   ec="white", lw=0.3)
            ax.errorbar(i + off, v, yerr=[[v - lo], [hi - v]], color="black",
                        lw=0.5, capsize=1.2, ls="none")
            if v < 0:
                ax.text(i + off, lo - 0.07, f"{v:+.2f}", ha="center", va="top",
                        fontsize=5.0)
            else:
                ax.text(i + off, hi + 0.07, f"{v:+.2f}", ha="center",
                        va="bottom", fontsize=5.0)
    ax.axhline(0, color="#7A7672", lw=0.6)
    ax.set_xticks(xg)
    ax.set_xticklabels([COND_LAB[c] for c in conds], fontsize=5.2)
    ax.set_xlabel("Secondary-structure input")
    ax.set_ylabel("Change in recovery\nrelative to standard (%)")
    ax.set_ylim(-1.15, 0.80)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=PACO, label="best sample"),
                       Patch(fc=PACO_D, label="t = 0.1 mean")],
              loc="lower left", fontsize=5.0, borderaxespad=0.15,
              handlelength=1.2, labelspacing=0.25)


def draw_b(ax, rng):
    panel_label(ax, "b")
    pc = C.ATT96["per_chain"]
    sm = C.ATT96["summary"]
    groups = [("delta_iface_minus_noniface", "ens2_delta_iface_minus_noniface",
               "ensemble", PACO),
              ("delta_shuffle", "shuffle_delta", "shuffled ligand\n(control)",
               NEG),
              ("delta_seed43", "seed43_delta", "single model\n(control)",
               PACO_D)]
    dvs = [np.array([v[k] for v in pc.values()]) * 100 for k, _, _, _ in groups]
    xg = np.arange(3)
    vp = ax.violinplot(dvs, positions=xg, widths=0.58, showextrema=False)
    for b, (_, _, _, col) in zip(vp["bodies"], groups):
        b.set_facecolor(col)
        b.set_edgecolor("#7A7672")
        b.set_alpha(0.38)
        b.set_lw(0.5)
    for i, (dv, (_, sk, lab, col)) in enumerate(zip(dvs, groups)):
        ax.scatter(rng.normal(i, 0.05, len(dv)), dv, s=2.2, c=col, alpha=0.45,
                   lw=0, zorder=3)
        mu = dv.mean()
        ax.plot([i - 0.29, i + 0.29], [mu, mu], color="black", lw=0.9, zorder=4)
        p = sm[sk]["wilcoxon_p_greater"]
        ax.text(i, dv.max() + 0.05, f"mean {mu:+.3f}\n{C.pfmt_sci(p)}",
                ha="center", va="bottom", fontsize=5.0)
    ax.axhline(0, color="#7A7672", lw=0.6, ls="--")
    ax.set_xticks(xg)
    ax.set_xticklabels([lab for _, _, lab, _ in groups], fontsize=5.2)
    ax.set_ylabel("Attention on ligand tokens\n"
                  "interface − non-interface (pp)")
    ax.set_ylim(-4.6, max(d.max() for d in dvs) + 3.4)


def main():
    fig, gs = C.make_canvas(66, 1, 2, width_ratios=[1.0, 1.08],
                            wspace=0.32, left=0.065, right=0.99,
                            top=0.86, bottom=0.21)
    rng = np.random.default_rng(0)
    draw_a(fig.add_subplot(gs[0, 0]))
    draw_b(fig.add_subplot(gs[0, 1]), rng)
    save(fig, "ed_fig4")


if __name__ == "__main__":
    main()
