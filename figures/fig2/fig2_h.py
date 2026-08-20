# -*- coding: utf-8 -*-
"""Fig 2h | 置信度–恢复率耦合(三联):链级 hexbin / 链内 ρ 小提琴 / logP 排序富集。

h1 链级 Spearman + bootstrap 95% CI;h2 链内 Spearman 分布(84% 为正);
h3 top-10% / top-1 logP 排序选择富集(pp)。hexbin 用莫兰迪蓝渐变 taco_blues。
单独运行:python fig2_h.py → out/fig2_h.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig2_common as C
from fig_style import TACO, panel_label, save


def draw(fig, gs):
    rng = np.random.default_rng(42)

    axh1 = fig.add_subplot(gs[3, 0:2])
    panel_label(axh1, "h", x=-0.24, y=1.1)
    hb = axh1.hexbin(C.conf_x, 100 * C.conf_y, gridsize=26, cmap="taco_blues",
                     mincnt=1, linewidths=0.1)
    axh1.text(0.05, 0.95,
              f"ρ = {C.rho_chain:.3f}\n95% CI [{C.rho_ci[0]:.3f}, {C.rho_ci[1]:.3f}]",
              transform=axh1.transAxes, va="top", fontsize=5.4,
              fontweight="bold", color="#2E5F8A")
    axh1.set_xlabel("Chain-mean log-probability (300 designs)")
    axh1.set_ylabel("Recovery, t = 0.1 mean (%)")
    axcb = fig.colorbar(hb, ax=axh1, shrink=0.78, pad=0.02)
    axcb.ax.tick_params(labelsize=4.5); axcb.outline.set_linewidth(0.4)

    axh2 = fig.add_subplot(gs[3, 2:4])
    vp = axh2.violinplot([C.within], widths=0.75, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor("#DCE6EF"); b.set_edgecolor(TACO); b.set_alpha(0.9)
    axh2.scatter(rng.normal(0, 0.035, len(C.within)), C.within, s=2, c=TACO,
                 alpha=0.35, lw=0)
    axh2.axhline(float(np.mean(C.within)), color="black", lw=0.9)
    axh2.axhline(0, color="#B07060", lw=0.5, ls=":")
    axh2.text(0.03, 0.95,
              (f"mean ρ = {np.mean(C.within):.3f}\n"
               f"{100*C.p04['within_chain_spearman']['frac_positive']:.0f}% of chains ρ > 0"),
              transform=axh2.transAxes, va="top", fontsize=5.2)
    axh2.set_xticks([]); axh2.set_xlim(-0.6, 0.6)
    axh2.set_ylabel("Within-chain Spearman(logP, recovery)")

    axh3 = fig.add_subplot(gs[3, 4:5])
    er10 = C.p04["logp_rank_enrichment"]["top10pct_minus_pool_mean"]
    er1 = C.p04["logp_rank_enrichment"]["top1_minus_pool_mean"]
    enr = [100 * er10["mean"], 100 * er1["mean"]]
    eci = [100 * (er10["mean"] - er10["ci95"][0]),
           100 * (er1["mean"] - er1["ci95"][0])]
    axh3.bar([0, 1], enr, 0.55, yerr=eci, color=[TACO, "#2E5F8A"],
             error_kw=dict(lw=0.5, capsize=1.5), ec="white")
    for x, v in zip([0, 1], enr):
        axh3.text(x, v + 0.14, f"+{v:.2f}", ha="center", fontsize=5.4,
                  fontweight="bold")
    axh3.axhline(0, color="#545450", lw=0.6)
    axh3.set_xticks([0, 1]); axh3.set_xticklabels(["top-10%", "top-1"],
                                                  fontsize=5.0)
    axh3.set_ylabel("Recovery − pool mean (pp)")
    axh3.set_xlabel("logP-ranked selection", fontsize=5.4)
    axh3.set_ylim(0, 2.0)
    return axh1, axh2, axh3


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_h")
    print("chain-level rho check:", round(C.rho_chain, 4),
          "| within n =", len(C.within))
