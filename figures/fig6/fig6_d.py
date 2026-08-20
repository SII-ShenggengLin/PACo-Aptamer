# -*- coding: utf-8 -*-
"""Fig 6d | 回顾性审计(方案 A):适配体 vs 结构同条件随机对照的 ΔlogP 双库分布
+ 家族内 logP–Kd 相关森林图。数据:79/88 号 survey JSON(经 fig6_common 预处理)。
单独运行:python fig6_d.py → out/fig6_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig6_common as C
from fig_style import TACO_D, panel_label, save


def draw(fig, gs):
    from matplotlib import gridspec
    gsd = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 3:9], width_ratios=[2.2, 1.0], wspace=0.85)
    axd = fig.add_subplot(gsd[0, 0])
    panel_label(axd, "d", x=-0.155, y=1.12)
    bins = np.linspace(min(-0.5, C.D_ALL.min() - 0.2), max(4.5, C.D_ALL.max() + 0.2), 46)
    axd.hist(C.D_APT, bins=bins, color="#A8A8A4", alpha=0.85, ec="white",
             lw=0.15, label=f"AptaDB clean (n = {len(C.D_APT)})")
    axd.hist(C.D_UTX, bins=bins, color="#D8A890", alpha=0.75, ec="white",
             lw=0.15, label=f"UTexas clean (n = {len(C.D_UTX)})")
    axd.axvline(0, color="#545450", lw=0.7)
    axd.axvline(np.mean(C.D_ALL), color="#B07060", lw=0.9, ls="--")
    axd.text(0.975, 0.62,
             (f"logP(aptamer) − logP(random | same structure & SS)\n"
              f"merged n = {len(C.D_ALL)}: mean {np.mean(C.D_ALL):+.2f},\n"
              f"{100*np.mean(C.D_ALL > 0):.1f}% of interactions above random"),
             transform=axd.transAxes, ha="right", va="top", fontsize=4.9,
             bbox=dict(fc="white", ec="#C4C4C0", lw=0.4, pad=1.5))
    axd.legend(fontsize=4.9, loc="upper left")
    axd.set_xlabel("Δ log-probability per interaction")
    axd.set_ylabel("Interactions")

    axd2 = fig.add_subplot(gsd[0, 1])
    order = np.argsort(C.FAM_RHO)
    for i, oi in enumerate(order):
        col = "#8C8C88" if C.FAM_RHO[oi] >= 0 else "#B07060"
        axd2.plot([min(C.FAM_RHO[oi], 0), max(C.FAM_RHO[oi], 0)], [i, i],
                  color=col, lw=1.4, solid_capstyle="round")
        axd2.scatter([C.FAM_RHO[oi]], [i], s=3 + 2.2 * C.FAM_N[oi], c=col, lw=0)
    axd2.axvline(0, color="#545450", lw=0.6)
    axd2.axvline(np.mean(C.FAM_RHO), color=TACO_D, lw=0.8, ls=":")
    axd2.text(0.97, 1.04,
              f"within-family ρ(logP, −log$_{{10}}$K$_d$)\n"
              f"{len(C.FAM_RHO)} families, mean {np.mean(C.FAM_RHO):+.2f} (n.s.)",
              transform=axd2.transAxes, ha="right", va="bottom", fontsize=4.8)
    axd2.set_yticks([])
    axd2.set_xlabel("Spearman ρ")
    axd2.set_xlim(-1.05, 1.05)
    return axd, axd2


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig6_d")
