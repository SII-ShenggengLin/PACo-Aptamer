# -*- coding: utf-8 -*-
"""Fig 6e | 计算筛选预算曲线(INF/logP/共识/oracle/random 五策略 × 5 预算)。

数据:84_consensus_screen.json。单独运行:python fig6_e.py →
out/fig6_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig6_common as C
from fig_style import NEG, TACO, panel_label, save


def draw(fig, gs):
    axe = fig.add_subplot(gs[1, 9:16])
    panel_label(axe, "e", x=-0.075, y=1.12)
    budgets = [0.01, 0.05, 0.10, 0.25, 0.50]
    styles = {"INF": (TACO, "-", "o", "INF ranking"),
              "logP": ("#8C8C88", "-", "s", "logP ranking"),
              "consensus": (NEG, "-", "^", "rank-average consensus"),
              "oracle": ("#3D5A45", "--", "*", "oracle"),
              "random": ("#545450", ":", "", "random")}
    for key, (col, ls, mk, lab) in styles.items():
        xs, ys, lo, hi = [], [], [], []
        for b in budgets:
            c = C.CS[key][f"budget_{b:.2f}"]
            xs.append(100 * b)
            ys.append(100 * c["gain"])
            lo.append(100 * (c["gain"] - c["ci95"][0]))
            hi.append(100 * (c["ci95"][1] - c["gain"]))
        axe.errorbar(xs, ys, yerr=[lo, hi], color=col, ls=ls, marker=mk,
                     ms=3 if mk != "*" else 5, lw=0.9, capsize=1.2,
                     elinewidth=0.5, label=lab)
    axe.axhline(0, color="#545450", lw=0.5)
    axe.set_xscale("log")
    axe.set_xticks([1, 5, 10, 25, 50])
    axe.set_xticklabels(["1", "5", "10", "25", "50"])
    axe.set_xlabel("Screening budget (% of pool)")
    axe.set_ylabel("Recovery − pool mean (pp)")
    axe.legend(fontsize=4.6, loc="upper center", bbox_to_anchor=(0.5, -0.32),
               ncol=3, handletextpad=0.3, columnspacing=0.8, frameon=True,
               facecolor="white", framealpha=0.85, edgecolor="none")
    axe.text(0.03, 0.10,
             "477 chains × 300 designs\nconsensus ranking is counterproductive:\n"
             "logP and INF are anti-correlated",
             transform=axe.transAxes, fontsize=4.5, color="#545450",
             bbox=dict(fc="white", ec="none", pad=1.0))
    return axe


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig6_e")
