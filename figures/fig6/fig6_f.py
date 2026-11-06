# -*- coding: utf-8 -*-
"""Fig 6f | 计算筛选预算曲线(INF/log probability/平均/oracle/random × 5 预算)。

数据:84_consensus_screen.json(477 chains × 300 designs,口径入图题)。
2026-10-23 第二轮:面板加宽(1.74→1.81 in),图例自轴内右上移至左上(与
d 统一);y 上限自 26 抬至 31,使 5 行图例(7.0 pt)的下缘越过 oracle 在
1% 预算的误差棒顶(14.66,程序化审计零相交);字号回归 make_canvas 全局
覆盖(刻度 7.0/轴题 7.5/图例 7.0,不再单独缩小);序列名 logP → log
probability 与全文口径统一(沿用 2026-10-23 第一轮)。
单独运行:python fig6_f.py → out/fig6_f.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig6_common as C
from fig_style import NATIVE, NEG, PACO, save


def draw(fig, L):
    axf = fig.add_axes(L["f"])
    # 编号 f 锚定画廊第 4 列边界(axes 左缘左移吃进 e 让出的宽度后仍对齐)
    C.panel_label_at(fig, "f", C.COLS[3] * C.FIG_W_IN, C.ROW3_TOP_IN)
    budgets = [0.01, 0.05, 0.10, 0.25, 0.50]
    styles = {"INF": (PACO, "-", "o", "INF ranking"),
              "logP": ("#B7B0A2", "-", "s", "log probability ranking"),
              "consensus": (NEG, "-", "^", "average of the two rankings"),
              "oracle": (NATIVE, "--", "*", "oracle"),
              "random": ("#7A7672", ":", "", "random")}
    for key, (col, ls, mk, lab) in styles.items():
        xs, ys, lo, hi = [], [], [], []
        for b in budgets:
            c = C.CS[key][f"budget_{b:.2f}"]
            xs.append(100 * b)
            ys.append(100 * c["gain"])
            lo.append(100 * (c["gain"] - c["ci95"][0]))
            hi.append(100 * (c["ci95"][1] - c["gain"]))
        axf.errorbar(xs, ys, yerr=[lo, hi], color=col, ls=ls, marker=mk,
                     ms=3 if mk != "*" else 5, lw=0.9, capsize=1.2,
                     elinewidth=0.5, label=lab)
    axf.set_xscale("log")
    axf.set_xticks([1, 5, 10, 25, 50])
    axf.set_xticklabels(["1", "5", "10", "25", "50"])
    axf.set_xlim(0.85, 60)
    axf.set_xlabel("Screening budget (% of pool)")
    axf.set_ylabel("Recovery enrichment (%)", labelpad=2)
    axf.set_yticks([0, 10, 20, 30])
    # 上限抬至 31:左上图例(7.0 pt,5 行)下缘 ≈16.3,越过 oracle 误差棒
    # 顶 14.66(1% 预算),其余曲线 ≤2.8,程序化审计零相交
    axf.set_ylim(-3.8, 31.0)
    axf.legend(loc="upper left", frameon=True, facecolor="white",
               framealpha=1.0, edgecolor="none", handlelength=1.5,
               borderaxespad=0.3, labelspacing=0.2, borderpad=0.3)
    return axf


if __name__ == "__main__":
    fig, L = C.make_canvas()
    draw(fig, L)
    save(fig, "fig6_f")
