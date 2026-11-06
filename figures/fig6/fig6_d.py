# -*- coding: utf-8 -*-
"""Fig 6d | 适配体 vs 结构同条件随机对照的 Δ log probability 分布(双库)。

数据:79/88 号 survey JSON(经 fig6_common 预处理;AptaDB n=214、UTexas n=228、
合并均值 +1.95,与正文/图题一致)。跨画廊(panel a)第 1–2 列图像,右缘对齐;
虚线合并均值按 v1.68 惯例入图例。2026-10-23 第二轮:图例自右上移至左上。
第四轮:x = 0 的零参考竖线删除 —— 改为 xlim/分箱自 0 起,零由坐标轴左缘
与刻度标示(全部 442 个 Δ>0,柱均落在轴右侧;图题只引用虚线均值线,
零线未入图题),左上图例下不再有任何线条可压。
单独运行:python fig6_d.py → out/fig6_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fig6_common as C
from fig_style import NEG, save


def draw(fig, L):
    axd = fig.add_axes(L["d"])
    C.panel_label_at(fig, "d", C.IMG_X[0], C.ROW3_TOP_IN)
    bins = np.linspace(0.0, 4.0, 41)
    axd.hist(C.D_APT, bins=bins, color="#8FA6BB", alpha=0.85, ec="white",
             lw=0.15)
    axd.hist(C.D_UTX, bins=bins, color="#D9C3A9", alpha=0.75, ec="white",
             lw=0.15)
    axd.axvline(np.mean(C.D_ALL), color=NEG, lw=0.9, ls="--")
    handles = [
        Patch(fc="#8FA6BB", ec="none", label=f"AptaDB (n = {len(C.D_APT)})"),
        Patch(fc="#D9C3A9", ec="none", label=f"UTexas (n = {len(C.D_UTX)})"),
        Line2D([], [], color=NEG, lw=0.9, ls="--",
               label=f"merged mean {np.mean(C.D_ALL):+.2f}"),
    ]
    axd.legend(handles=handles, loc="upper left", frameon=True,
               facecolor="white", framealpha=1.0, edgecolor="none",
               handlelength=1.4, borderaxespad=0.4)
    axd.set_xlabel("\u0394 log probability per interaction")
    axd.set_ylabel("Interactions")
    axd.set_xlim(0.0, 4.0)
    # 第五轮 8 pt 图例变高:ylim 自 ~23 抬到 26,图例底 ~19,越过其下柱顶 17
    axd.set_ylim(0.0, 26.0)
    return axd


if __name__ == "__main__":
    fig, L = C.make_canvas()
    draw(fig, L)
    save(fig, "fig6_d")
