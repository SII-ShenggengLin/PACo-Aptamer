# -*- coding: utf-8 -*-
"""Fig 4f | SS 保真随设计同一性的变化:五方法分箱曲线(独立编号面板)。

2026-10-22 起自成面板(原 d 右侧子图过窄),宽度大幅增加;
图内释义文字框删除(由正文/图注承载),补五方法图例(同 c 色序)。
单独运行:python fig4_f.py → out/fig4_f.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig4_common as C
from fig_style import METHOD_COLOR, panel_label, save

BIN_KEYS = ["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"]
BIN_POS = [0.2, 0.45, 0.55, 0.65, 0.75, 0.9]


def draw(fig, gs):
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "f")
    for mlab, mkey in C.MKEY:
        d = C.P03[mkey]["inf_by_identity"]
        xs, ys = [], []
        for bk, bp in zip(BIN_KEYS, BIN_POS):
            if bk in d:
                xs.append(bp); ys.append(d[bk]["inf_mean"])
        ax.plot(xs, ys, marker="o", ms=2.8, lw=0.9, color=METHOD_COLOR[mlab],
                label=mlab)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Design identity")
    ax.set_ylabel("SS fidelity, INF")
    # 方法色与 c/d 相同且 c 位于正上方,图例只在 c 画(42 mm 面板内
    # 五项两列图例必然侵入 gRNAde 高曲线区)
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_f")
