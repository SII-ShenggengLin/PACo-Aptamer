# -*- coding: utf-8 -*-
"""Fig 4a | 消融哑铃图:每变体一对点(全局 best / 界面 best),灰线相连。

2026-10-22 起由表格改为图(同图类型多样化;表格脚注文字全部删除,
口径说明移入图注)。口径编码沿用全图惯例:实心点 = 全局(477 测试链),
半透明点 = 界面(115 条 <5 Å 界面链);数值标在点旁,不加粗。
x 轴 60–90% 放大,使单模型家族 0.6–0.8% 的全局差异可读。
单独运行:python fig4_a.py → out/fig4_a.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig4_common as C
from fig_style import PACO, panel_label, save

ROWS = [("Full model", "baseline"),
        ("− ensemble", "no_ensemble"),
        ("− SS conditioning", "no_ss"),
        ("− augmentation", "no_aug"),
        ("− ligand context", "no_ligand")]


def draw(fig, gs):
    axa = fig.add_subplot(gs[0, 0])
    panel_label(axa, "a")
    ys = np.arange(len(ROWS))[::-1]          # Full model 在最上
    for y, (_, key) in zip(ys, ROWS):
        r = C.P00[key]
        g = 100 * r["global_best_477"]
        i = 100 * r["iface"]["iface_best"]
        axa.plot([g, i], [y, y], color="#B7B0A2", lw=1.1, zorder=2)
        axa.plot(g, y, "o", ms=4.2, mfc=PACO, mec="white", mew=0.4, zorder=3)
        axa.plot(i, y, "o", ms=4.2, mfc=PACO, mec="white", mew=0.4,
                 alpha=0.45, zorder=3)
        axa.text(g - 0.7, y, f"{g:.1f}", ha="right", va="center", fontsize=5.2)
        axa.text(i + 0.7, y, f"{i:.1f}", ha="left", va="center", fontsize=5.2)
    axa.set_yticks(ys)
    axa.set_yticklabels([lab for lab, _ in ROWS])
    axa.set_xlim(60, 90)
    axa.set_ylim(-0.5, 5.15)
    axa.set_xticks([60, 65, 70, 75, 80, 85, 90])
    axa.set_xlabel("Recovery, best sample per chain (%)")
    # 双行单列图例置于顶部空带(6 pt 下单行双列宽约 52 mm 超出面板)
    axa.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.2, markerfacecolor="0.25",
               markeredgecolor="none", label="global (filled)"),
        Line2D([], [], marker="o", ls="none", ms=4.2, markerfacecolor="0.25",
               markeredgecolor="none", alpha=0.45,
               label="interface (translucent)")],
        loc="upper left", fontsize=6.0, handlelength=1.2,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    return axa


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_a")
