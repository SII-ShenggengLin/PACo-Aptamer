# -*- coding: utf-8 -*-
"""Fig 3i | 预测骨架 × 零样本配体交叉切片(105 号,zs 32 链 vs 非 zs 161 链)。

PACo-Aptamer − RhoDesign 逐链优势的组均值 + 链级 bootstrap 95% CI,
best/t01 双口径(填充/半透明,与 d 同款编码):zs 组 best 优势保持
(+6.7%, P = 4.8 × 10⁻⁶)而 t = 0.1 优势消失(−0.1%, P = 0.42),
非 zs 组两口径均显著 —— 正文"边界结果"升入主图(原 ED Fig. 3b,
ED 图保留逐链 Δ 细节)。面板内不写 P(数字由图注承载)。
单独运行:python fig3_i.py → out/fig3_i.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np
from matplotlib.lines import Line2D

import fig3_common as C
from fig_style import PACO, panel_label, save

SUBS = [("zs_subset", "zero-shot ligands\n(32 chains)"),
        ("non_zs_subset", "non-zero-shot ligands\n(161 chains)")]


def draw(fig, gs):
    ax = fig.add_subplot(gs[2, 4:6])
    panel_label(ax, "i")
    for i, (key, _) in enumerate(SUBS):
        d = C.ED.ZS105[key]["delta_PACo_minus_RhoDesign"]
        for j, metric in enumerate(("best", "t01")):
            dd = d[metric]
            v = 100 * dd["mean"]
            lo, hi = 100 * dd["ci95"][0], 100 * dd["ci95"][1]
            xx = i + (j - 0.5) * 0.34
            ax.errorbar(xx, v, yerr=[[v - lo], [hi - v]], marker="o",
                        ms=4.4, mfc=PACO, mec="white", mew=0.4,
                        ecolor="black", elinewidth=0.6, capsize=1.4,
                        capthick=0.6, alpha=1.0 if j == 0 else 0.45,
                        ls="none", zorder=3)
            ax.text(xx, hi + 0.35, f"{v:+.1f}", ha="center", va="bottom",
                    fontsize=4.8)
    ax.axhline(0, color="#7A7672", lw=0.6, ls=":")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.4, markerfacecolor="0.25",
               markeredgecolor="none",
               label="best sample per chain (filled)"),
        Line2D([], [], marker="o", ls="none", ms=4.4, markerfacecolor="0.25",
               markeredgecolor="none", alpha=0.45,
               label="t = 0.1 mean (translucent)")],
        loc="lower right", fontsize=5.5, handlelength=1.2,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([lab for _, lab in SUBS])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-5.5, 11.5)
    ax.set_yticks([-5, 0, 5, 10])
    # 单行 y 标签(正文短语):两行长标签旋转后横向占 8.3 mm,会把 h 的轴线顶穿
    ax.set_ylabel("Advantage over RhoDesign (%)")
    C.tighten_ylabels(fig)
    return ax


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_i")
