# -*- coding: utf-8 -*-
"""Fig 6e | 家族内 log probability–亲和力相关森林图(独立编号)。

19 个 ≥3 条实测 Kd 的家族,Spearman ρ(log probability,−log10 Kd),均值 +0.11
(n.s.,与正文/图题/SI Note 4 一致);点面积随家族成员数增大;变量对子的
定义留在图题。2026-10-23 第二轮:面板加宽(1.42→1.52 in),框式图例与茎/点
重叠的问题以"均值虚线顶旁直接文字注记"取代图例框(顶部留空带内无任何
数据元素;n.s. 与变量定义由图题承载),文字 7.0 pt 黑色。
单独运行:python fig6_e.py → out/fig6_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig6_common as C
from fig_style import PACO, save


def draw(fig, L):
    axe = fig.add_axes(L["e"])
    # 编号 e 与编号 c 像素对齐(c 的 axes 左缘 = 列边界 + 0.015,其字母
    # 锚在左缘外 1.2 mm);axes 左缘已越过边界吃进 d 让出的宽度
    # (见 fig6_common 行 3 注释),故用 panel_label_at
    C.panel_label_at(fig, "e", C.COLS[2] * C.FIG_W_IN + 0.015, C.ROW3_TOP_IN)
    order = np.argsort(C.FAM_RHO)
    for i, oi in enumerate(order):
        col = "#7A7672" if C.FAM_RHO[oi] >= 0 else "#B5838D"
        axe.plot([min(C.FAM_RHO[oi], 0), max(C.FAM_RHO[oi], 0)], [i, i],
                 color=col, lw=1.3, solid_capstyle="round")
        axe.scatter([C.FAM_RHO[oi]], [i], s=2.5 + 1.1 * C.FAM_N[oi], c=col,
                    lw=0)
    axe.axvline(0, color="#7A7672", lw=0.6)
    axe.axvline(np.mean(C.FAM_RHO), color=PACO, lw=0.9, ls=":")
    # 均值虚线的文字注记:置于顶部留空带、紧贴虚线右侧(茎/点均止于
    # y = n_families − 1,注记带在其上,实测零相交);第五轮 8 pt,
    # xlim 放宽到 ±1.2 容纳加宽后的注记
    axe.text(np.mean(C.FAM_RHO) + 0.05, len(C.FAM_RHO) - 1 + 2.3,
             f"mean \u03c1 = {np.mean(C.FAM_RHO):+.2f}",
             ha="left", va="center", fontsize=8.0, color="black")
    axe.set_yticks([])
    axe.set_xlabel("Spearman \u03c1")
    axe.set_xlim(-1.2, 1.2)
    axe.set_ylim(-1.3, len(C.FAM_RHO) - 1 + 3.6)
    return axe


if __name__ == "__main__":
    fig, L = C.make_canvas()
    draw(fig, L)
    save(fig, "fig6_e")
    print("families:", len(C.FAM_RHO), "n range:",
          min(C.FAM_N), "-", max(C.FAM_N))
