# -*- coding: utf-8 -*-
"""Fig 2b | 主表:splits_v2 test 477 链五方法双口径恢复率 + 覆盖率 + 池规模。

最优数值加粗着色;TACo 行雾蓝底。数据:63_main_table_gfix.json。
单独运行:python fig2_b.py → out/fig2_b.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save


def draw(fig, gs):
    axb = fig.add_subplot(gs[0, 3:8]); axb.axis("off")
    panel_label(axb, "b", x=-0.052, y=1.05)
    hdr = ["Method", "Best sample (%)", "t = 0.1 mean (%)", "Coverage", "Pool size"]
    cell = []
    for m in METHOD_ORDER:
        t = C.main["TACoAptamer" if m == "TACo-Aptamer" else m]
        pool = {"TACo-Aptamer": "3T × 100 × 2", "RhoDesign": "3T × 100",
                "AlignIF": "T0.1 × 20", "NA-MPNN": "T0.1 × 10",
                "gRNAde": "3T × 100"}[m]
        cell.append([m, f"{100*t['recovery_best_mean']:.2f}",
                     f"{100*t['recovery_t01_mean']:.2f}", t["coverage"], pool])
    tb = axb.table(cellText=cell, colLabels=hdr, cellLoc="center",
                   bbox=[0.0, 0.13, 1.0, 0.91])
    tb.auto_set_font_size(False); tb.set_fontsize(5.6)
    best_b = max(float(r[1]) for r in cell); best_t = max(float(r[2]) for r in cell)
    for (i, j), c in tb.get_celld().items():
        c.set_linewidth(0.4); c.set_edgecolor("#C4C4C0")
        if i == 0:
            c.set_facecolor("#EFEDEA"); c.set_text_props(fontweight="bold")
        else:
            m = cell[i - 1][0]
            c.set_facecolor("#EDF2F7" if m == "TACo-Aptamer" else "white")
            if j in (1, 2):
                c.set_text_props(color=METHOD_COLOR[m])
                if float(cell[i - 1][j]) == (best_b if j == 1 else best_t):
                    c.set_text_props(color=METHOD_COLOR[m], fontweight="bold")
    axb.text(0.5, 0.035,
             "splits_v2 test, n = 477 chains · canonical native sequences · random baseline 25%",
             transform=axb.transAxes, ha="center", fontsize=5.0, color="#545450")
    return axb


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_b")
