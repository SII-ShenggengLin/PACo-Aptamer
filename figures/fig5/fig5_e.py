# -*- coding: utf-8 -*-
"""Fig 5e | 序列可行性 GC–MFE 2D 图(五方法点 + native 五角星参照)。

方法名标签手动错位避让;引线连至各自数据点。
单独运行:python fig5_e.py → out/fig5_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, NATIVE, panel_label, save


def draw(fig, gs):
    axe = fig.add_subplot(gs[1, 4])
    panel_label(axe, "e", x=-0.30, y=1.12)
    nat = C.GC["native_ref"]
    LBL = {  # (文字锚点偏移, ha)
        "TACo": ((2.2, 0.018), "left"), "RhoDesign": ((-2.2, -0.030), "right"),
        "AlignIF": ((2.2, -0.016), "left"), "NA-MPNN": ((-2.2, 0.020), "right"),
        "gRNAde": ((2.2, 0.018), "left")}
    for m in METHOD_ORDER:
        key = "TACo" if m == "TACo-Aptamer" else m
        g = C.GC["methods"][key]
        gx, gy = 100 * g["gc_mean"], g["mfe_nt_mean"]
        axe.scatter(gx, gy, s=22, c=METHOD_COLOR[m], zorder=4, ec="white",
                    lw=0.4)
        off, ha = LBL[key]
        axe.annotate(key, (gx, gy), xytext=(gx + off[0], gy + off[1]),
                     fontsize=4.8, color=METHOD_COLOR[m], ha=ha, va="center",
                     arrowprops=dict(arrowstyle="-", lw=0.4,
                                     color=METHOD_COLOR[m], shrinkA=0,
                                     shrinkB=3))
    axe.scatter(100 * nat["gc_mean"], nat["mfe_nt_mean"], marker="*", s=120,
                c=NATIVE, zorder=6, ec="#3D5A45", lw=0.6)
    axe.annotate("native", (100 * nat["gc_mean"], nat["mfe_nt_mean"]),
                 xytext=(100 * nat["gc_mean"] - 2.4, nat["mfe_nt_mean"] + 0.026),
                 fontsize=5.2, color="#3D5A45", fontweight="bold", ha="right",
                 arrowprops=dict(arrowstyle="-", lw=0.4, color="#3D5A45",
                                 shrinkA=0, shrinkB=5))
    axe.set_xlabel("GC content (%)")
    axe.set_ylabel("MFE per nucleotide\n(kcal mol$^{-1}$ nt$^{-1}$)")
    axe.set_xlim(42, 64); axe.set_ylim(-0.48, -0.17)
    return axe


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_e")
