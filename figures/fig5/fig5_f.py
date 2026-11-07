"""Fig 5f | 序列可行性 GC–MFE 2D 图(五方法点 + native 五角星参照)。"""
import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, NATIVE, panel_label, save

LBL = {
    "PACo-Aptamer": ((58.2, -0.262), "left", "center"),
    "RhoDesign": ((50.6, -0.222), "right", "center"),
    "AlignIF": ((51.8, -0.352), "center", "center"),
    "NA-MPNN": ((47.6, -0.313), "right", "center"),
    "gRNAde": ((59.9, -0.446), "left", "center"),
}

def draw(fig, gs):
    axf = fig.add_subplot(gs[1, 2])
    panel_label(axf, "f")
    nat = C.GC["native_ref"]
    for m in METHOD_ORDER:
        key = "PACo" if m == "PACo-Aptamer" else m
        g = C.GC["methods"][key]
        gx, gy = 100 * g["gc_mean"], g["mfe_nt_mean"]
        axf.scatter(gx, gy, s=22, c=METHOD_COLOR[m], zorder=6, ec="white",
                    lw=0.4)
        (tx, ty), ha, va = LBL[m]
        axf.annotate(m, (gx, gy), xytext=(tx, ty), fontsize=5.2, ha=ha,
                     va=va, color="black",
                     arrowprops=dict(arrowstyle="-", lw=0.4,
                                     color=METHOD_COLOR[m], shrinkA=1,
                                     shrinkB=3))

    axf.scatter(100 * nat["gc_mean"], nat["mfe_nt_mean"], marker="*", s=60,
                c=NATIVE, zorder=4, ec="#7A7672", lw=0.6)
    axf.annotate("native", (100 * nat["gc_mean"], nat["mfe_nt_mean"]),
                 xytext=(100 * nat["gc_mean"] + 0.9,
                         nat["mfe_nt_mean"] + 0.014),
                 fontsize=5.2, ha="left", va="center", color="black",
                 arrowprops=dict(arrowstyle="-", lw=0.4, color="#7A7672",
                                 shrinkA=1, shrinkB=4))
    axf.set_xlabel("GC content (%)")
    axf.set_ylabel("MFE per nucleotide\n(kcal/mol)")
    axf.set_xlim(41.5, 66.5); axf.set_ylim(-0.53, -0.14)
    axf.set_xticks([45, 50, 55, 60, 65])
    axf.set_yticks([-0.5, -0.4, -0.3, -0.2])
    return axf

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_f", pad=0.05)
