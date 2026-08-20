# -*- coding: utf-8 -*-
"""Fig 4f | 自洽结构 scTM 水平箱线:native 上界 + 5 方法 + recovery-matched 负对照。

右侧列注 mean · %pass@0.45;TM = 0.45 参考虚线。
单独运行:python fig4_f.py → out/fig4_f.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig4_common as C
from fig_style import METHOD_COLOR, NATIVE, NEG, panel_label, save


def draw(fig, gs):
    axf = fig.add_subplot(gs[1, 4:6])
    panel_label(axf, "f", x=-0.28, y=1.12)
    frows = [("native", "native", NATIVE),
             ("NA-MPNN", "NA-MPNN", METHOD_COLOR["NA-MPNN"]),
             ("TACo-Aptamer", "TACo", METHOD_COLOR["TACo-Aptamer"]),
             ("RhoDesign", "RhoDesign", METHOD_COLOR["RhoDesign"]),
             ("random-matched", "randmatch", NEG),
             ("gRNAde", "gRNAde", METHOD_COLOR["gRNAde"])]
    for i, (lab, key, col) in enumerate(frows):
        vals = np.array([v[f"sctm_{key}"] for v in C.SCTM["per_chain"].values()
                         if f"sctm_{key}" in v])
        yp = len(frows) - 1 - i
        bp = axf.boxplot([vals], positions=[yp], vert=False, widths=0.55,
                         whis=1.5, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", lw=0.7),
                         boxprops=dict(lw=0.4), whiskerprops=dict(lw=0.4),
                         capprops=dict(lw=0.4))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.85)
        axf.scatter([np.mean(vals)], [yp], color="black", s=5, lw=0, zorder=5)
        sm = C.SCTM["summary"][key]
        axf.text(1.02, yp, f"{sm['mean']:.3f} · {100*sm['pass_045']:.0f}%",
                 va="center", fontsize=4.7, transform=axf.get_yaxis_transform(),
                 fontweight="bold" if key in ("native", "TACo") else "normal")
    axf.set_yticks(range(len(frows)))
    axf.set_yticklabels([r[0].replace("-Aptamer", "") for r in frows][::-1],
                        fontsize=5.0)
    axf.axvline(0.45, color="#8C8C88", lw=0.5, ls=":")
    axf.text(0.455, 5.42, "TM = 0.45", fontsize=4.4, color="#8C8C88")
    axf.set_xlabel("scTM (RhoFold+ refold, vs native)")
    axf.set_xlim(0, 1.0)
    axf.text(1.02, 5.48, "mean · %pass@0.45", fontsize=4.4,
             transform=axf.get_yaxis_transform(), color="#545450")
    return axf


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig4_f")
