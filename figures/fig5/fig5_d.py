# -*- coding: utf-8 -*-
"""Fig 5d | 家族组成保真 KL(LOO TM 簇 profile + 随机参照锚)。

random ref 虚线为最大熵锚(注明"lower is not better across the divide");
标签置于两柱间隙上方 + 白底衬,避免与柱体/旋转刻度标签相碰。
单独运行:python fig5_d.py → out/fig5_d.{pdf,png}(与拼合版物理尺寸一致)
"""
import numpy as np

import fig5_common as C
from fig_style import METHOD_COLOR, METHOD_ORDER, panel_label, save


def draw(fig, gs):
    axd = fig.add_subplot(gs[1, 3])
    panel_label(axd, "d", x=-0.30, y=1.12)
    mm = METHOD_ORDER
    kvm = [C.KL["methods"][("TACo" if m == "TACo-Aptamer" else m)]["kl_mean"]
           for m in mm]
    kci = [C.KL["methods"][("TACo" if m == "TACo-Aptamer" else m)]["kl_mean"] -
           C.KL["methods"][("TACo" if m == "TACo-Aptamer" else m)]["ci95"][0]
           for m in mm]
    xd = np.arange(len(mm))
    axd.bar(xd, kvm, 0.62, yerr=kci, color=[METHOD_COLOR[m] for m in mm],
            error_kw=dict(lw=0.5, capsize=1.3), ec="white", lw=0.3)
    for x0, v in zip(xd, kvm):
        axd.text(x0, v + 0.10, f"{v:.2f}", ha="center", fontsize=4.8,
                 fontweight="bold" if v == min(kvm) else "normal")
    rr = C.KL["random_ref"]
    axd.axhline(rr["kl_mean"], color="#B07060", lw=0.7, ls="--")
    axd.text(0.60, rr["kl_mean"] + 0.10, f"random ref {rr['kl_mean']:.2f}",
             fontsize=4.7, color="#B07060", ha="center", va="bottom",
             transform=axd.get_yaxis_transform(),
             bbox=dict(fc="white", ec="none", pad=0.8))
    axd.set_xticks(xd); axd.set_xticklabels(
        [m.replace("-Aptamer", "") for m in mm], rotation=32, ha="right")
    axd.set_ylabel("KL(designs | family profile)")
    axd.set_ylim(3.0, 5.3)
    return axd


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig5_d")
