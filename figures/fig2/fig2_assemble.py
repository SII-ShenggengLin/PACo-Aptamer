# -*- coding: utf-8 -*-
"""Fig 2 | 严格防泄漏基准下的全面领先(主结果图)—— 组合入口。

各子图绘图代码在 fig2_a.py … fig2_i.py(可独立运行、与拼合版共用
fig2_common.make_canvas 的同一画布几何,字号/配色天然一致):
  a 评估协议流程  b 六方法主表  c 恢复率柱状(best/t01 双口径+Wilcoxon P)
  d,e head-to-head 逐链散点(vs RhoDesign / vs AlignIF)  f 逐靶 Δ 双向条形
  g 界面恢复率(绝对+归一化)  h 置信度-恢复率相关(链间/链内/富集)
  i 长度分层箱线

运行:python fig2_assemble.py → out/fig2_main_results.{pdf,png}
"""
import fig2_a
import fig2_b
import fig2_c
import fig2_d
import fig2_e
import fig2_f
import fig2_g
import fig2_h
import fig2_i
import fig2_common as C
from fig_style import save


def build():
    fig, gs = C.make_canvas()
    fig2_a.draw(fig, gs)
    fig2_b.draw(fig, gs)
    fig2_c.draw(fig, gs)
    fig2_d.draw(fig, gs)
    fig2_e.draw(fig, gs)
    fig2_f.draw(fig, gs)
    fig2_g.draw(fig, gs)
    fig2_h.draw(fig, gs)
    fig2_i.draw(fig, gs)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig2_main_results")
    print("chain-level rho check:", round(C.rho_chain, 4),
          "| within n =", len(C.within))
