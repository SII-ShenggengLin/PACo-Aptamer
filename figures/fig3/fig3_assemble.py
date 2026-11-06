# -*- coding: utf-8 -*-
"""Fig 3 | 未见配体与预测骨架上的泛化 —— 组合入口。

各子图绘图代码在 fig3_a.py … fig3_i.py(可独立运行、与拼合版共用
fig3_common.make_canvas 的同一画布几何,字号/配色天然一致):
  a,b,c zero-shot 93 结果(柱状 + head-to-head vs RhoDesign + 逐链优势直方图)
  d,e,f indep 235 结果(点图 + head-to-head + 直方图,与上行列对齐)
  g,h,i 预测骨架 193 链(晶体↔预测哑铃图;骨架质量依赖散点;
      zs×predbb 交叉切片)—— 三行三列,列边界与上两行严格对齐

运行:python fig3_assemble.py → out/fig3_generalisation.{pdf,png}
"""
import fig3_a
import fig3_b
import fig3_c
import fig3_d
import fig3_e
import fig3_f
import fig3_g
import fig3_h
import fig3_i
import fig3_common as C
from fig_style import save


def build():
    fig, gs = C.make_canvas()
    fig3_a.draw(fig, gs)
    fig3_b.draw(fig, gs)
    fig3_c.draw(fig, gs)
    fig3_d.draw(fig, gs)
    fig3_e.draw(fig, gs)
    fig3_f.draw(fig, gs)
    fig3_g.draw(fig, gs)
    fig3_h.draw(fig, gs)
    fig3_i.draw(fig, gs)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig3_generalisation")
