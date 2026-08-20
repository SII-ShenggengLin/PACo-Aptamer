# -*- coding: utf-8 -*-
"""Fig 3 | Zero-shot 配体级隔离泛化 —— 组合入口。

各子图绘图代码在 fig3_a.py … fig3_g.py(可独立运行、与拼合版共用
fig3_common.make_canvas 的同一画布几何,字号/配色天然一致):
  a 三重隔离协议总览(结构簇 8:1:1 + 时间切分 + 配体级隔离层级图解)
  b,c,d zs 93 结果(柱状 + head-to-head vs RhoDesign + 逐链 Δ 小提琴)
  e,f,g indep 235 结果(同上布局)

运行:python fig3_assemble.py → out/fig3_zeroshot_protocol.{pdf,png}
"""
import fig3_a
import fig3_b
import fig3_c
import fig3_d
import fig3_e
import fig3_f
import fig3_g
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
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig3_zeroshot_protocol")
