# -*- coding: utf-8 -*-
"""Fig 5 | 生成质量:多样性、新颖性、进化保真 —— 组合入口。

各子图绘图代码在 fig5_a.py … fig5_f.py(可独立运行、与拼合版共用
fig5_common.make_canvas 的同一画布几何,字号/配色天然一致):
  a 温度扫描 recovery–diversity 前沿(温度渐变着色)
  b 新颖性 rank-ordered 最大相似度曲线(vs 训练库)
  c 新颖性双参照哑铃点图(训练库 vs AptaDB 中位数)
  d 生成池 4-mer 组成 t-SNE(60 链 × 5 方法 × 30 设计 + native)
  e 家族组成保真 KL(LOO TM 簇 profile,含随机参照锚)
  f 序列可行性 GC–MFE 2D 图(native 五角星参照)

运行:python fig5_assemble.py → out/fig5_generation_quality.{pdf,png}
"""
import fig5_a
import fig5_b
import fig5_c
import fig5_common as C
import fig5_d
import fig5_e
import fig5_f
from fig_style import save


def build():
    fig, gs = C.make_canvas()
    fig5_a.draw(fig, gs)
    fig5_b.draw(fig, gs)
    fig5_c.draw(fig, gs)
    fig5_d.draw(fig, gs)
    fig5_e.draw(fig, gs)
    fig5_f.draw(fig, gs)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig5_generation_quality")
