# -*- coding: utf-8 -*-
"""Fig 1 | TACo-Aptamer 架构与数据(overview)—— 组合入口。

各子图绘图代码在 fig1_a.py … fig1_e.py(可独立运行、与拼合版共用
fig1_common.make_canvas 的同一画布几何,字号/配色天然一致):
  a 范式示意(现有方法 vs 靶标条件化)
  b 数据漏斗(PDB → 纯 RNA/复合物 → 双重聚类 → 三重隔离划分)
  c 模型架构数据流(GVP + 配体 token 混合序列 Transformer + 自回归解码)
  d 两阶段训练(S1 预训练 → S2 复合物微调 → ens2 集成)
  e 主结果预览 mini-panel

运行:python fig1_assemble.py → out/fig1_overview.{pdf,png}
"""
import fig1_a
import fig1_b
import fig1_c
import fig1_d
import fig1_e
import fig1_common as C
from fig_style import save


def build():
    fig, gs = C.make_canvas()
    fig1_a.draw(fig, gs)
    fig1_b.draw(fig, gs)
    fig1_c.draw(fig, gs)
    fig1_d.draw(fig, gs)
    fig1_e.draw(fig, gs)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig1_overview")
