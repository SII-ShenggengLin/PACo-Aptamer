# -*- coding: utf-8 -*-
"""Fig 1 | PACo-Aptamer 模型架构(overview)—— 组合入口。

各子图与拼合版共用 fig1_common 的同一画布几何(fig.add_axes 绝对坐标,
纵横比与物理尺寸一致,圆圈不变形),字号/配色天然一致:
  a 架构总览(条件输入 → 编码 → 联合注意力 → 自回归解码;真实结构素材)
  b GVP encoder 细节   c transformer encoder block 细节
  d transformer decoder block 细节
真实结构素材由 fig1_pymol.py(PyMOL 无头)与 fig1_assets_data.py
(RNAfold/RNAplot)渲染 → fig1/assets/(示例链 1U8D 链 A)。

运行:python fig1_assemble.py → ../out/fig1_overview.{pdf,png}
"""
import fig1_a
import fig1_b
import fig1_c
import fig1_d
import fig1_common as C


def build():
    fig, axs = C.new_canvas()
    fig1_a.draw(axs[0])
    fig1_b.draw(axs[1])
    fig1_c.draw(axs[2])
    fig1_d.draw(axs[3])
    for ax, letter in zip(axs, "abcd"):
        C.panel_letter(ax, letter)
    return fig


if __name__ == "__main__":
    C.save(build(), "fig1_overview")
