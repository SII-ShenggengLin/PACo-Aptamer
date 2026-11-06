# -*- coding: utf-8 -*-
"""Fig 4 | 配体条件化的价值("从折叠到结合"核心证据)—— 组合入口。

各子图绘图代码在 fig4_a..i.py(可独立运行、与拼合版共用
fig4_common.make_canvas 的同一画布几何,字号/配色天然一致)。
2026-10-22 第二轮起为 3×3、九个独立编号面板(与 Fig. 2/3 同款 183×159mm
画布,行列严格对齐;a–h 字母含义不变,按阅读顺序重排,i 为新增):
  a 消融哑铃(全局 best / 界面 best 双口径)
  b 界面恢复率柱状(≤5 Å 界面位置,绝对口径)
  c 每核苷酸恢复率 vs 距配体距离(绝对口径)
  d 同 c,相对各链自身均值
  e SS 条件保真 INF 柱状(native / random 点状参照线)
  f SS 保真随设计同一性变化
  g 自洽结构 scTM 箱线(native + 4 方法 + recovery-matched 负对照)
  h recovery-matched 负对照逐链 scTM 差分布
  i 碱基对接触保真 scMCC 点距图(自 Supplementary Fig. 10 提升)

运行:python fig4_assemble.py → out/fig4_ligand_conditioning.{pdf,png}
"""
import fig4_a
import fig4_b
import fig4_c
import fig4_common as C
import fig4_d
import fig4_e
import fig4_f
import fig4_g
import fig4_h
import fig4_i
from fig_style import save


def build():
    fig, gs = C.make_canvas()
    fig4_a.draw(fig, gs)
    fig4_b.draw(fig, gs)
    fig4_c.draw(fig, gs)
    fig4_d.draw(fig, gs)
    fig4_e.draw(fig, gs)
    fig4_f.draw(fig, gs)
    fig4_g.draw(fig, gs)
    fig4_h.draw(fig, gs)
    fig4_i.draw(fig, gs)
    C.tighten_ylabels(fig)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig4_ligand_conditioning")
