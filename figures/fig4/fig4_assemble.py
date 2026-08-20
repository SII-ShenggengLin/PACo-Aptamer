# -*- coding: utf-8 -*-
"""Fig 4 | 配体条件化的价值("从折叠到结合"核心证据)—— 组合入口。

各子图绘图代码在 fig4_a/b/c/f/g.py(可独立运行、与拼合版共用
fig4_common.make_canvas 的同一画布几何,字号/配色天然一致;
panel 字母沿用论文正文引用,历史裁撤 d/e 不复用):
  a 消融双口径表(全局 best / 界面 best / Δ(界面-整体) / vs full (pp, P))
  b 每核苷酸恢复率 vs 距配体距离(绝对 + 归一化)
  c SS 条件保真 INF(多裁判框架 + 设计同一性分箱)
  f 自洽结构 scTM(native 上界 + 5 方法 + recovery-matched 负对照)
  g recovery-matched 负对照逐链 Δ 分布

运行:python fig4_assemble.py → out/fig4_ligand_conditioning.{pdf,png}
"""
import fig4_a
import fig4_b
import fig4_c
import fig4_common as C
import fig4_f
import fig4_g
from fig_style import save


def build():
    fig, gs = C.make_canvas()
    fig4_a.draw(fig, gs)
    fig4_b.draw(fig, gs)
    fig4_c.draw(fig, gs)
    fig4_f.draw(fig, gs)
    fig4_g.draw(fig, gs)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig4_ligand_conditioning")
