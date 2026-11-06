# -*- coding: utf-8 -*-
"""Fig 2e | 链级置信度–恢复率耦合 hexbin(ρ 与 95% CI 由图注承载)。

hexbin 用莫兰迪蓝渐变 paco_blues;计数色条内嵌于坐标区右缘中下段
(该区域零数据点,实测空置),不再外挂在面板右侧。
注意:C.conf_y 在公共模块中已是百分数(0-100),这里不再乘 100
(旧版多乘一层,纵轴数值大了 100 倍;ρ 为秩相关不受影响)。
单独运行:python fig2_e.py → out/fig2_e.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig2_common as C
from fig_style import panel_label, save


def draw(fig, gs):
    axe = fig.add_subplot(gs[1, 1])
    panel_label(axe, "e")
    hb = axe.hexbin(C.conf_x, C.conf_y, gridsize=26, cmap="paco_blues",
                    mincnt=1, linewidths=0.1)
    axe.set_xlim(C.conf_x.min(), C.conf_x.max())
    axe.set_xlabel("Mean log probability of sampled designs")
    axe.set_ylabel("Recovery, t = 0.1 mean (%)")
    axcb = axe.inset_axes([0.875, 0.06, 0.05, 0.50])
    cb = fig.colorbar(hb, cax=axcb)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=4.6, pad=1.2, length=1.5, width=0.5)
    return axe


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_e")
