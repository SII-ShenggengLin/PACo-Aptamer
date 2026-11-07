"""Fig 3 | 未见配体与预测骨架上的泛化 —— 组合入口。"""
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
