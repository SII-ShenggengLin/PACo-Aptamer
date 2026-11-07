"""Fig 4 | 配体条件化的价值("从折叠到结合"核心证据)—— 组合入口。"""
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
