"""Fig 5 | 生成质量:多样性、新颖性、进化保真 —— 组合入口。"""
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
