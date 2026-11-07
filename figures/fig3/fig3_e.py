"""Fig 3e | independent test 235 链 head-to-head:PACo-Aptamer vs RhoDesign 逐链散点。"""
import fig3_common as C
from fig_style import save

def draw(fig, gs):
    axh = fig.add_subplot(gs[1, 2:4])
    C.draw_h2h(axh, "independent_test", "e")
    return axh

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig3_e")
