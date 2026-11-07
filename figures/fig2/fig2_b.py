"""Fig 2b | head-to-head 逐链散点:PACo-Aptamer 与 RhoDesign(477 链)。"""
import fig2_common as C
from fig_style import panel_label, save

def draw(fig, gs):
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    C.draw_h2h(ax, "RhoDesign")
    return ax

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_b")
