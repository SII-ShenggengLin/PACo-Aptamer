"""Fig 2c | head-to-head 逐链散点:PACo-Aptamer 与 AlignIF(477 链)。"""
import fig2_common as C
from fig_style import panel_label, save

def draw(fig, gs):
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "c")
    C.draw_h2h(ax, "AlignIF")
    return ax

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_c")
