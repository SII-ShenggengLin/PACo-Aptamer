"""Assemble Figure 6 from the released final results and assets."""
import fig6_b
import fig6_c
import fig6_d
import fig6_e
import fig6_f
import fig6_gallery
import fig6_common as C
from fig_style import save

def build():
    fig, layout = C.make_canvas()
    fig6_gallery.draw(fig)
    fig6_b.draw(fig, layout)
    fig6_c.draw(fig, layout)
    fig6_d.draw(fig, layout)
    fig6_e.draw(fig, layout)
    fig6_f.draw(fig, layout)
    return fig

if __name__ == "__main__":
    save(build(), "fig6_case_study", pad=0.06)
