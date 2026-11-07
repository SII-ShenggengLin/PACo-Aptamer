"""Fig 1 | PACo-Aptamer 模型架构(overview)—— 组合入口。"""
import fig1_a
import fig1_b
import fig1_c
import fig1_d
import fig1_common as C

def build():
    fig, axs = C.new_canvas()
    fig1_a.draw(axs[0])
    fig1_b.draw(axs[1])
    fig1_c.draw(axs[2])
    fig1_d.draw(axs[3])
    for ax, letter in zip(axs, "abcd"):
        C.panel_letter(ax, letter)
    return fig

if __name__ == "__main__":
    C.save(build(), "fig1_overview")
