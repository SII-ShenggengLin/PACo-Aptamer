# -*- coding: utf-8 -*-
"""Fig 1c | 模型架构数据流:GVP 编码器 + 配体 token 混合序列 Transformer + 自回归解码。

输入三 chips(backbone / SS map / ligand pocket atoms)→ GVP×3 + 配体 MLP →
混合序列 Transformer 编码器×3(joint attention)→ 自回归解码器×3。
单独运行:python fig1_c.py → out/fig1_c.{pdf,png}(与拼合版物理尺寸一致)
"""
import fig1_common as C
from fig_style import TACO_D, panel_label, save


def draw(fig, gs):
    axc = fig.add_subplot(gs[1, 0:4])
    axc.set_xlim(0, 20); axc.set_ylim(0, 10); axc.set_axis_off()
    panel_label(axc, "c", x=-0.035, y=1.12)
    # 输入三 chips
    C.box(axc, 0.2, 7.6, 4.3, 2.0, C.C_DATA,
          "RNA backbone\n9 atoms / nt (C1'-centered\nframes, dihedrals)", fs=4.6)
    C.box(axc, 0.2, 4.9, 4.3, 2.0, C.C_SS,
          "SS pairing map\nRNAfold MFE base pairs\n(edge features)", fs=4.6,
          tc="#3D5A45")
    C.box(axc, 0.2, 2.2, 4.3, 2.0, "#F6E3DC",
          "Ligand pocket atoms\nelement one-hot + dist. RBF\n(token per atom)",
          fs=4.6, ec="#E8A088")
    # GVP
    for y0 in (7.6, 4.9):
        C.arrow(axc, 4.5, y0 + 1.0, 5.5, 6.3, rad=0.1)
    C.arrow(axc, 4.5, 3.2, 5.5, 4.0, rad=-0.1, color="#B0755F")
    C.box(axc, 5.5, 5.15, 3.4, 2.6, C.C_MODEL,
          "GVP encoder ×3\nscalar 512 · vector 256\nSS pair edges", fs=4.6)
    C.box(axc, 5.5, 1.8, 3.4, 2.2, "#D8C8B0",
          "ligand embedding\n2-layer MLP (512)", fs=4.6, ec="#E8A088")
    C.arrow(axc, 8.9, 6.45, 9.9, 6.45)
    C.arrow(axc, 8.9, 2.9, 9.9, 5.0, color="#B0755F")
    C.box(axc, 9.9, 3.9, 4.6, 4.2, C.C_MODEL,
          "Transformer encoder ×3\n512 d · 4 heads\n\nmixed sequence:\nRNA tokens + ligand tokens\n(joint attention)", fs=4.6)
    C.arrow(axc, 14.5, 6.0, 15.5, 6.0)
    C.box(axc, 15.5, 3.9, 4.3, 4.2, C.C_MODEL,
          "Transformer\ndecoder ×3\n\nautoregressive\nA/C/G/U\n(temperature\nsampling)", fs=4.6)
    axc.text(17.65, 3.35, "sequence", ha="center", fontsize=4.8, color=TACO_D,
             fontweight="bold")
    axc.text(10, 1.4, "cross-attention carries ligand context to every decoding step",
             fontsize=4.5, color="#B0755F", ha="center", style="italic")
    return axc


if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig1_c")
