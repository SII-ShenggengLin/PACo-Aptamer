"""Fig 5 公共模块:统一画布几何 + 数据准备。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load

FIG_W_IN = 183 * MM
FIG_H_IN = 118 * MM
GRID_KW = dict(left=0.075, right=0.988, bottom=0.095, top=0.965,
               wspace=0.36, hspace=0.42)

def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×118mm 画布(2 行 × 3 列)。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(2, 3, figure=fig, **GRID_KW)
    return fig, gs

def lighten(hex_color, frac):
    """把 hex 颜色向白色混合 frac(0-1),用于温度渐变。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r += int((255 - r) * frac); g += int((255 - g) * frac); b += int((255 - b) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"

TD = load("temperature_diversity.json")
NOV = load("sequence_novelty.json")
TS = load("sequence_embedding.json")
KL = load("family_diversity.json")
GC = load("gc_mfe.json")
