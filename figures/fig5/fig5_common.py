# -*- coding: utf-8 -*-
"""Fig 5 公共模块:统一画布几何 + 数据准备。

所有子图脚本(fig5_a..e)与组合脚本(fig5_assemble)共用:
- make_canvas():同一张 183×118 mm 画布、同一 GridSpec(2,5),
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:温度扫描 82 / 新颖性 83 / t-SNE 85 / 家族 KL 85 / GC–MFE 82。
- lighten():方法色向白混合,用于温度渐变标记。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load  # noqa: E402

# ---------- 画布几何(所有面板唯一事实来源) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 118 * MM
GRID_KW = dict(hspace=0.52, wspace=1.15, height_ratios=[1.0, 1.05])


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×118mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(2, 5, figure=fig, **GRID_KW)
    return fig, gs


def lighten(hex_color, frac):
    """把 hex 颜色向白色混合 frac(0-1),用于温度渐变。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r += int((255 - r) * frac); g += int((255 - g) * frac); b += int((255 - b) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


# ================= 数据 =================
TD = load("82_temp_diversity.json")          # 温度扫描(含 uniq)
NOV = load("83_novelty.json")                # 新颖性 rank 曲线
TS = load("85_tsne_points.json")             # t-SNE 点
KL = load("85_family_kl.json")               # 家族 KL
GC = load("82_gc_mfe.json")                  # GC–MFE
