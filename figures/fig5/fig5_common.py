# -*- coding: utf-8 -*-
"""Fig 5 公共模块:统一画布几何 + 数据准备。

所有子图脚本(fig5_a..f)与组合脚本(fig5_assemble)共用:
- make_canvas():同一张 183×118 mm 画布、同一 GridSpec(2,3),
  三列等宽 → 每行子图横坐标对齐、每列纵坐标对齐,panel 字母同步对齐;
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:温度扫描 82_gfix / 新颖性 83_gfix(训练库 + AptaDB 中位数)/
  t-SNE 85_gfix / 家族 KL 85_gfix / GC–MFE 82_gfix。
- lighten():方法色向白混合,用于温度渐变标记。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import MM, apply_style, load  # noqa: E402

# ---------- 画布几何(所有面板唯一事实来源) ----------
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


# ================= 数据 =================
# 以下五份均为 gRNAde C/G 解码修正版(2026-10-16;TS 为按原参数/种子重算的
# 联合 t-SNE,仅 gRNAde 点向量随修正池改变,其余方法向量不变)
TD = load("82_temp_diversity_gfix.json")     # 温度扫描(含 uniq)
NOV = load("83_novelty_gfix.json")           # 新颖性 rank 曲线 + AptaDB 中位数
TS = load("85_tsne_points_gfix.json")        # t-SNE 点(修正版重算)
KL = load("85_family_kl_gfix.json")          # 家族 KL
GC = load("82_gc_mfe_gfix.json")             # GC–MFE
