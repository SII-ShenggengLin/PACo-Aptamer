# -*- coding: utf-8 -*-
"""Fig 6 渲染图透明边裁剪(PyMOL 渲染后的后处理步骤)。

PyMOL 按内容包围盒等比取景,内容宽高比 ≠ 画幅比例时四周留大片透明边,
使画布上"标题/注记 ←→ 结构图像"的视觉距离虚大。本脚本:
  1. 画廊 12 张(2400×1600):各自裁到内容 bbox + 16px 边距,等比放大贴满
     统一 1.5 画幅的长轴、居中 → 格子几何/对齐不变,长轴零留白、短轴留白
     由该格内容宽高比决定(0–17%);分辨率 ≥600 ppi @ 显示宽 1.562 in;
  2. b 两张(3000×1500):各自裁到内容 bbox + 16px,贴长轴放大后居中到
     统一公共画幅(两图同尺寸,b 槽位等幅对齐)。
管线:fig6_gallery_pymol.py / fig6_pymol.py → python fig6_trim.py → 拼图。
"""
import glob
import os

import numpy as np
from PIL import Image

ROOT = "/inspire/hdd/global_user/linshenggeng-240107100015/RNA_tertiary_stru_IF"
GAL_DIR = os.path.join(ROOT, "figures", "out", "fig6_gallery")
OUT = os.path.join(ROOT, "figures", "out")
PAD = 16


def content_bbox(img):
    alpha = np.array(img.convert("RGBA"))[..., 3] > 8
    rows = np.where(alpha.any(axis=1))[0]
    cols = np.where(alpha.any(axis=0))[0]
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1   # 半开区间


def crop_content(path):
    img = Image.open(path).convert("RGBA")
    x0, y0, x1, y1 = content_bbox(img)
    return img.crop((max(0, x0 - PAD), max(0, y0 - PAD),
                     min(img.width, x1 + PAD), min(img.height, y1 + PAD)))


def unify(paths, aspect):
    """裁剪 → 等比放大贴满统一 aspect 画幅的长轴 → 居中覆盖保存。"""
    pieces = [crop_content(p) for p in paths]
    # 画幅尺寸:各内容贴长轴后的最大宽/高,再互相适配到目标 aspect
    w = max(round(c.width * min(1.0, aspect / (c.width / c.height)))
            if c.width / c.height < aspect else c.width for c in pieces)
    h = max(round(c.height * min(1.0, (c.width / c.height) / aspect))
            if c.width / c.height > aspect else c.height for c in pieces)
    if w / h < aspect:
        w = round(h * aspect)
    else:
        h = round(w / aspect)
    for c, path in zip(pieces, paths):
        s = min(w / c.width, h / c.height)          # 贴满长轴且不超出画布
        c2 = c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(c2, ((w - c2.width) // 2, (h - c2.height) // 2), c2)
        canvas.save(path)
        print(f"trim {os.path.basename(path)}: -> {w}x{h} "
              f"(content {c2.width}x{c2.height})")
    return w, h


if __name__ == "__main__":
    gal = sorted(glob.glob(os.path.join(GAL_DIR, "*.png")))
    w, h = unify(gal, 1.5)
    print(f"gallery unified canvas {w}x{h} (aspect {w/h:.3f})")

    duo = [os.path.join(OUT, "fig6_pymol_overview.png"),
           os.path.join(OUT, "fig6_pymol_pocket.png")]
    w, h = unify(duo, 1.96)
    print(f"b renders unified canvas {w}x{h} (aspect {w/h:.3f})")
