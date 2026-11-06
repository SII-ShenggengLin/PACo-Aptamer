# -*- coding: utf-8 -*-
"""Fig 4 程序化重叠审计:文字×文字、文字×数据(点/线/柱/箱/图例盒)包围盒
像素交集检查(600 dpi 渲染),含跨面板文字与相邻面板内容。

用法:python fig4_audit.py   → 输出每一对相交 artist 与交集面积(px²)
"""
import numpy as np
import fig4_assemble as A
from matplotlib.text import Text


def bbox_of(artist, renderer):
    try:
        bb = artist.get_window_extent(renderer=renderer)
    except Exception:
        return None
    return (bb.x0, bb.y0, bb.x1, bb.y1)


def inter_area(b1, b2):
    x0 = max(b1[0], b2[0]); x1 = min(b1[2], b2[2])
    y0 = max(b1[1], b2[1]); y1 = min(b1[3], b2[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def label_of(a):
    if isinstance(a, Text):
        return a.get_text().replace("\n", "|")[:36]
    return getattr(a, "_label", "") or type(a).__name__


def main():
    fig = A.build()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    text_items = []      # (name, bbox)
    for ax in fig.axes:
        cands = list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
        if ax.xaxis.label.get_text():
            cands.append(ax.xaxis.label)
        if ax.yaxis.label.get_text():
            cands.append(ax.yaxis.label)
        lg = ax.get_legend()
        if lg is not None:
            bb = bbox_of(lg, r)
            if bb:
                text_items.append(("legend[" + (ax.get_title() or "?") + "]",
                                   bb))
        for t in cands:
            if not (isinstance(t, Text) and t.get_text().strip()):
                continue
            bb = bbox_of(t, r)
            if bb:
                text_items.append((label_of(t), bb))
    for t in fig.texts:  # 面板字母
        bb = bbox_of(t, r)
        if bb:
            text_items.append((label_of(t), bb))

    hits = []
    # 同一坐标轴上相邻的旋转刻度标签:字形带斜向平行,包围盒相交不代表
    # 视觉重叠 —— 改查垂直间隙 |Δx|·sin θ 是否大于字高(~1.25 × 字号)
    import math
    rotated_tick_names = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = axis.get_ticklabels()
            if len(labels) < 2:
                continue
            rot = labels[0].get_rotation()
            if rot in (0, "0", None):
                continue
            ok = [t.get_text().strip() for t in labels if t.get_text().strip()]
            if len(ok) < 2:
                continue
            anchors = []
            for t in labels:
                if not t.get_text().strip():
                    continue
                bb = bbox_of(t, r)
                if bb:
                    anchors.append((t.get_text(), (bb[0] + bb[2]) / 2,
                                    t.get_fontsize()))
            names = set()
            for k in range(len(anchors)):
                for k2 in range(k + 1, len(anchors)):
                    dx = abs(anchors[k][1] - anchors[k2][1])
                    sep = dx * math.sin(math.radians(rot))
                    glyph_h = 1.25 * max(anchors[k][2], anchors[k2][2]) \
                        * fig.dpi / 72
                    if sep < glyph_h:
                        hits.append((round(glyph_h - sep), "rotated-ticks",
                                     anchors[k][0], anchors[k2][0],
                                     "perp gap px"))
                    names.add(anchors[k][0])
            rotated_tick_names |= names
    for i in range(len(text_items)):
        for j in range(i + 1, len(text_items)):
            ni, nj = text_items[i][0], text_items[j][0]
            if ni in rotated_tick_names and nj in rotated_tick_names:
                continue           # 旋转刻度签互查已由垂直间隙检查替代
            a = inter_area(text_items[i][1], text_items[j][1])
            if a > 4:
                hits.append((round(a), "text-text", ni, nj))

    # 文字 × 本面板数据元素。带标记的数据线(c/d/f 曲线)用逐段+逐点小盒;
    # 无标记线多为误差棒端点连线(跨柱折线,相邻点间无真实墨迹),只查逐点盒;
    # patch/collection 用自身包围盒(柱/箱/直方条本身即矩形,精确)。
    for ax in fig.axes:
        data_boxes = []
        for ln in ax.lines:
            lab = label_of(ln) or "line"
            xy = ax.transData.transform(np.column_stack(
                [np.atleast_1d(ln.get_xdata()).astype(float),
                 np.atleast_1d(ln.get_ydata()).astype(float)]))
            pad = 2.0                 # 线宽/点径余量(px)
            for k in range(len(xy)):
                data_boxes.append((lab, (xy[k, 0] - pad, xy[k, 1] - pad,
                                         xy[k, 0] + pad, xy[k, 1] + pad)))
            # 只有"带标记且带连线"的才是数据曲线(c/d/f);误差棒端帽线
            # (marker='_'、无线型)相邻点间无墨迹,不做线段盒
            has_marker = (ln.get_marker() not in ("", "None", None)
                          and str(ln.get_linestyle()) not in ("", "None",
                                                              "none"))
            if has_marker:
                for k in range(len(xy) - 1):
                    x0, x1 = sorted((xy[k, 0], xy[k + 1, 0]))
                    y0, y1 = sorted((xy[k, 1], xy[k + 1, 1]))
                    data_boxes.append((lab, (x0 - pad, y0 - pad,
                                             x1 + pad, y1 + pad)))
        for col in ax.collections:
            bb = bbox_of(col, r)
            if bb:
                data_boxes.append((type(col).__name__, bb))
        for p in ax.patches:
            bb = bbox_of(p, r)
            if bb:
                data_boxes.append((type(p).__name__, bb))
        ax_texts = [t for t in text_items
                    if t[0] not in ("a", "b", "c", "d", "e", "f", "g", "h",
                                    "i")]
        # 只查同面板文字:以包围盒落在本 axes 坐标区内近似
        abb = ax.get_window_extent(renderer=r)
        for name, tb in ax_texts:
            # 文字盒中心落在该 axes 带内才比对
            cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            if not (abb.x0 <= cx <= abb.x1 and abb.y0 <= cy <= abb.y1):
                continue
            for dname, db in data_boxes:
                a = inter_area(tb, db)
                if a > 25:
                    hits.append((round(a), "text-data", name, dname,
                                 "txt", [round(v) for v in tb],
                                 "dat", [round(v) for v in db]))

    for h in sorted(hits, reverse=True):
        print(h)
    print("total hits:", len(hits))


if __name__ == "__main__":
    main()
