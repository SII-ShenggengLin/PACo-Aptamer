# -*- coding: utf-8 -*-
"""Fig 6 组装后审计:文字/图像/图例重叠 + 面板字母与坐标轴对齐 + 字号清单。

用法:python fig6_audit.py(先跑 fig6_assemble.py 生成整图)
检查项:
  1. 全图所有文字 artist(标题/注记/轴题/刻度签/图例文字/面板字母)两两
     bbox 不相交;
  2. 文字不与画廊结构图、b 渲染图的 bbox 相交,且上下净空 ≥2 pt;
  3. 相邻画廊行"注记底 → 下一行格题顶"净空 ≥3 mm;
  4. d/f 图例框与柱/误差棒在数据空间零相交;e 均值注记与茎/点零相交;
  5. 面板字母按行同 y、按列同 x(容差 0.5 mm);
  6. b/c、d/e/f 行内 axes 顶/底对齐(容差 0.5 mm);
  7. 全部文字颜色为黑色、不加粗(面板字母除外,惯例粗体);
  8. 字号清单(同一类别应一致)。
"""
import sys

import numpy as np

import fig6_assemble
import fig6_common as C
import fig6_gallery


def _bbox(artist, fig):
    r = fig.canvas.get_renderer()
    return artist.get_window_extent(r)


def _in2mm(v):
    return v / fig.get_dpi() * 25.4


fig = fig6_assemble.build()
fig.canvas.draw()
r = fig.canvas.get_renderer()
DPI = fig.get_dpi()
issues = []

# ---------- 收集文字 artist ----------
texts = []          # (标签, artist, 类别)
for t in fig.texts:
    if t.get_text().strip():
        texts.append((t.get_text().strip()[:28], t, "fig"))
for ax in fig.axes:
    if ax.get_title():
        texts.append((ax.get_title()[:28], ax.title, "ax-title"))
    for t in ax.texts:
        if t.get_text().strip():
            texts.append((t.get_text()[:28], t, "ax-text"))
    texts.append((ax.xaxis.label.get_text()[:28], ax.xaxis.label, "ax-xlab"))
    texts.append((ax.yaxis.label.get_text()[:28], ax.yaxis.label, "ax-ylab"))
    if ax.axison:                     # imshow 轴 axis off,刻度签 bbox 过期
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for kind, tls in (("x", ax.get_xticklabels()),
                          ("y", ax.get_yticklabels())):
            for tl in tls:
                if not tl.get_text():
                    continue
                # 视窗外(如 ylim 收口前的 25、xlim 外的 −1.5)不绘制的刻度
                # artist 的 bbox 过期,跳过
                p = tl.get_position()[0 if kind == "x" else 1]
                lo, hi = (x0, x1) if kind == "x" else (y0, y1)
                if not (lo - 1e-9 <= p <= hi + 1e-9):
                    continue
                texts.append((tl.get_text(), tl, "tick"))
    leg = ax.get_legend()
    if leg is not None:
        bb = leg.get_window_extent(r)
        texts.append(("<legend-box>", type("L", (), {
            "get_window_extent": lambda self, r, bb=bb: bb})(), "legend"))
        for t in leg.get_texts():
            texts.append((t.get_text()[:28], t, "legend"))


# ---------- 1. 文字两两相交(交集需 >1 pt × >1 pt,排除字形轴承相触) ----------
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        (la, aa, ta), (lb, ab, tb) = texts[i], texts[j]
        if ta == "legend" and tb == "legend":
            continue                   # 图例框与其内部文字同源
        single = (ta == "ax-text" and tb == "ax-text"
                  and len(la) == 1 and len(lb) == 1)
        ba, bb_ = _bbox(aa, fig), _bbox(ab, fig)
        ow = min(ba.x1, bb_.x1) - max(ba.x0, bb_.x0)
        oh = min(ba.y1, bb_.y1) - max(ba.y0, bb_.y0)
        # 序列条单字母网格:matplotlib 单字符 bbox 含轴承留白(实测比字形
        # 墨迹宽 ~40%),墨迹级净空由下方 cell-width 检查保证,bbox 阈值放宽
        thr = 3.5 if single else 1.0
        if ow > thr and oh > 1.0:      # pt
            issues.append(f"TEXT-TEXT overlap: '{la}' <-> '{lb}' "
                          f"({ow:.1f}x{oh:.1f}pt)")

# c 序列条:格宽须容纳 7 pt 字形墨迹(~0.045 in)+ ≥7 mil 白边
_cax = [ax for ax in fig.axes
        if any(t.get_text() == "native" for t in ax.texts)]
if _cax:
    _p = _cax[0].get_position()
    _span = _cax[0].get_xlim()[1] - _cax[0].get_xlim()[0]
    _cell = _p.width * fig.get_figwidth() / _span
    print(f"[c] cell width {_cell*1000:.1f} mil (7 pt glyph ≈ 45 mil)")
    if _cell < 0.052:
        issues.append(f"[c] cell width {_cell*1000:.0f} mil too narrow "
                      f"for 7 pt letters")

# ---------- 2. 文字 vs 图像 bbox ----------
img_axes = []
for ax in fig.axes:
    if ax.get_images():
        img_axes.append(ax)
for lab, a, ta in texts:
    if ta in ("tick", "ax-xlab", "ax-ylab"):
        continue  # 统计图轴外装饰,不入图像检查
    ba = _bbox(a, fig)
    for ax in img_axes:
        bi = ax.get_window_extent(r)
        if ba.overlaps(bi):
            issues.append(f"TEXT-IMAGE overlap: '{lab}' on image axes")
        else:
            # 净空检查:上下方向最小间距 ≥2 pt
            gap = min(abs(ba.y0 - bi.y1), abs(bi.y0 - ba.y1))
            if gap < 2.0 and (ba.x1 > bi.x0 and ba.x0 < bi.x1):
                issues.append(
                    f"TEXT-IMAGE clearance {gap:.1f}pt < 2pt: '{lab}'")

# ---------- 3. 画廊行间净空 ----------
gals = [ax for ax in fig.axes
        if ax.get_images() and ax.get_title().count("\u00b7")]
gals.sort(key=lambda ax: (-ax.get_position().y1, ax.get_position().x0))
anns = {}
for ax in gals:
    t = ax.texts[0]
    anns[ax] = _bbox(t, fig)
for k in range(4):                       # 每列跨行比较
    col = [ax for ax in gals
           if abs(ax.get_position().x0 - gals[0].get_position().x0
                  + (k * 0)) < 1]        # 占位,按列分组见下
by_col = {}
for ax in gals:
    by_col.setdefault(round(ax.get_position().x0, 4), []).append(ax)
for col in by_col.values():
    col.sort(key=lambda ax: -ax.get_position().y1)
    for u, v in zip(col, col[1:]):       # u 在上一行
        gap_mm = (anns[u].y0 - _bbox(v.title, fig).y1) / DPI * 25.4
        if gap_mm < 3.0:
            issues.append(f"GALLERY row clearance {gap_mm:.2f}mm < 3mm "
                          f"({v.get_title()[:12]})")

# ---------- 4. 数据空间:图例/注记 vs 数据 ----------
axd = [ax for ax in fig.axes if ax.get_xlabel()
       .startswith("\u0394 log")][0]
leg = axd.get_legend()
lb = leg.get_window_extent(r)
inv = axd.transData.inverted()
(lx0, ly0), (lx1, ly1) = inv.transform(((lb.x0, lb.y0), (lb.x1, lb.y1)))
bins = np.linspace(0.0, 4.0, 41)
hmax = max(np.histogram(C.D_APT, bins=bins)[0].max(),
           np.histogram(C.D_UTX, bins=bins)[0].max())
sel = (bins[:-1] < lx1) & (bins[:-1] + 0.1 > lx0)
h_leg = max(np.histogram(C.D_APT, bins=bins)[0][sel].max(),
            np.histogram(C.D_UTX, bins=bins)[0][sel].max())
ylim = axd.get_ylim()
print(f"[d] legend x∈[{lx0:.2f},{lx1:.2f}] y<={ly1:.1f}; "
      f"max bar under legend {h_leg}; ylim {ylim}")
if ly0 < hmax and h_leg > ly0:
    issues.append(f"[d] legend overlaps bars ({h_leg} > {ly0:.1f})")
for ln in axd.lines:                   # 竖直参考线不得穿过图例框
    xd = inv.transform((ln.get_xdata()[0], 0))[0]
    if lx0 - 1e-9 <= xd <= lx1 + 1e-9:
        issues.append(f"[d] reference line at x={xd:.2f} crosses legend")

axe = [ax for ax in fig.axes
       if ax.get_xlabel() == "Spearman \u03c1"][0]
txt = [t for t in axe.texts if t.get_text().startswith("mean")][0]
tb = txt.get_window_extent(r)
(tx0, ty0), (tx1, ty1) = inv_axe = axe.transData.inverted().transform(
    ((tb.x0, tb.y0), (tb.x1, tb.y1)))
for i, oi in enumerate(np.argsort(C.FAM_RHO)):
    y = i
    if (C.FAM_RHO[oi] >= tx0 and C.FAM_RHO[oi] <= tx1
            and y >= ty0 - 0.9 and y <= ty1 + 0.9):
        issues.append(f"[e] mean note overlaps family {i} "
                      f"(rho={C.FAM_RHO[oi]:.2f})")

axf = [ax for ax in fig.axes
       if ax.get_xlabel().startswith("Screening")][0]
lf = axf.get_legend().get_window_extent(r)
invf = axf.transData.inverted()
(fx0, fy0), (fx1, fy1) = invf.transform(((lf.x0, lf.y0), (lf.x1, lf.y1)))
worst = 0.0
for key in ("INF", "logP", "consensus", "oracle", "random"):
    for b in [0.01, 0.05, 0.10, 0.25, 0.50]:
        c_ = C.CS[key][f"budget_{b:.2f}"]
        x = 100 * b
        hi = 100 * c_["ci95"][1]
        if np.log10(x) >= np.log10(fx0) - 1e-9 and x <= fx1 * 1.001:
            worst = max(worst, hi)
print(f"[f] legend x∈[{fx0:.1f},{fx1:.1f}] bottom y={fy0:.2f}; "
      f"max CI top under legend {worst:.2f}")
if worst >= fy0:
    issues.append(f"[f] legend overlaps errorbars ({worst:.2f} >= {fy0:.2f})")

# ---------- 5. 面板字母对齐 ----------
letters = {t.get_text(): t for t in fig.texts if t.get_text() in "abcdef"}
rows = [("a",), ("b", "c"), ("d", "e", "f")]
for row in rows:
    ys = [_bbox(letters[c_], fig).y0 for c_ in row]
    if max(ys) - min(ys) > 0.5 / 25.4 * DPI:
        issues.append(f"LETTER row misalign {row}: "
                      f"{[f'{y:.1f}' for y in ys]}")
cols = {"a": 0, "b": 0, "d": 0, "c": 1, "e": 1, "f": 2}
for colx in (0, 1, 2):
    xs = [_bbox(letters[c_], fig).x0 for c_ in cols if cols[c_] == colx]
    if max(xs) - min(xs) > 0.5 / 25.4 * DPI:
        issues.append(f"LETTER col misalign {colx}: "
                      f"{[f'{x:.1f}' for x in xs]}")

# ---------- 6. axes 行内对齐 ----------
# b1/b2(两幅渲染 axes)须同尺寸;c 占满整个 b/c 行(顶 = ROW2_TOP,
# 高 = ROW2_H);d/e/f 共用 ROW3 几何,顶/底须一致
b_axes = [ax for ax in fig.axes if ax.get_images()
          and ("8HZD" in ax.get_title() or "pocket" in ax.get_title())]
c_ax = [ax for ax in fig.axes
        if any(t.get_text() == "native" for t in ax.texts)]
tol = 0.5 / 25.4                       # 0.5 mm(fig fraction 由 in 换算)
if len(b_axes) == 2:
    p0, p1 = (a.get_position() for a in b_axes)
    if (abs(p0.y0 - p1.y0) > tol / C.FIG_H_IN
            or abs(p0.y1 - p1.y1) > tol / C.FIG_H_IN):
        issues.append(f"AXES b1/b2 vertical misalign "
                      f"{p0.y0:.4f}/{p1.y0:.4f}")
else:
    issues.append(f"AXES b1/b2 not found ({len(b_axes)})")
if len(c_ax) == 1:
    p = c_ax[0].get_position()
    if abs(p.y1 - (1 - C.ROW2_TOP_IN / C.FIG_H_IN)) > tol / C.FIG_H_IN:
        issues.append("AXES c top not at ROW2_TOP")
    if abs(p.height - C.ROW2_H_IN / C.FIG_H_IN) > tol / C.FIG_H_IN:
        issues.append("AXES c height != ROW2_H")
else:
    issues.append(f"AXES c not found ({len(c_ax)})")
row3 = [ax for ax in fig.axes
        if abs(ax.get_position().y0 - (1 - (C.ROW3_TOP_IN + C.ROW3_H_IN)
                                       / C.FIG_H_IN)) < 0.01]
for name, row in (("d/e/f", row3),):
    tops = [round(a.get_position().y1, 4) for a in row]
    bots = [round(a.get_position().y0, 4) for a in row]
    if len(set(tops)) > 1 or len(set(bots)) > 1:
        issues.append(f"AXES row {name} misalign tops={set(tops)} "
                      f"bottoms={set(bots)}")

# ---------- 7. 文字颜色/粗细 ----------
for t in fig.texts + [x for ax in fig.axes for x in ax.texts]:
    if t.get_text().strip() in list("abcdef"):
        continue                          # 面板字母惯例粗体
    c_ = t.get_color()
    if c_ != "black":
        issues.append(f"COLOR not black: '{t.get_text()[:20]}' {c_}")
    if t.get_fontweight() not in ("normal", None):
        issues.append(f"BOLD text: '{t.get_text()[:20]}'")

# ---------- 8. 字号清单 ----------
sizes = {}
for lab, a, ta in texts:
    try:
        s = a.get_fontsize()
    except AttributeError:
        continue
    sizes.setdefault(ta, set()).add(round(s, 1))
for leg_owner in (axd, axf):
    for t in leg_owner.get_legend().get_texts():
        sizes.setdefault("legend", set()).add(round(t.get_fontsize(), 1))
print("[fonts]", {k: sorted(v) for k, v in sizes.items()})

print("=" * 60)
if issues:
    print(f"AUDIT FAIL: {len(issues)} issue(s)")
    for s in issues:
        print(" -", s)
    sys.exit(1)
print("AUDIT PASS: no overlap, letters/axes aligned, all-black regular text")
