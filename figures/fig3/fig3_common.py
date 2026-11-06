# -*- coding: utf-8 -*-
"""Fig 3 公共模块:统一画布几何 + 数据准备 + 三联面板共用绘图函数。

所有子图脚本(fig3_a..h)与组合脚本(fig3_assemble)共用:
- make_canvas():与 Fig. 2 相同的 183×159 mm 画布、GridSpec(3,6),
  行 0/1 各三联(a–c zero-shot 93 链、d–f independent 235 链,列严格对齐),
  行 2 g(h,0:3 预测骨架柱状)+ h(scTM 散点)各半宽;
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:69_alayer_gfix(per_chain + views;gRNAde C/G 解码修正)、81_wilcoxon、
  splits_v2;预测骨架数据经 figures/ed/ed_common 单一事实来源引入。
- draw_bars / draw_h2h / draw_hist:a–c 与 d–f 两组三联共用
  (zero_shot_targets 与 independent_test 各一套,布局完全相同)。
  面板规范与 Fig. 2 一致:面板内不写统计文字(数字由正文/图注承载),
  柱状图图例样块用中性近黑色、横坐标方法全名居中 20°、数字一律不加粗;
  head-to-head 无底纹,优势分布为正负分色直方图(与 Fig. 2f 同款),
  胜率/配对检验入图注。
"""
import json
import os
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import (  # noqa: E402
    MM, METHOD_COLOR, METHOD_ORDER, NEG, ROOT, PACO, apply_style, load,
    panel_label)

# ---------- 画布几何(所有面板唯一事实来源;与 Fig. 2 相同的网格节奏) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 159 * MM
# wspace 0.40 + tighten_ylabels():相邻面板 y 轴标签组(约 8.6 mm)完整落在
# 列间隙内并两侧留白 —— 程序化边界检测验证(标签组内边距收紧贡献约
# 1.6 mm/边界,不再侵入左邻面板的轴线带)
# hspace 0.40:行间空隙约 15 mm(容得下上一行 20° 旋转方法名 ~8 mm 与
# 下一行面板字母 ~4 mm),面板相应加高;此前 0.60 的 ~20 mm 被作者判定偏空
GRID_KW = dict(hspace=0.40, wspace=0.42,
               left=0.078, right=0.98, top=0.965, bottom=0.065)


def tighten_ylabels(fig):
    """列间呼吸感:y 刻度签 pad 3.5→1.5 pt、y 轴标签 labelpad 4→1.5 pt。

    每条列边界的标签组收窄约 1.6 mm,全部转化为面板之间的白色空隙;
    只作用于本图,不改全局 fig_style。
    """
    for ax in fig.axes:
        ax.tick_params(axis="y", pad=1.5)
        lbl = ax.yaxis.label.get_text()
        if lbl:
            ax.set_ylabel(lbl, labelpad=1.5)


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×159mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(3, 6, figure=fig, **GRID_KW)
    return fig, gs


# ================= 数据 =================
al = load("69_alayer_gfix.json")   # gRNAde 列 C/G 解码修正;其余同 alayer_v2_retrained
pc = al["per_chain"]
wil = load("81_wilcoxon.json")
sp = json.load(open(os.path.join(ROOT, "data", "splits_v2", "splits_v2.json")))
views = al["views"]
MKEY = {m: ("PACoAptamer" if m == "PACo-Aptamer" else m) for m in METHOD_ORDER}

# ---------- 预测骨架数据(g,h 面板;经 ed_common 单一事实来源) ----------
sys.path.insert(0, os.path.join(ROOT, "figures", "ed"))
import ed_common as ED  # noqa: E402  94/100–103/105 predbb、86 scTM、63 晶体口径

STEMS193 = ED.ZS105["full193"]["chains"]  # 193 链(五方法 predbb 链集一致)
PRED_ORDER = ED.PRED_ORDER


def view_arrays(view):
    """返回 (stems, x=RhoDesign%, y=PACo%),按 stem 排序的逐链恢复率。"""
    stems = sorted(sp[view])
    x = np.array([pc[s]["RhoDesign"]["recovery_best"] for s in stems]) * 100
    y = np.array([pc[s]["PACoAptamer"]["recovery_best"] for s in stems]) * 100
    return stems, x, y


# ================= 三联面板共用绘图 =================
def draw_bars(ax, view, letter):
    """双口径柱状(best 实色 / t01 半透明)+ 中性样块图例(与 Fig. 2a 一致)。

    柱色=方法(x 轴全名自明),口径只靠色调区分;Wilcoxon 数字由正文/图注承载。
    """
    panel_label(ax, letter)
    vb = [100 * views[view][MKEY[m]]["best"] for m in METHOD_ORDER]
    vt = [100 * views[view][MKEY[m]]["t01"] for m in METHOD_ORDER]
    xs = np.arange(len(METHOD_ORDER)); w = 0.38
    ax.bar(xs - w / 2, vb, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           ec="white", lw=0.3)
    ax.bar(xs + w / 2, vt, w, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           alpha=0.45, ec="white", lw=0.3)
    for x, v in list(zip(xs - w / 2, vb)) + list(zip(xs + w / 2, vt)):
        ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=4.8)
    ax.legend(handles=[
        Patch(fc="0.25", ec="none", label="best sample per chain (solid)"),
        Patch(fc="0.25", alpha=0.45, ec="none",
              label="t = 0.1 mean (translucent)")],
        loc="upper right", fontsize=5.5, handlelength=1.4,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    ax.set_xticks(xs)
    ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="center", va="top")
    ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.set_ylabel("Sequence recovery (%)")
    ax.set_ylim(0, 110)
    tighten_ylabels(ax.figure)


def draw_h2h(ax, view, letter):
    """head-to-head 逐链散点(PACo-Aptamer vs RhoDesign)。

    与 Fig. 2b,c 相同规范:无底纹、面板内不写统计文字,蓝点 = PACo-Aptamer
    占优的链,灰点 = RhoDesign 占优,虚线 = 二者相等;不锁 equal-aspect,
    保证与同行柱状/小提琴面板顶底边对齐。
    """
    panel_label(ax, letter)
    _, x, y = view_arrays(view)
    lim = (40, 100)
    ax.plot(lim, lim, color="#7A7672", lw=0.6, ls="--", zorder=1)
    ax.scatter(x[y > x], y[y > x], s=3.5, c=PACO, alpha=0.55, lw=0, zorder=2)
    ax.scatter(x[y <= x], y[y <= x], s=3.5, c="#B7B0A2", alpha=0.55, lw=0,
               zorder=2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=PACO,
                              markeredgecolor="none", markersize=3,
                              alpha=0.8, label="PACo-Aptamer higher"),
                       Line2D([], [], color="#7A7672", lw=0.6, ls="--",
                              label="equal recovery")],
              loc="lower right", fontsize=5.5, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("RhoDesign recovery (%)")
    ax.set_ylabel("PACo-Aptamer recovery (%)")
    tighten_ylabels(ax.figure)


def draw_hist(ax, view, letter):
    """逐链恢复率优势(PACo-Aptamer − RhoDesign)直方图,正负分色(与 Fig. 2f 同款)。

    取代旧版单一大提琴(作者反馈太丑,与 Fig. 2f v1.63 同一裁决):
    正 = 石板蓝、负 = 灰玫红、恰好相等 = 零点处灰色窄条(equal recovery),
    2.5 个百分点分箱、零点为分箱边界;零线虚线、黑色均值线;数字在图注。
    """
    panel_label(ax, letter)
    _, x, y = view_arrays(view)
    dv = (y - x)
    edges = np.arange(-20, 20.01, 2.5)
    centers = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(dv, bins=edges)
    share = 100 * counts / len(dv)
    tie_share = 100 * np.mean(dv == 0)
    # 0 所在分箱 [0, 2.5) 拆开:平局质量改为零点处的独立灰条,不并入"更高"侧
    share[centers == 1.25] -= tie_share
    pos = centers > 0
    ax.bar(centers[pos], share[pos], width=2.3, color=PACO, ec="white",
           lw=0.3)
    ax.bar(centers[~pos], share[~pos], width=2.3, color=NEG, ec="white",
           lw=0.3)
    ax.bar([0], [tie_share], width=0.9, color="#B7B0A2", ec="white", lw=0.3)
    ax.axvline(0, color="#7A7672", lw=0.6, ls=":")
    ax.axvline(np.mean(dv), color="black", lw=0.9)
    ax.legend(handles=[
        Patch(fc=PACO, ec="none", label="PACo-Aptamer higher"),
        Patch(fc=NEG, ec="none", label="RhoDesign higher"),
        Patch(fc="#B7B0A2", ec="none", label="equal recovery")],
        loc="upper left", fontsize=5.5, handlelength=1.2,
        handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 30)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("Advantage over RhoDesign (%)")
    ax.set_ylabel("Chains (%)")
    tighten_ylabels(ax.figure)
