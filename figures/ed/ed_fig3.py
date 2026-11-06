# -*- coding: utf-8 -*-
"""ED Fig. 3 | Designing from predicted backbones(预测骨架基准,193 链)——逐链细节。

单面板图(无 panel 字母):Δ(predbb − crystal) best 口径逐链分布小提琴
+ 抖动点 + 黑色均值线;均值数字标在各小提琴上方(ED4b 同款),
小提琴加宽(0.65)收窄间距; zs 交叉切片在主图 Fig. 3i。
x 轴用与方法名一致的完整称谓(PACo-Aptamer 等),全宽单面板空间充足,
水平居中摆放(主图窄面板才用 20° 旋转);Δ 记号不进图。
单独运行:python figures/ed/ed_fig3.py → figures/out/ed_fig3.{pdf,png}
"""
import numpy as np

import ed_common as C
from fig_style import METHOD_COLOR, save

STEMS = C.ZS105["full193"]["chains"]  # 193 链(五方法 predbb 链集一致)


def _delta_best(method):
    pre = C.predbb_recovery(method)
    cry = C.crystal_recovery(method)
    return np.array([100 * (pre[s][0] - cry[s]["recovery_best"]) for s in STEMS])


def draw(ax, rng):
    ms = C.PRED_ORDER
    dvs = [_delta_best(m) for m in ms]
    xg = np.arange(len(ms))
    vp = ax.violinplot(dvs, positions=xg, widths=0.65, showextrema=False)
    for b, m in zip(vp["bodies"], ms):
        b.set_facecolor(METHOD_COLOR[m])
        b.set_edgecolor("#7A7672")
        b.set_alpha(0.38)
        b.set_lw(0.5)
    for i, (dv, m) in enumerate(zip(dvs, ms)):
        ax.scatter(rng.normal(i, 0.08, len(dv)), dv, s=2.2,
                   c=METHOD_COLOR[m], alpha=0.5, lw=0, zorder=3)
        mu = dv.mean()
        ax.plot([i - 0.22, i + 0.22], [mu, mu], color="black", lw=0.9, zorder=4)
        ax.text(i, dv.max() + 3, f"{mu:+.1f}%", fontsize=5.0, ha="center",
                va="bottom", color="black")
    ax.axhline(0, color="#7A7672", lw=0.6, ls="--")
    ax.set_xticks(xg)
    ax.set_xticklabels(ms, fontsize=5.2)
    ax.set_xlim(-0.42, len(ms) - 0.58)
    ax.set_ylabel("Change in recovery, best sample\n(predicted − crystal, %)")
    ax.set_ylim(-75, 44)


def main():
    fig, gs = C.make_canvas(68, 1, 1,
                            left=0.062, right=0.99, top=0.93, bottom=0.15)
    rng = np.random.default_rng(0)
    draw(fig.add_subplot(gs[0, 0]), rng)
    save(fig, "ed_fig3")


if __name__ == "__main__":
    main()
