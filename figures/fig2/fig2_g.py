"""Fig 2g | logP 排序筛选收益曲线:保留前 f% 设计的平均恢复率 − 池均值。"""
import numpy as np

import fig2_common as C
from fig_style import PACO, panel_label, save

FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]

def _curve():
    means, los, his = [], [], []
    for f in FRACS:
        per = []
        for lp, rc in zip(C.pools_lp, C.pools_rc):
            order = np.argsort(-lp)
            k = max(1, int(len(lp) * f))
            per.append(rc[order[:k]].mean() - rc.mean())
        per = np.array(per)
        if f == 1.0:
            means.append(0.0); los.append(0.0); his.append(0.0)
            continue
        rng = np.random.RandomState(0)
        bs = [rng.choice(per, len(per), replace=True).mean()
              for _ in range(2000)]
        means.append(100 * per.mean())
        los.append(100 * np.percentile(bs, 2.5))
        his.append(100 * np.percentile(bs, 97.5))
    return np.array(means), np.array(los), np.array(his)

MEANS, LOS, HIS = _curve()

def draw(fig, gs):
    axg = fig.add_subplot(gs[2, 0])
    panel_label(axg, "g")
    xs = np.array(FRACS)
    axg.axhline(0, color="#7A7672", lw=0.6, ls=":")
    axg.fill_between(xs, LOS, HIS, color="#8FA6BB", alpha=0.18, lw=0,
                     zorder=1)
    axg.plot(xs, MEANS, "-o", color=PACO, lw=0.9, ms=2.6, zorder=3)
    for f, ha, dx in ((0.01, "left", 2), (0.10, "center", 0)):
        i = FRACS.index(f)
        axg.annotate(f"+{MEANS[i]:.2f}", (xs[i], MEANS[i]),
                     textcoords="offset points", xytext=(dx, 4),
                     ha=ha, fontsize=5.0, fontweight="bold")
    axg.set_xscale("log"); axg.minorticks_off()
    axg.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    axg.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    axg.set_xlim(0.008, 1.35)
    axg.set_ylim(-0.12, 2.35)
    axg.set_yticks([0, 1, 2])
    axg.set_xlabel("Designs kept by log-probability ranking (%)")
    axg.set_ylabel("Recovery − pool mean (%)")
    return axg

if __name__ == "__main__":
    fig, gs = C.make_canvas()
    draw(fig, gs)
    save(fig, "fig2_g")
