# -*- coding: utf-8 -*-
"""Fig 2 公共模块:统一画布几何 + 数据准备 + 跨面板共用绘图函数。

所有子图脚本(fig2_a..i)与组合脚本(fig2_assemble)共用:
- make_canvas():同一张 183×175 mm 画布、同一 GridSpec(4,8),
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:主表/Wilcoxon/逐链恢复率/ens2 池置信度数组(链级 ρ + bootstrap 95% CI)。
- draw_h2h():d、e 两面板共用的 head-to-head 散点。
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import (  # noqa: E402
    MM, SUR, TACO, apply_style, load)

# ---------- 画布几何(所有面板唯一事实来源) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 175 * MM
GRID_KW = dict(hspace=0.68, wspace=1.35,
               height_ratios=[1.0, 1.3, 1.15, 1.2])


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×175mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(4, 8, figure=fig, **GRID_KW)
    return fig, gs


def pfmt_p(p):
    if p < 1e-4:
        m = int(np.floor(np.log10(p)))
        return f"P = {p/10**m:.1f} × 10$^{{{m}}}$"
    return f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"


# ================= 数据 =================
main = load("btest_v2_main_table.json")["table"]
wil = load("81_wilcoxon.json")
p04 = load("77_p04_confidence_recovery.json")

pc_taco = load("metrics_TACo_v2_ens2noiface_full.json")["per_chain"]
pc = {"TACo-Aptamer": {k: v["recovery_best"] for k, v in pc_taco.items()}}
for m, f, key in [("RhoDesign", "metrics_RhoDesign_btest_v2.json", "recovery_best_overall"),
                  ("gRNAde", "metrics_gRNAde_btest_v2.json", "recovery_best_overall"),
                  ("NA-MPNN", "metrics_NAMPNN_btest_v2.json", "recovery_best"),
                  ("AlignIF", "metrics_AlignIF_btest_v2.json", "recovery_best")]:
    pc[m] = {k: v[key] for k, v in load(f)["per_chain"].items()}
STEMS = sorted(pc_taco)
chain_len = {s: pc_taco[s]["len"] for s in STEMS}

# ens2 池一次性读入(链级散点 + 链内相关共用,77 号同口径:
# x=全池 300 条 mean logP,y=文件内 recovery_t01_mean,N 位计分母)
pool_dir = os.path.join(SUR, "77_ens2_pool_btest")
conf_x, conf_y, within = [], [], []
for s in STEMS:
    pf = os.path.join(pool_dir, f"{s}.json")
    if not os.path.exists(pf):
        continue
    d = json.load(open(pf))
    nt, pool, lps = d["native"], d["pool"], d["logp"]
    conf_x.append(float(np.mean(lps)))
    conf_y.append(100 * d["recovery_t01_mean"])
    if len(lps) >= 30 and np.std(lps) > 0:
        rc = [sum(1 for a, b in zip(sq, nt) if a == b) / len(nt) for sq in pool]
        if np.std(rc) > 0:
            r, _ = stats.spearmanr(lps, rc)
            if np.isfinite(r):
                within.append(float(r))
conf_x, conf_y, within = map(np.array, (conf_x, conf_y, within))
rho_chain, _ = stats.spearmanr(conf_x, conf_y)
# ρ 的 95% CI(bootstrap,链级重采样)
_rng = np.random.default_rng(0)
_rhos = []
for _ in range(2000):
    ii = _rng.integers(0, len(conf_x), len(conf_x))
    if np.std(conf_x[ii]) > 0 and np.std(conf_y[ii]) > 0:
        _rhos.append(stats.spearmanr(conf_x[ii], conf_y[ii]).statistic)
rho_ci = np.percentile(_rhos, [2.5, 97.5])


# ================= 共用绘图 =================
def draw_h2h(ax, opp):
    """d/e 两面板共用:逐链 head-to-head 散点(TACo vs opp)。"""
    x = np.array([pc[opp][s] for s in STEMS]) * 100
    y = np.array([pc["TACo-Aptamer"][s] for s in STEMS]) * 100
    lim = (5, 100)
    ax.fill_between([0, 100], [0, 100], [100, 100], color="#EDF2F7", zorder=0)
    ax.plot(lim, lim, color="#8C8C88", lw=0.6, ls="--", zorder=1)
    win = float(np.mean(y > x))
    ax.scatter(x[y <= x], y[y <= x], s=3.5, c=TACO, alpha=0.55, lw=0, zorder=2)
    ax.scatter(x[y > x], y[y > x], s=3.5, c="#B8B8B4", alpha=0.55, lw=0, zorder=2)
    wj = wil["B_test_477"][opp]["recovery_best"]
    ax.text(0.06, 0.94, (f"TACo-Aptamer wins {100*win:.0f}% of chains\n"
            f"Δ = +{100*wj['delta']:.1f} pp, {pfmt_p(wj['p_two_sided'])}"),
            transform=ax.transAxes, va="top", fontsize=5.4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{opp} recovery (%)")
    ax.set_ylabel("TACo-Aptamer recovery (%)")
    return win
