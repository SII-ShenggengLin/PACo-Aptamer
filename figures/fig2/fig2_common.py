# -*- coding: utf-8 -*-
"""Fig 2 公共模块:统一画布几何 + 数据准备 + 跨面板共用绘图函数。

所有子图脚本(fig2_a..i)与组合脚本(fig2_assemble)共用:
- make_canvas():同一张 183×159 mm 画布、同一 GridSpec(3,3) 满格,
  上行 a/b/c 恢复率对比、中行 d(ECDF)/e(hexbin)/f(链内直方图),
  下行 g(筛选曲线)/h(长度箱线)/i(配体家族优势),
  子图单独运行与拼合后的物理尺寸完全一致 → 字号/配色天然统一。
- 数据:主表/逐链恢复率/ens2 池置信度数组(链级 ρ + bootstrap 95% CI)。
- draw_h2h():b、c 两面板共用的 head-to-head 散点。
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_style import (  # noqa: E402
    MM, SUR, PACO, apply_style, load)

# ---------- 画布几何(所有面板唯一事实来源) ----------
FIG_W_IN = 183 * MM
FIG_H_IN = 159 * MM
GRID_KW = dict(hspace=0.60, wspace=0.21,
               height_ratios=[1.13, 0.93, 1.11],
               left=0.05, right=0.98, top=0.965, bottom=0.065)


def make_canvas():
    """返回 (fig, gs):与拼合版完全相同的 183×159mm 画布。"""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = gridspec.GridSpec(3, 3, figure=fig, **GRID_KW)
    return fig, gs


# ================= 数据 =================
# 63_main_table_gfix / 81_wilcoxon_gfix:gRNAde 列 C/G 解码修正(2026-10-16),
# 其余方法与旧 btest_v2_main_table / 81_wilcoxon 逐值一致
_mt = load("63_main_table_gfix.json")
main = _mt["table"]
p04 = load("77_p04_confidence_recovery.json")

# 逐链恢复率一律取主表 per_chain(canonical 口径,与正文数字同源;
# 81 号统计亦基于该来源,保证散点、胜率、ECDF 三者一致)
pc_paco = _mt["per_chain"]["PACoAptamer"]
pc = {m: {k: v["recovery_best"] for k, v in _mt["per_chain"][m].items()}
      for m in ("RhoDesign", "gRNAde", "NA-MPNN", "AlignIF")}
pc["PACo-Aptamer"] = {k: v["recovery_best"] for k, v in pc_paco.items()}
STEMS = sorted(pc_paco)
chain_len = {s: pc_paco[s]["len"] for s in STEMS}

# ens2 池一次性读入(链级散点 + 链内相关 + g 面板富集曲线共用,77 号同口径:
# x=全池 300 条 mean logP,y=文件内 recovery_t01_mean,N 位计分母)
pool_dir = os.path.join(SUR, "77_ens2_pool_btest")
conf_x, conf_y, within = [], [], []
pools_lp, pools_rc = [], []   # 每链 (logP 数组, 逐设计恢复率数组)
for s in STEMS:
    pf = os.path.join(pool_dir, f"{s}.json")
    if not os.path.exists(pf):
        continue
    d = json.load(open(pf))
    nt, pool, lps = d["native"], d["pool"], d["logp"]
    conf_x.append(float(np.mean(lps)))
    conf_y.append(100 * d["recovery_t01_mean"])
    rc = np.array([sum(1 for a, b in zip(sq, nt) if a == b) / len(nt)
                   for sq in pool])
    pools_lp.append(np.asarray(lps, dtype=float)); pools_rc.append(rc)
    if len(lps) >= 30 and np.std(lps) > 0:
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
    """b/c 两面板共用:逐链 head-to-head 散点(PACo vs opp)。

    面板内不写统计文字,含义交给图例:蓝点 = PACo 占优的链,
    灰点 = 对手占优,虚线 = 二者相等;胜率与 Wilcoxon 数字由正文/图注承载。
    """
    from matplotlib.lines import Line2D
    x = np.array([pc[opp][s] for s in STEMS]) * 100
    y = np.array([pc["PACo-Aptamer"][s] for s in STEMS]) * 100
    lim = (5, 100)
    ax.plot(lim, lim, color="#7A7672", lw=0.6, ls="--", zorder=1)
    win = float(np.mean(y > x))
    ax.scatter(x[y > x], y[y > x], s=3.5, c=PACO, alpha=0.55, lw=0, zorder=2)
    ax.scatter(x[y <= x], y[y <= x], s=3.5, c="#B7B0A2", alpha=0.55, lw=0,
               zorder=2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=PACO,
                              markeredgecolor="none", markersize=3,
                              alpha=0.8, label="PACo-Aptamer higher"),
                       Line2D([], [], color="#7A7672", lw=0.6, ls="--",
                              label="equal recovery")],
              loc="lower right", fontsize=5.0, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{opp} recovery (%)")
    ax.set_ylabel("PACo-Aptamer recovery (%)")
    return win
