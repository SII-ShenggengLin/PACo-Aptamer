# -*- coding: utf-8 -*-
"""SI Fig. | Fingerprint robustness(Tanimoto 稳健性三指纹)。

三个 mini-panel(Morgan / MACCS keys / RDKit topological):
87 条 zero-shot 链,逐链 Δ recovery t01(PACo-Aptamer − RhoDesign,%)vs
配体对 247 个训练配体的最大 Tanimoto 相似度;OLS 虚线 + Spearman ρ/P。
面板标题统一"名称 (参数, 位数)":MACCS = 166 结构键 + bit0 = 167 bit;
RDKit topological = RDKFingerprint 默认参数(minPath=1, maxPath=7,
fpSize=2048),位数见 data/106_tanimoto_robust.py。坐标轴只留单行短标签,
"对 247 个训练配体"与 Δ 方向(PACo-Aptamer − RhoDesign)由 SI 图注说明。
数据源 data/survey/106_tanimoto_robust.json;Morgan 列逐位断言等于
data/survey/93_tanimoto_zs.json(max_abs_diff = 0)。
校验:Δt01 ρ = −0.401 / −0.341 / −0.345(MACCS/topo P ≤ 1.2×10⁻³)。
单独运行:python figures/si/si_fingerprint.py → figures/out/si/si_fingerprint.{pdf,png}
"""
import numpy as np
from scipy import stats

import si_common as C
from fig_style import panel_label

FPS = [("morgan", "Morgan (radius 2, 2048 bit)"),
       ("maccs", "MACCS keys (167 bit)"),
       ("rdkit_topo", "RDKit topological (2048 bit)")]
EXPECT = {"morgan": (-0.4007, 1.203e-04), "maccs": (-0.3406, 1.246e-03),
          "rdkit_topo": (-0.3448, 1.074e-03)}


def main():
    d = C.load("106_tanimoto_robust.json")
    t93 = C.load("93_tanimoto_zs.json")
    s93 = {s["stem"]: s["max_tanimoto"] for s in t93["scatter"]}
    pc = d["per_chain"]
    # Morgan 与 93 逐位一致
    for stem, v in pc.items():
        assert abs(v["max_tanimoto_morgan"] - s93[stem]) < 1e-12, stem
    for fp, (er, ep) in EXPECT.items():
        sp = d["spearman"][fp]["sim_vs_delta_t01"]
        assert abs(sp["rho"] - er) < 5e-4 and abs(sp["p"] - ep) < 1e-5, fp

    fig, gs = C.make_canvas(58, 1, 3, wspace=0.34, left=0.065, right=0.995,
                            top=0.88, bottom=0.18)
    for i, (fp, lab) in enumerate(FPS):
        ax = fig.add_subplot(gs[0, i])
        panel_label(ax, "abc"[i])
        x = np.array([v[f"max_tanimoto_{fp}"] for v in pc.values()])
        y = np.array([100 * v["delta_t01_PACo_minus_RhoDesign"]
                      for v in pc.values()])
        ax.scatter(x, y, s=5.0, color=C.PACO, lw=0, alpha=0.75, zorder=3)
        lr = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, lr.slope * xs + lr.intercept, ls="--", lw=0.9,
                color="#7A7672", zorder=2)
        ax.axhline(0, color="#7A7672", lw=0.5, zorder=1)
        sp = d["spearman"][fp]["sim_vs_delta_t01"]
        ax.text(0.03, 0.04,
                f"Spearman ρ = {sp['rho']:.3f}\n{C.pfmt_sci(sp['p'])}, "
                f"n = {len(x)}",
                transform=ax.transAxes, fontsize=5.2, va="bottom",
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))
        ax.set_title(lab)
        ax.set_xlabel("Maximum Tanimoto similarity")
        if i == 0:
            ax.set_ylabel("Δ recovery, t = 0.1 mean (%)")
        ax.set_ylim(-14, 24)
    C.save_si(fig, "fingerprint")


if __name__ == "__main__":
    main()
