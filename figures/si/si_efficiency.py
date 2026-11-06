# -*- coding: utf-8 -*-
"""SI Fig. | Inference efficiency(效率四联,6 条探针链/方法)。

(a) 每链 runtime(s,ens2 口径,min–max 误差棒);(b) 每样本耗时(ms,log);
(c) 吞吐(samples/s,log);(d) 峰值显存(GiB,均值 + min–max 区间)。
方法名用正文全称(PACo-Aptamer);探针集差异与指示性比较的说明在
SI 图题与 Supplementary Table 8 注记中,不画在图内。
数据源 data/survey/99_efficiency_{PACoAptamer,AlignIF,gRNAde,NA-MPNN,
RhoDesign}.json,逐方法汇总后与 104_ed_tables/ed_table4_efficiency.csv
断言一致(PACo 2.77s/9.2ms/149.3/0.6–2.1GB;AlignIF 474.6ms;
gRNAde 21.3ms;NA-MPNN 355.9ms;RhoDesign 154.7ms)。
单独运行:python figures/si/si_efficiency.py → figures/out/si/si_efficiency.{pdf,png}
"""
import numpy as np

import si_common as C
from fig_style import panel_label


def _metrics():
    m = {}
    d = C.load("99_efficiency_PACoAptamer.json")["per_chain"]
    v = list(d.values())
    m["PACoAptamer"] = dict(
        wall=np.mean([x["ens2"]["wall_s"] for x in v]),
        ms=1e3 * np.mean([x["ens2"]["s_per_sample"] for x in v]),
        thr=np.mean([x["ens2"]["samples_per_s"] for x in v]),
        mem=np.mean([x["ens2"]["peak_mem_mb"] for x in v]),
        mem_lo=min(x["ens2"]["peak_mem_mb"] for x in v),
        mem_hi=max(x["ens2"]["peak_mem_mb"] for x in v),
        wall_lo=min(x["ens2"]["wall_s"] for x in v),
        wall_hi=max(x["ens2"]["wall_s"] for x in v))
    d = C.load("99_efficiency_AlignIF.json")["per_chain"]
    v = list(d.values())
    m["AlignIF"] = dict(
        wall=np.mean([x["wall_s"] for x in v]),
        ms=1e3 * np.mean([x["s_per_sample"] for x in v]),
        thr=np.mean([x["samples_per_s"] for x in v]),
        mem=np.mean([x["peak_mem_mb_approx"] for x in v]),
        mem_lo=min(x["peak_mem_mb_approx"] for x in v),
        mem_hi=max(x["peak_mem_mb_approx"] for x in v),
        wall_lo=min(x["wall_s"] for x in v), wall_hi=max(x["wall_s"] for x in v))
    d = C.load("99_efficiency_gRNAde.json")["records"]
    gw = sum(x["wall_s"] for x in d)
    gn = sum(3 * x["n_samples_per_temp"] for x in d)
    m["gRNAde"] = dict(
        wall=np.mean([x["wall_s"] for x in d]),
        ms=1e3 * gw / gn,
        thr=gn / gw,
        mem=np.mean([x["gpu_mem_peak_mib"] for x in d]),
        mem_lo=min(x["gpu_mem_peak_mib"] for x in d),
        mem_hi=max(x["gpu_mem_peak_mib"] for x in d),
        wall_lo=min(x["wall_s"] for x in d), wall_hi=max(x["wall_s"] for x in d))
    d = C.load("99_efficiency_NA-MPNN.json")["records"]
    nw = sum(x["wall_s"] for x in d)
    m["NA-MPNN"] = dict(
        wall=np.mean([x["wall_s"] for x in d]),
        ms=1e3 * nw / (10 * len(d)),
        thr=10 * len(d) / nw,
        mem=np.mean([x["gpu_mem_peak_mib_polling"] for x in d]),
        mem_lo=min(x["gpu_mem_peak_mib_polling"] for x in d),
        mem_hi=max(x["gpu_mem_peak_mib_polling"] for x in d),
        wall_lo=min(x["wall_s"] for x in d), wall_hi=max(x["wall_s"] for x in d))
    d = C.load("99_efficiency_RhoDesign.json")["probes"]
    v = list(d.values())
    nsamp = lambda x: 3 * x["n_per_temp"]
    rw = sum(x["wall_time_s"] for x in v)
    rn = sum(nsamp(x) for x in v)
    m["RhoDesign"] = dict(
        wall=np.mean([x["wall_time_s"] for x in v]),
        ms=1e3 * rw / rn,
        thr=rn / rw,
        mem=np.mean([x["peak_mem_MiB"] for x in v]),
        mem_lo=min(x["peak_mem_MiB"] for x in v),
        mem_hi=max(x["peak_mem_MiB"] for x in v),
        wall_lo=min(x["wall_time_s"] for x in v),
        wall_hi=max(x["wall_time_s"] for x in v))
    return m


def _assert_vs_csv(m):
    rows = {r["method"]: r for r in
            C.load_csv("104_ed_tables/ed_table4_efficiency.csv")}
    for k, r in rows.items():
        assert abs(m[k]["wall"] - float(r["wall_s_per_chain_mean"])) < 5e-3, k
        assert abs(m[k]["ms"] - float(r["ms_per_sample"])) < 0.06, k
        assert abs(m[k]["thr"] - float(r["samples_per_s"])) < 0.05, k
        assert abs(m[k]["mem"] - float(r["peak_mem_mb_mean"])) < 0.6, k
        assert abs(m[k]["mem_lo"] - float(r["peak_mem_mb_min"])) < 0.15, k
        assert abs(m[k]["mem_hi"] - float(r["peak_mem_mb_max"])) < 0.15, k


def _name(m):
    """完整方法名(与正文一致,PACo 用全称 PACo-Aptamer)。"""
    return "PACo-Aptamer" if m == "PACoAptamer" else m


def _panel(ax, letter, vals, order, ylab, fmt, log=False, ranges=None,
           scale=1.0):
    panel_label(ax, letter)
    x = np.arange(len(order))
    v = [vals[m] * scale for m in order]
    kw = {}
    if ranges:
        lo = [(vals[m] - ranges[m][0]) * scale for m in order]
        hi = [(ranges[m][1] - vals[m]) * scale for m in order]
        kw = dict(yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.5))
    ax.bar(x, v, 0.62, color=[C.MCOLOR[m] for m in order], ec="white", lw=0.3,
           zorder=3, **kw)
    for xi, vi in zip(x, v):
        top = vi
        if ranges:
            top = ranges[order[int(xi)]][1] * scale
        ax.annotate(fmt.format(vi), (xi, top), xytext=(0, 2.2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.0, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([_name(m) for m in order], fontsize=5.0, rotation=20,
                       ha="center", va="top", color="black")
    ax.tick_params(labelcolor="black")
    ax.set_ylabel(ylab, color="black")
    if log:
        ax.set_yscale("log")
        hi_all = max((ranges[m][1] if ranges else v) * scale
                     for m, v in zip(order, [vals[m] for m in order]))
        ax.set_ylim(min(v) * 0.55, max(max(v), hi_all) * 2.6)
    else:
        hi_all = max((ranges[m][1] if ranges else v) * scale
                     for m, v in zip(order, [vals[m] for m in order]))
        ax.set_ylim(0, max(max(v), hi_all) * 1.18)


def main():
    m = _metrics()
    _assert_vs_csv(m)
    order = ["PACoAptamer", "gRNAde", "RhoDesign", "NA-MPNN", "AlignIF"]
    fig, gs = C.make_canvas(60, 1, 4, wspace=0.30, left=0.055, right=0.995,
                            top=0.87, bottom=0.18)
    _panel(fig.add_subplot(gs[0, 0]), "a",
           {k: v["wall"] for k, v in m.items()}, order,
           "Runtime per chain (s)", "{:.2f}",
           ranges={k: (v["wall_lo"], v["wall_hi"]) for k, v in m.items()})
    _panel(fig.add_subplot(gs[0, 1]), "b",
           {k: v["ms"] for k, v in m.items()}, order,
           "Time per sample (ms)", "{:.1f}", log=True)
    _panel(fig.add_subplot(gs[0, 2]), "c",
           {k: v["thr"] for k, v in m.items()}, order,
           "Throughput (samples/s)", "{:.1f}", log=True)
    _panel(fig.add_subplot(gs[0, 3]), "d",
           {k: v["mem"] / 1024 for k, v in m.items()}, order,
           "Peak GPU memory (GiB)", "{:.2f}",
           ranges={k: (v["mem_lo"] / 1024, v["mem_hi"] / 1024)
                   for k, v in m.items()})
    C.save_si(fig, "efficiency")


if __name__ == "__main__":
    main()
