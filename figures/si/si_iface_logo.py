"""SI Fig. | Interface consensus composition(界面位置序列 logo + 115 链汇总)。"""
import matplotlib.pyplot as plt
from matplotlib import gridspec

import si_common as C
from fig_style import panel_label

NT_COL = {"A": "#A3B18A", "U": "#B5838D", "G": "#C08B7C", "C": "#6B7B8D"}
ROW_ORDER = ["native", "PACo", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
ROW_LAB = {"native": "native", "PACo": "PACo-Aptamer", "RhoDesign": "RhoDesign",
           "gRNAde": "gRNAde", "NA-MPNN": "NA-MPNN", "AlignIF": "AlignIF"}
CASES = [("8HB1_A", "a"), ("8D5L_A", "b")]
YLO, YHI_PAD = -0.50, 0.10

def draw_logo(ax, case, letter):
    panel_label(ax, letter)
    d = C.load("interface_sequence_logos.json")["case_detail"][case]
    pos = sorted(map(int, d["per_method"]["PACo"]["per_pos"].keys()))
    npos, nrow = len(pos), len(ROW_ORDER)
    ylo, yhi = YLO, nrow + YHI_PAD

    fb = ax.get_position()
    ax_h_in = fb.height * ax.figure.get_figheight()
    ax_w_mm = fb.width * ax.figure.get_figwidth() * 25.4
    gutter_mm = 14.5
    gutter_units = gutter_mm * (npos + 0.45) / (ax_w_mm - gutter_mm)
    x_range = npos + 0.28 + gutter_units + 0.18
    pt_per_y = ax_h_in * 72.0 / (yhi - ylo)
    pt_per_x = ax_w_mm / 25.4 * 72.0 / x_range
    fs_full = min(0.97 * pt_per_y / 0.728, 0.94 * pt_per_x / 0.718)

    for r, m in enumerate(ROW_ORDER):
        y0 = nrow - 1 - r
        for xi, p in enumerate(pos):
            if m == "native":
                freqs = {d["native_seq"][p]: 1.0}
            else:
                freqs = d["per_method"][m]["per_pos"][str(p)]["freq"]
            base = 0.0
            for nt, f in sorted(freqs.items(), key=lambda kv: kv[1]):
                if f <= 0.05:
                    continue
                ax.text(xi + 0.5, y0 + base, nt, fontsize=f * fs_full,
                        fontweight="bold", color=NT_COL[nt],
                        ha="center", va="baseline")
                base += f
        ax.text(-0.18, y0 + 0.5, ROW_LAB[m], ha="right", va="center",
                fontsize=5.6, color="black")
    for xi, p in enumerate(pos):
        ax.text(xi + 0.5, -0.16, str(p), ha="center", va="top",
                rotation=90, fontsize=5.0, color="black")
    ax.set_xlim(-(gutter_units + 0.18), npos + 0.28)
    ax.set_ylim(ylo, yhi)
    ax.axis("off")
    ax.set_title(f"{case} ({d['n_iface']} interface positions)",
                 fontsize=6.0)

def draw_summary(ax):
    panel_label(ax, "c")
    s = C.load("interface_sequence_logos.json")["all115_summary"]
    order = ["PACo", "RhoDesign", "gRNAde", "NA-MPNN", "AlignIF"]
    xg = C.np.arange(len(order)); wg = 0.30
    v1 = [100 * s[m]["consensus_eq_native_micro"] for m in order]
    v2 = [100 * s[m]["match_frac_mean"] for m in order]
    ax.bar(xg - wg / 2, v1, wg * 0.9, color=[C.MCOLOR[m] for m in order],
           ec="white", lw=0.3, zorder=3)
    ax.bar(xg + wg / 2, v2, wg * 0.9, color=[C.MCOLOR[m] for m in order],
           alpha=0.42, ec="white", lw=0.3, zorder=3)
    for xi, v in zip(xg - wg / 2, v1):
        ax.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=5.2)
    for xi, v in zip(xg + wg / 2, v2):
        ax.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=5.2)
    ax.set_xticks(xg)
    ax.set_xticklabels([C.MLAB[m] for m in order], fontsize=5.6)
    ax.set_ylabel("Agreement with native (%)")
    ax.set_ylim(0, 76)
    ax.tick_params(axis="y", labelsize=5.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#7A7672", label="consensus = native (micro)"),
                       Patch(fc="#7A7672", alpha=0.45,
                             label="per-chain match fraction (macro)")],
              loc="upper right", fontsize=5.2)

def main():
    C.apply_style()
    fig = plt.figure(figsize=(183 * C.MM, 96 * C.MM))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.32, 1.0],
                           hspace=0.45, left=0.045, right=0.985,
                           top=0.90, bottom=0.115)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                              wspace=0.10)
    for i, (case, letter) in enumerate(CASES):
        draw_logo(fig.add_subplot(gs_top[0, i]), case, letter)
    draw_summary(fig.add_subplot(gs[1]))
    C.save_si(fig, "iface_logo")

if __name__ == "__main__":
    main()
