# -*- coding: utf-8 -*-
"""Fig 1 素材数据准备:1U8D 链 A 的真实二级结构与几何图数据。

示例链选取:1U8D 链 A(鸟嘌呤核糖开关,配体 HPA,67 nt)。RNAfold 预测
的 MFE 结构为三向连接(18 对,27% 配对),视觉上多茎带环,比单发夹结构
更适合作为"预测二级结构"示意;同一条链同时提供骨架、单核苷酸、配体口袋
与几何图素材,保持图内示例一致。

步骤(全部数据驱动,不自绘):
  1. 从 data/rcsb_pdb/1U8D.pdb 抽链 A 的 RNA 序列与 C4' 坐标;
  2. 调用 ViennaRNA(RNAfold 预测 MFE 二级结构,RNAplot -o svg 出
     专业二维布局),解析 SVG 得每个核苷酸的布局坐标与配对表;
  3. 用该布局渲染圆圈风格二级结构图 assets/fig1_ss.png
     (白圈黑边节点、黑色细骨架、陶土色配对连线 —— 连线缩短到圈边缘,
     保证茎区配对横档可见;不标字母 —— 67 nt 在图内尺寸下字母不可读,
     圆点布局是国际通用的紧凑画法);
  4. 由 C4' 坐标计算 kNN(k=15,与训练配置一致)边,连同配对边写入
     assets/fig1_1u8d.json,供 fig1_pymol.py 渲染真实几何图。

运行:python figures/fig1/fig1_assets_data.py
"""
import json
import os
import re
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PDB = os.path.join(ROOT, "data", "rcsb_pdb", "1U8D.pdb")
CHAIN = "A"
ASSETS = os.path.join(HERE, "assets")
K_NEIGHBORS = 15          # 与 tools/PACoAptamer/train.py 一致

RES1 = {"A": "A", "U": "U", "G": "G", "C": "C"}


def read_chain_a():
    """返回 (resi 列表, 序列, C4' 坐标 Nx3)。"""
    resis, resn, c4 = [], {}, {}
    with open(PDB) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != CHAIN:
                continue
            rn = line[17:20].strip()
            if rn not in RES1:
                continue
            i = int(line[22:26])
            if i not in resn:
                resis.append(i)
                resn[i] = rn
            if line[12:16].strip() == "C4'":
                c4[i] = [float(line[30:38]), float(line[38:46]),
                         float(line[46:54])]
    resis.sort()
    seq = "".join(RES1[resn[i]] for i in resis)
    coords = np.array([c4[i] for i in resis], dtype=np.float64)
    return resis, seq, coords


def run_rnafold(seq):
    """RNAfold 得 dbn;RNAplot -o svg 得布局,解析坐标与配对(0-based)。"""
    tmp = tempfile.mkdtemp(prefix="fig1_ss_")
    fa = os.path.join(tmp, "seq.fa")
    with open(fa, "w") as f:
        f.write(">example\n%s\n" % seq)
    out = subprocess.run(["RNAfold", "--noPS"], stdin=open(fa),
                         capture_output=True, text=True, check=True).stdout
    lines = out.strip().splitlines()
    dbn = next(l.split()[0] for l in lines
               if l and set(l.split()[0]) <= set("()."))
    subprocess.run(["RNAplot", "-o", "svg"], input="\n".join(lines),
                   capture_output=True, text=True, check=True, cwd=tmp)
    svg = open(os.path.join(tmp, "example_ss.svg")).read()
    pts = [(float(x), float(y)) for x, y in
           re.findall(r'<text class="nucleotide" x="(-?[\d.]+)" '
                      r'y="(-?[\d.]+)">', svg)]
    pairs = sorted((int(i) - 1, int(j) - 1) for i, j in
                   re.findall(r'<line class="basepairs" id="(\d+),(\d+)"',
                              svg))
    assert len(pts) == len(seq), (len(pts), len(seq))
    return dbn, pairs, pts


def render_ss_png(pts, pairs, path):
    """按 RNAplot 布局画圆圈风格二级结构(透明底 PNG)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    xy = np.array(pts)
    xy[:, 1] *= -1                       # SVG y 向下 → 数学坐标
    lo, hi = xy.min(0), xy.max(0)
    span = (hi - lo).max()
    pad = 0.10 * span
    r = 0.022 * span                     # 节点半径
    # 图内放置尺寸约 0.42 in 高:素材按 ~1.9 in 渲染(约 4.5× 缩小),
    # 线宽按最终 ~0.5–0.7pt 反推
    fig = plt.figure(figsize=((hi[0] - lo[0] + 2 * pad) / 115,
                              (hi[1] - lo[1] + 2 * pad) / 115))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal"); ax.set_axis_off()
    n = len(xy)
    for i in range(n - 1):               # 骨架(黑色细线)
        ax.plot([xy[i, 0], xy[i + 1, 0]], [xy[i, 1], xy[i + 1, 1]],
                color="black", lw=2.0, zorder=2, solid_capstyle="round")
    for i, j in pairs:                   # 配对(陶土横档,缩短到圈边缘)
        p, q = xy[i], xy[j]
        u = q - p
        u /= np.linalg.norm(u)
        p, q = p + r * u, q - r * u
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#C08B7C", lw=3.0,
                zorder=4, solid_capstyle="round")
    for x, y in xy:                      # 白圈黑边节点
        ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=2.0,
                            zorder=3))
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    print("saved", path)


def knn_pairs(coords, k):
    n = len(coords)
    out = []
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        for j in np.argsort(d)[1:k + 1]:
            if j > i:
                out.append((i, int(j)))
    return sorted(set(out))


def main():
    os.makedirs(ASSETS, exist_ok=True)
    resis, seq, coords = read_chain_a()
    print("chain", CHAIN + ":", len(seq), "nt:", seq)
    dbn, pairs, pts = run_rnafold(seq)
    print("dbn:", dbn, "| pairs:", len(pairs))
    render_ss_png(pts, pairs, os.path.join(ASSETS, "fig1_ss.png"))
    knn = knn_pairs(coords, K_NEIGHBORS)
    with open(os.path.join(ASSETS, "fig1_1u8d.json"), "w") as f:
        json.dump({"pdb": "1U8D", "chain": CHAIN, "seq": seq, "dbn": dbn,
                   "resis": resis, "pairs_ss": pairs, "pairs_knn": knn}, f)
    print("kNN edges:", len(knn), "-> assets/fig1_1u8d.json")


if __name__ == "__main__":
    main()
