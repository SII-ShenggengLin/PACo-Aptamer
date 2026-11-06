# -*- coding: utf-8 -*-
"""Fig 1a | 架构总览:条件输入 → 编码 → 联合注意力 → 自回归解码。

示例链 1U8D 链 A(鸟嘌呤核糖开关–hypoxanthine;RNAfold 预测三向连接)。
四个栏框带顶部加粗标题(无填充);栏内模块框一律白底黑框;主题浅蓝只用于
特征条与结构渲染,陶土橙仅作配体点缀。解码列为单列主流程(前缀 → decoder
layer ×3 → softmax → 采样 → 完整序列字母方格,右侧自回归反馈);栏间连接
为贴框缘的实线箭头。
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import fig1_common as C
from fig1_common import fbox, img_place, opath, txt, featbar

FS = C.FS_BOX
SM = C.FS_SM
TINY = C.FS_TINY

# 栏框(顶 12.96,底 0.36;标题嵌入框顶、无填充带)。解码列为单列,
# 释放的宽度分给前三列;纵向按加高后的画布整体放宽、上下边距对称。
COLS = ((0.35, 8.30, "Conditioning inputs"), (9.35, 8.30, "Encoders"),
        (18.35, 7.60, "Joint RNA–ligand attention"),
        (26.65, 6.00, "Autoregressive decoder"))


def draw(ax):
    # v1.36 列标题带加宽(框高 12.72→12.86、bh 0.86→1.00):框顶上移
    # 占用画布顶部余量,带底位置不变(列内内容一律不动),标题上下
    # 留白从 0.28 加宽到 0.35 单位,四列标题仍严格同线
    for x, w, t in COLS:
        C.panel_frame(ax, x, 0.36, w, 12.86, t, fs=C.FS_HEAD, bh=1.00)

    _inputs(ax)
    _encoders(ax)
    _joint(ax)
    _decoder(ax)

    # ---- 栏间实线箭头(起止贴框缘) ----
    for y in (11.01, 7.22, 2.62):        # inputs → encoders
        opath(ax, [(8.65, y), (9.35, y)], lw=0.8)
    for y in (10.09, 1.73):              # encoders → joint attention
        opath(ax, [(17.65, y), (18.35, y)], lw=0.8)
    opath(ax, [(25.95, 6.72), (26.65, 6.72)], lw=0.8)   # joint → decoder


def _inputs(ax):
    # ---- RNA 骨架 + 单核苷酸 9 原子 ----
    fbox(ax, 0.57, 9.79, 7.86, 2.43)
    img_place(ax, f"{C.ASSETS}/fig1_bb.png", 2.70, 11.42, h=1.36)
    img_place(ax, f"{C.ASSETS}/fig1_nt.png", 6.00, 11.42, h=1.36)
    txt(ax, 2.70, 10.35, "RNA backbone", fs=SM)
    txt(ax, 6.00, 10.35, "9 atoms / nt", fs=SM)

    # ---- 预测二级结构 → 配对矩阵 ----
    fbox(ax, 0.57, 5.195, 7.86, 4.05)
    txt(ax, 4.50, 8.71, "Predicted secondary structure", fs=FS)
    img_place(ax, f"{C.ASSETS}/fig1_ss.png", 2.591, 6.90, h=2.55)
    opath(ax, [(4.531, 6.90), (5.369, 6.90)], lw=0.8)
    pairs_ss, _, resis, seq = C.load_pairs()
    C.pairmap_glyph(ax, 6.809, 6.90, 0.040, pairs_ss, resis, seq)

    # ---- 配体口袋(背景卡通 + 口袋残基球棍 + 配体球) ----
    fbox(ax, 0.57, 0.58, 7.86, 4.07)
    txt(ax, 4.50, 4.26, "Ligand pocket atoms", fs=FS)
    img_place(ax, f"{C.ASSETS}/fig1_pocket.png", 4.50, 2.70, h=2.52)
    txt(ax, 4.50, 0.94, "Pocket residues + ligand sticks", fs=SM)


def _encoders(ax):
    # ---- GVP 编码器(几何图 → GVP layer → 节点嵌入) ----
    # 内容组整体下沉 0.49:原先贴着标题,框底留出 1.1 单位空带;
    # 下沉后标题下方与框底留白相等,encoders→joint 箭头(y=10.09)
    # 恰好对准双通道输出条的中点
    fbox(ax, 9.57, 7.96, 7.86, 4.26)
    # 框内标题一律置于框顶与首个内容元素之间居中(与 joint 列标题同款)
    txt(ax, 13.50, 11.60, "GVP encoder ×3", fs=FS)
    img_place(ax, f"{C.ASSETS}/fig1_graph.png", 11.54, 10.03, h=1.90)
    txt(ax, 11.54, 8.68, "Geometric graph", fs=SM)
    # GVP layer 盒右移 0.28,几何图→盒、盒→输出条两段箭头随之拉长
    opath(ax, [(13.19, 10.03), (13.78, 10.03)], lw=0.8)
    fbox(ax, 13.78, 9.48, 2.20, 1.10)
    txt(ax, 14.88, 10.25, "GVP layer", fs=SM)
    txt(ax, 14.88, 9.81, "×3", fs=SM)
    # 输出:标量/向量双通道条。v1.32 改为两条独立水平箭头:两条输出条
    # 的中心高度(10.48/9.58)都落在 GVP 框右缘高度范围内,各自从框缘
    # 直射对应条即可 —— 旧画法两肘形箭头共享框缘出发的干线与同一条
    # 竖线,视觉上重叠成一根线再分叉
    opath(ax, [(15.98, 10.48), (16.52, 10.48)], lw=0.8)
    opath(ax, [(15.98, 9.58), (16.52, 9.58)], lw=0.8)
    featbar(ax, 16.52, 10.13, 0.30, 0.70, fc=C.C_FEAT)
    featbar(ax, 16.52, 9.23, 0.30, 0.70, fc=C.C_VEC)
    txt(ax, 16.96, 10.48, "$\\hat{s}$", fs=SM, ha="left")
    txt(ax, 16.96, 9.58, "$\\hat{v}$", fs=SM, ha="left")

    # ---- 配体原子嵌入(竖直流,整组居中;与上方 GVP 框同宽对齐) ----
    # v1.44 整组下移 0.28:原布局自标题向下排布,余量全落在框底(内
    # 边距 0.92,而面板 a 其他模块框为 0.17-0.53),内容偏上、底部空
    # 带过大(用户反馈);下移后框底内边距 0.64,与其他框同级,组内
    # 间距(标题-首框 0.60、段间箭头 0.40)保持不变
    fbox(ax, 9.57, 0.58, 7.86, 6.86)
    txt(ax, 13.50, 6.56, "Ligand atom embedding", fs=FS)
    fbox(ax, 10.90, 5.24, 5.20, 0.72)
    txt(ax, 13.50, 5.60, "Ligand heavy atoms", fs=SM)
    opath(ax, [(12.30, 5.24), (12.30, 4.84)], lw=0.8)
    opath(ax, [(14.70, 5.24), (14.70, 4.84)], lw=0.8)
    fbox(ax, 9.98, 3.72, 3.40, 1.12)
    txt(ax, 11.68, 4.49, "Element", fs=SM)
    txt(ax, 11.68, 4.05, "one-hot (5)", fs=SM)
    fbox(ax, 13.62, 3.72, 3.40, 1.12)
    txt(ax, 15.32, 4.49, "Distance to RNA", fs=SM)
    txt(ax, 15.32, 4.05, "RBF ×16", fs=SM)
    opath(ax, [(12.30, 3.72), (12.30, 3.32)], lw=0.8)
    opath(ax, [(14.70, 3.72), (14.70, 3.32)], lw=0.8)
    fbox(ax, 10.90, 2.60, 5.20, 0.72)
    txt(ax, 13.50, 2.96, "2-layer MLP → 512-d", fs=SM)
    opath(ax, [(13.50, 2.60), (13.50, 2.24)], lw=0.8)
    # v1.34 输出条下的 "Ligand atom tokens" 标注删除:与框标题
    # "Ligand atom embedding" 近重复;条本身保留,与联合注意力列的
    # 箭头(y=1.73 对条中心)不变,"tokens" 语义由联合列标题承接
    featbar(ax, 13.25, 1.22, 0.50, 1.02, fc=C.C_VEC)


def _joint(ax):
    # v1.31 列框底 0.90→0.73:下方让出的空间与注释/图例整体下移一起,
    # 供两个模块框加高(1.35→1.50 与 1.00→1.25)
    fbox(ax, 18.90, 0.73, 6.50, 11.39)
    # 标题下移到框顶(12.12)与注意力矩阵上缘(11.08)之间居中
    txt(ax, 22.15, 11.60, "RNA + ligand tokens", fs=FS)

    # 联合注意力矩阵:上左 RNA 块(主题蓝),下右配体块(浅陶土)
    n_rna, n_lig, cell = 4, 3, 0.64
    n = n_rna + n_lig
    cx = 22.15
    x0, y1 = cx - n * cell / 2, 11.08
    for i in range(n):
        for j in range(n):
            in_rna = i < n_rna and j < n_rna
            in_lig = i >= n_rna and j >= n_rna
            fc = C.C_VEC if in_rna else ("#E8CFC4" if in_lig else C.C_WHITE)
            ax.add_patch(Rectangle((x0 + j * cell, y1 - (i + 1) * cell),
                                   cell, cell, fc=fc, ec="white", lw=0.5,
                                   zorder=2))
    ax.plot([x0, x0 + n * cell], [y1 - n_rna * cell] * 2, color=C.C_TEXT,
            lw=1.0, zorder=3)
    ax.plot([x0 + n_rna * cell] * 2, [y1 - n * cell, y1], color=C.C_TEXT,
            lw=1.0, zorder=3)
    ax.add_patch(Rectangle((x0, y1 - n * cell), n * cell, n * cell,
                           fc="none", ec=C.C_EDGE, lw=0.9, zorder=3))
    # v1.61 矩阵 RNA 块的 "RNA" 由白色改深色(C_TEXT,用户反馈):
    # 与下右配体块的 "Lig" 及全图文字一致;#A9C4D9 中蓝上深字对比足够
    txt(ax, x0 + 2 * cell, y1 - 2 * cell, "RNA", fs=SM, color=C.C_TEXT,
        weight="bold", z=4)
    txt(ax, x0 + 5.5 * cell, y1 - 5.5 * cell, "Lig", fs=SM, color=C.C_TEXT,
        weight="bold", z=4)

    opath(ax, [(cx, 6.60), (cx, 6.05)], lw=0.8)
    # 两个模块框与上方注意力矩阵同宽(左右边缘对齐,4.48 = 7×0.64);
    # v1.31 加高:上框 1.35→1.50、下框 1.00→1.25,两框间箭头相应变短
    # (0.55→0.40),文字在加高后的框内重新居中
    fbox(ax, 19.91, 4.55, 4.48, 1.50)
    txt(ax, cx, 5.56, "Joint attention ×3", fs=FS)
    txt(ax, cx, 5.04, "512-d · 4 heads", fs=SM)
    opath(ax, [(cx, 4.47), (cx, 4.07)], lw=0.8)
    fbox(ax, 19.91, 2.82, 4.48, 1.25)
    txt(ax, cx, 3.44, "$p(\\hat{s}_t \\mid s_{<t}, \\mathrm{RNA}, "
                      "\\mathrm{ligand})$", fs=SM)
    txt(ax, cx, 2.33, "Joint states passed to", fs=C.FS_TXT)
    txt(ax, cx, 1.85, "decoder cross-attention", fs=C.FS_TXT)
    # 图例(与注释同步下移,注释-图例间距与图例-框底间距保持原有节奏)
    ax.add_patch(Rectangle((20.89, 1.19), 0.24, 0.24, fc=C.C_VEC,
                           ec=C.C_EDGE, lw=0.5, zorder=3))
    txt(ax, 21.23, 1.31, "RNA", fs=TINY, ha="left")
    ax.add_patch(Rectangle((22.25, 1.19), 0.24, 0.24, fc="#E8CFC4",
                           ec=C.C_EDGE, lw=0.5, zorder=3))
    txt(ax, 22.59, 1.31, "ligand", fs=TINY, ha="left")


def _decoder(ax):
    # 单列竖直流(输入 → decoder layer ×3 → softmax → 采样 → 完整序列),
    # 整列铺满高度;右侧为自回归反馈回路
    # v1.57 竖排 "Autoregressive feedback" 标签删除(用户反馈;反馈语
    # 义由回路线本身与图题"feeding each sample back"承载);回路右移
    # 贴框(竖线 31.40→32.25,与框右缘净空 0.40),四个流程框再加宽
    # 并居中于列框中心:4.10→4.50(26.95–31.05 → 27.40–31.90),
    # 中心 29.00→29.65(v1.45 为右侧竖排标签让位左偏 0.65,标签既除
    # 回位正中),右缘距回路 0.35 不变;底部字母行(27.36–31.94,中
    # 心 29.65)恰与框组近乎同宽对齐
    # v1.45 四个流程框曾加宽 3.50→4.10(用户反馈左右太窄)
    cx = 29.65
    # v1.59 四框等距、四箭等长(用户反馈):外包络不动(底 1.28、输
    # 出箭尖 10.96,底部回路挂点 1.755 与顶部字母行都不动),框尺寸
    # 不动,仅移框位 —— 框间间隙统一 (10.96-1.28-5.50)/4 = 1.045;
    # 四支箭统一 v1.46 收笔惯例:框顶上方 0.05 起笔 + 平头端帽(框顶
    # 缘线全程连续),尖端抵下一框底缘,墨长一律 0.995(此前四支为
    # 1.27/0.91+0.12 空档/0.93/0.90,且起笔惯例不一)
    # v1.58 首框文字并作一行(Partial sequence s<t>,用户反馈),框
    # 上下收窄 1.35→0.95(与 Softmax 0.95 / Sample 0.90 单行框同级)
    fbox(ax, 27.40, 1.28, 4.50, 0.95)
    txt(ax, cx, 1.755, "Partial sequence $s_{<t}$", fs=SM)
    opath(ax, [(cx, 2.28), (cx, 3.275)], lw=0.8, capstyle="butt")
    # 单框 + "×3" 标注(与其他模块框样式一致,不用叠框暗示层数)
    fbox(ax, 27.40, 3.275, 4.50, 2.70)
    txt(ax, cx, 5.355, "Decoder layer ×3", fs=SM)
    txt(ax, cx, 4.625, "Masked self-attn ·", fs=SM)
    txt(ax, cx, 3.895, "Cross-attn · FFN", fs=SM)
    opath(ax, [(cx, 6.025), (cx, 7.02)], lw=0.8, capstyle="butt")
    fbox(ax, 27.40, 7.02, 4.50, 0.95)
    txt(ax, cx, 7.495, "Softmax $p(\\hat{s}_t)$", fs=SM)
    opath(ax, [(cx, 8.02), (cx, 9.015)], lw=0.8, capstyle="butt")
    fbox(ax, 27.40, 9.015, 4.50, 0.90)
    txt(ax, cx, 9.465, "Sample $\\hat{s}_t$", fs=SM)
    # 输出箭头(→ 字母行):与其他三支同惯例同长(v1.46 的 0.05 收笔
    # +平头端帽由此推广到全列)
    opath(ax, [(cx, 9.965), (cx, 10.96)], lw=0.8, capstyle="butt")
    # 输出:完整设计序列(与 d 中前缀同款字母方格);行与标签以列框
    # 中心(29.65)对齐 —— v1.57 起流程框组中心亦为 29.65,同轴
    # v1.60 字母方格放大(用户反馈:字母太挤):0.34×0.35 → 0.40×0.45,
    # 间距 0.385→0.44(白缝 ~0.04),行宽 4.575→5.24(27.03–32.27),
    # 仍以列心 29.65 居中、两侧距列框各 0.38(略宽于流程框组);行底
    # 11.06 不变(与输出箭头尖 10.96 净空 0.10),行顶 11.51 与 RNA
    # sequence 标签(11.93)净空 ~0.30;6pt 字母在 0.40 宽格内左右各
    # 留 ~0.06(原 ~0.03,几乎顶满)
    seq = "AUGGCUAGCUAA"
    for k, ch in enumerate(seq):
        x0 = 27.03 + k * 0.44
        ax.add_patch(Rectangle((x0, 11.06), 0.40, 0.45,
                               fc=C.NT_COLORS[ch], ec="white", lw=0.5,
                               zorder=3))
        txt(ax, x0 + 0.20, 11.285, ch, fs=SM, color="white", weight="bold",
            z=4)
    txt(ax, 29.65, 11.93, "RNA sequence", fs=SM)
    # 自回归反馈:采样结果送回前缀(v1.57 随框右缘外移 31.05→31.90、
    # 竖线 31.40→32.25;竖排标签已删;v1.59 上挂点随 Sample 框新中心
    # 9.56→9.465,下挂点 1.755 不变)
    opath(ax, [(31.90, 9.465), (32.25, 9.465), (32.25, 1.755), (31.90, 1.755)],
          lw=0.8)


if __name__ == "__main__":
    fig = plt.figure(figsize=(C.FIG_W_IN, C.FIG_H_IN), dpi=C.DPI)
    ax = fig.add_axes(C.RECT_A)
    ax.set_xlim(C.A_LIM[0], C.A_LIM[1])
    ax.set_ylim(C.A_LIM[2], C.A_LIM[3])
    ax.axis("off")
    draw(ax)
    C.save(fig, "fig1_a")
