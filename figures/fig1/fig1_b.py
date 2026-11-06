# -*- coding: utf-8 -*-
"""Fig 1b | GVP encoder 细节(对应 a 中同名模块)。

几何图节点携带标量(512)与向量(256)双通道;每层 GVP 中,标量经
Message MLP 更新,向量经无偏置线性变换后由 MLP 输出的 σ 门控缩放;
向量范数 ‖·‖ 汇入 MLP 输入,边特征(kNN 距离 + SS 配对)一并进入。
模块框白底黑框;标题带与外框齐边;输入箭头起于特征条/边特征框边缘,
输出箭头自框内 Message MLP / Linear 两框右缘平头引出,垂直穿过模块
框缘;σ 门控虚线全程走框外走廊,σ 圈居于两条输出线正中,上下两支
虚线箭头等长(共轴等腰三角头),不与框缘线或其他实线交叉重合。
坐标系与物理轴等比(圆圈不变形)。
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

import fig1_common as C
from fig1_common import fbox, featbar, opath, txt

FS = C.FS_BOX
SM = C.FS_SM
TINY = C.FS_TINY


def dash_varrow(ax, x, y_from, y_tip, hl=0.19, hw=0.066, lw=0.9,
                color=C.C_EDGE):
    """竖直虚线箭头:Line2D 虚杆 + 显式等腰三角头。

    v1.53:matplotlib 对带 dash 的 FancyArrowPatch 在竖直方向会把
    三角头剪切 ~0.02 单位(下指右歪、上指左歪,底边偏离轴线,读作
    "半边残缺";像素实测于门控①②与 Norm 三箭)。改为手工多边形,
    底边与尖端严格共轴;hl/hw 取自 ms=4.5 头的实渲染比例(长 0.19、
    宽 0.132),外观与其他箭头一致。
    """
    d = -1 if y_tip < y_from else 1
    y_base = y_tip - d * hl
    ax.add_line(Line2D([x, x], [y_from, y_base], lw=lw, color=color,
                       zorder=3, linestyle=(0, (3, 2)),
                       solid_capstyle="round"))
    ax.add_patch(Polygon([(x, y_tip), (x - hw, y_base), (x + hw, y_base)],
                         closed=True, fc=color, ec="none", zorder=3))


def draw(ax):
    # v1.35 标题带加高(bh 0.72→0.84、面板框 9.38→9.52),内容整体
    # 下移 0.15:标题上下留白从 ~0.21 单位加宽到 ~0.27/0.45
    # v1.48 面板加宽(9.09→9.805,xlim 9.571→10.285,承接 d 收窄释放
    # 的行宽):输出走廊原是全图最挤处(s′/v′ 标签距框缘仅 0.16),且
    # 两条输出箭头起点 7.05 在模块框内部、横穿框缘 x=7.33,读作"箭
    # 头与线框重合"(用户反馈)。输出箭头改为起于框缘 7.33 并配平头
    # 端帽(伸出而不压线),特征条/标签随走廊外移 0.50(8.35→8.85、
    # 9.07→9.57),左右留白 0.26/0.34 趋衡;其余元素不动
    C.panel_frame(ax, 0.24, 0.24, 9.805, 9.52, "GVP encoder ×3",
                  fs=C.FS_HEAD, bh=0.84)

    # ---- 输入(左列) ----
    # v1.55 输入标签删去维度数字(Scalar 512→Scalar、Vector 256→
    # Vector,用户反馈):维度信息由图题 "scalar (512) and vector (256)
    # channels" 承载,图内标签只留通道名,与右侧 New scalar/New vector
    # 对仗更工整
    featbar(ax, 1.00, 6.40, 0.50, 1.50, fc=C.C_FEAT)
    txt(ax, 1.25, 6.10, "Scalar", fs=SM)
    featbar(ax, 1.00, 3.70, 0.50, 1.50, fc=C.C_VEC)
    txt(ax, 1.25, 3.40, "Vector", fs=SM)
    fbox(ax, 0.50, 1.40, 2.30, 0.55)
    txt(ax, 1.65, 1.675, "Edge features", fs=SM)

    # ---- GVP 层(中) ----
    # v1.33 输出侧重排:标量/向量改为两条等长的对称平行直箭头(各 1.30,
    # 与输入肘形的水平段同量级,也与 a 面板 GVP 框的双通道平行输出呼
    # 应);σ 圈改为旁路门控 —— 挂在向量线上方,短虚线箭头下指向量线,
    # 表达"σ 门控缩放输出向量"(与图题一致)。旧画法(σ 串联在向量路
    # 径上)迫使标量箭头独走 1.45 单位,比面板内其他箭头长一倍多,孤
    # 立不协调;v1.32 的框缘定位(右缘 7.33、圈离开框缘)保持不变
    fbox(ax, 3.30, 1.75, 4.03, 7.00)
    txt(ax, 5.315, 8.40, "GVP layer ×3", fs=FS)
    fbox(ax, 3.80, 6.05, 3.25, 0.85)
    txt(ax, 5.425, 6.475, "Message MLP", fs=SM)
    fbox(ax, 3.80, 3.05, 3.25, 0.85)
    txt(ax, 5.425, 3.475, "Linear (no bias)", fs=SM)
    # 向量范数上行汇入 MLP(不与其他线交叉);v1.53 改用 dash_varrow
    # (虚线 FancyArrowPatch 的竖直箭头三角会被剪切偏轴,见其 docstring)
    dash_varrow(ax, 5.60, 3.90, 6.03)
    txt(ax, 5.42, 4.95, "Norm", fs=TINY, ha="right")
    # σ 门控信号(旁路):标量输出线上黑点引出(与 c/d 残差引出点同一
    # 惯例)→ 虚线箭头下指入 σ 圈 → 圈底短虚线箭头再下指向量输出线
    # v1.53 两箭改用 dash_varrow(共轴三角,修虚线 patch 的剪切偏轴);
    # σ 圈移到两条输出线的正中(4.20→4.97),两支箭总长均为 1.22,
    # 圈顶/圈底到上/下线净空对称(各 1.26),尖端仍各留 0.04 停在线外
    # v1.51 上段改为箭头:原来是无头虚线,方向感只由下段承担
    # v1.50 箭头不再压框缘线:旧画法自 MLP 底部下行后右行,水平虚线段
    # 必然横穿模块框右缘(7.33),dash 与箭头三角压线读作"箭头与线框
    # 重合"(用户反馈);改在框外 s′ 线上取点引出,虚线不再穿任何实线
    ax.add_patch(Circle((8.05, 6.47), 0.05, fc="black", ec="none", zorder=6))
    dash_varrow(ax, 8.05, 6.47, 5.25)
    # v1.52 σ 与 Norm 由斜体改正体:两者是整幅图仅有的斜体文本,
    # 与其余标签不一致(用户反馈);字体族不变
    ax.add_patch(Circle((8.05, 4.97), 0.24, fc="white", ec="black", lw=0.7,
                        zorder=5))
    txt(ax, 8.05, 4.97, "σ", fs=C.FS_TXT, z=6)
    dash_varrow(ax, 8.05, 4.73, 3.51)
    # 输入箭头(起于特征条/边特征框边缘,止于模块框边缘)
    opath(ax, [(1.50, 7.15), (2.75, 7.15), (2.75, 6.47), (3.80, 6.47)])
    opath(ax, [(1.50, 4.45), (2.95, 4.45), (2.95, 3.47), (3.80, 3.47)])
    opath(ax, [(2.80, 1.67), (3.10, 1.67), (3.10, 6.30), (3.80, 6.30)])
    # 输出:标量/向量两条对称平行直箭头
    # v1.56 尾端延伸到框内两内框右缘(7.33→7.05,用户反馈):箭头真正
    # 自 Message MLP / Linear 引出(标量箭 y=6.47=MLP 框心、向量箭
    # y=3.47=Linear 框心),垂直穿过模块框缘 —— 与全图"正交穿越"惯
    # 例一致(v1.32);σ 门控已移出走廊(v1.50),此处穿越不再与门控
    # 虚线挤在一起。v1.48 曾把尾端收到框缘 7.33(平头端帽),当时走
    # 廊内 σ 肘形穿墙杂乱、穿越被读作压线;现恢复至内框,平头端帽保留
    # v1.54 删去条侧 s′/v′ 数学记号,改在条下方标注 New scalar /
    # New vector(与左列输入条的下置标签同一版式,条底下方 0.30、以
    # 条中心 9.10 居中;实测右缘 9.917,距面板框 0.128,左缘与 σ 圈
    # 净空 ~0.24;v1.55 起左列标签为 Scalar/Vector,左右对仗)
    opath(ax, [(7.05, 6.47), (8.85, 6.47)], capstyle="butt")
    opath(ax, [(7.05, 3.47), (8.85, 3.47)], capstyle="butt")
    featbar(ax, 8.85, 5.72, 0.50, 1.50, fc=C.C_FEAT)
    txt(ax, 9.10, 5.42, "New scalar", fs=SM)
    featbar(ax, 8.85, 2.72, 0.50, 1.50, fc=C.C_VEC)
    txt(ax, 9.10, 2.42, "New vector", fs=SM)

    txt(ax, 4.79, 0.75, "Message passing over kNN + SS-pair edges",
        fs=C.FS_TXT)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(C.FIG_W_IN, C.FIG_H_IN), dpi=C.DPI)
    ax = fig.add_axes([C.RECT_BX[0], C.RECT_BCD_Y[0],
                     C.RECT_BCD_WB, C.RECT_BCD_Y[1]])
    ax.set_xlim(C.D_LIM_B[0], C.D_LIM_B[1])
    ax.set_ylim(C.D_LIM_B[2], C.D_LIM_B[3])
    ax.axis("off")
    draw(ax)
    C.save(fig, "fig1_b")
