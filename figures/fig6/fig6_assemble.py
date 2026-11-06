# -*- coding: utf-8 -*-
"""Fig 6 | 案例研究与回顾性验证 —— 组合入口。

各子图绘图代码在 fig6_*.py(可独立运行、与拼合版共用 fig6_common.make_canvas
的同一画布几何与画廊列网格,字号/配色/对齐天然一致):
  a 12 案画廊:回折-天然叠加结构网格 + 下方居中图例(fig6_gallery.py,
    需先 fig6_gallery_pymol.py 渲染)
  b 8HZD_A 结构全景 + 口袋放大,对齐画廊前两列图像;每幅渲染上下各一行
    格题/注记文字(2026-10-23 起,渲染图内不嵌字;需先 fig6_pymol.py 渲染)
  c 逐位恢复序列条(native + 5 方法最优设计,界面恢复率列),跨画廊第 3–4 列
  d Δ log probability 双库分布(图例含合并均值虚线,左上)
  e 家族内 log probability–Kd 相关森林图(均值虚线旁文字注记,
    2026-10-23 自 d 拆出独立编号)
  f 计算筛选预算曲线(INF/log probability/平均/oracle/random,图例左上)
(旧 panel c 三链柱状已并入画廊格内注记,存档于 attic_fig6_v1/。)

运行:python fig6_assemble.py → out/fig6_case_audit.{pdf,png}
组装后可跑 python fig6_audit.py 做重叠/对齐/字号程序化审计。
"""
import fig6_b
import fig6_c
import fig6_d
import fig6_e
import fig6_f
import fig6_gallery
import fig6_common as C
from fig_style import save


def build():
    fig, L = C.make_canvas()
    fig6_gallery.draw(fig)
    fig6_b.draw(fig, L)
    fig6_c.draw(fig, L)
    fig6_d.draw(fig, L)
    fig6_e.draw(fig, L)
    fig6_f.draw(fig, L)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig6_case_audit", pad=0.06)   # 外白边 1.5 mm(tight-bbox 默认
                                             # 0.5 mm 过薄,出版裁切无余量)
    import numpy as np
    print(f"[check] apt n={len(C.D_APT)} mean={np.mean(C.D_APT):.3f} | "
          f"utx n={len(C.D_UTX)} mean={np.mean(C.D_UTX):.3f} | "
          f"merged {np.mean(C.D_ALL):.3f} frac<0 {np.mean(C.D_ALL < 0):.3f} | "
          f"families {len(C.FAM_RHO)} mean_rho {np.mean(C.FAM_RHO):.3f} | "
          f"iface {C.IFACE}")
