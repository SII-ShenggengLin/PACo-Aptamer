# -*- coding: utf-8 -*-
"""Fig 6 | 案例研究与回顾性验证 —— 组合入口。

各子图绘图代码在 fig6_*.py(可独立运行、与拼合版共用
fig6_common.make_canvas 的同一画布几何,字号/配色天然一致):
  a 12 案画廊:回折-天然叠加结构网格(需先 fig6_gallery_pymol.py 渲染)
  b 8HZD_A 结构全景 + 口袋放大(需先 fig6_pymol.py 渲染)
  c 逐位恢复序列条(native + 5 方法最优设计,界面位橙标 + 界面恢复率列)
  d 回顾性审计(ΔlogP 双库分布 + 家族内 Kd 相关森林图)
  e 计算筛选预算曲线(INF/logP/共识/oracle)
(旧 panel c 三链柱状已并入画廊格内注记,fig6_c.py 保留存档不再调用)

运行:python fig6_assemble.py → out/fig6_case_audit.{pdf,png}
"""
import fig6_a
import fig6_b
import fig6_d
import fig6_e
import fig6_gallery
import fig6_common as C
from fig_style import save


def build():
    fig, gs = C.make_canvas()
    fig6_gallery.draw(fig)
    fig6_a.draw(fig, gs)
    fig6_b.draw(fig, gs)
    fig6_d.draw(fig, gs)
    fig6_e.draw(fig, gs)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, "fig6_case_audit")
    import numpy as np
    print(f"[check] apt n={len(C.D_APT)} mean={np.mean(C.D_APT):.3f} | "
          f"utx n={len(C.D_UTX)} mean={np.mean(C.D_UTX):.3f} | "
          f"merged {np.mean(C.D_ALL):.3f} frac<0 {np.mean(C.D_ALL < 0):.3f} | "
          f"families {len(C.FAM_RHO)} mean_rho {np.mean(C.FAM_RHO):.3f} | "
          f"iface {C.IFACE}")
