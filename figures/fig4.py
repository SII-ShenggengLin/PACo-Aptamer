# -*- coding: utf-8 -*-
"""Fig 4 入口包装:实际代码在 figures/fig4/ 下按子图拆分。

  fig4/fig4_a.py … fig4_i.py  各子图独立绘图脚本(可单独运行,字号配色一致)
  fig4/fig4_common.py          统一画布几何 + 数据准备(唯一事实来源)
  fig4/fig4_assemble.py        组合为整图(本包装等价于直接运行它)

本文件保留习惯入口:python fig4.py → out/fig4_ligand_conditioning.{pdf,png}
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fig4"))
runpy.run_path(os.path.join(HERE, "fig4", "fig4_assemble.py"),
               run_name="__main__")
