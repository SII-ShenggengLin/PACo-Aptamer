# -*- coding: utf-8 -*-
"""Fig 3 入口包装:实际代码在 figures/fig3/ 下按子图拆分。

  fig3/fig3_a.py … fig3_i.py  各子图独立绘图脚本(可单独运行,字号配色一致)
  fig3/fig3_common.py         统一画布几何 + 数据准备(唯一事实来源)
  fig3/fig3_assemble.py       组合为整图(本包装等价于直接运行它)

本文件保留习惯入口:python fig3.py → out/fig3_generalisation.{pdf,png}
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fig3"))
runpy.run_path(os.path.join(HERE, "fig3", "fig3_assemble.py"),
               run_name="__main__")
