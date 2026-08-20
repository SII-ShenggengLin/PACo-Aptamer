# -*- coding: utf-8 -*-
"""Fig 6 入口包装:实际代码在 figures/fig6/ 下按子图拆分。

  fig6/fig6_pymol.py    PyMOL 结构渲染(egarps_env 独立运行)
  fig6/fig6_a.py … fig6_e.py  各子图独立绘图脚本(可单独运行,字号配色一致)
  fig6/fig6_assemble.py 组合为整图(本包装等价于直接运行它)

本文件保留习惯入口:python fig6.py → out/fig6_case_audit.{pdf,png}
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fig6"))
runpy.run_path(os.path.join(HERE, "fig6", "fig6_assemble.py"),
               run_name="__main__")
