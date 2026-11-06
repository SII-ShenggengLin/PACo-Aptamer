# -*- coding: utf-8 -*-
"""Fig 1 入口包装:实际代码在 figures/fig1/ 下按子图拆分。

  fig1/fig1_a.py … fig1_f.py  各子图独立绘图脚本(可单独运行,字号配色一致)
  fig1/fig1_common.py         统一画布几何 + 共享元件(唯一事实来源)
  fig1/fig1_pymol.py          真实结构素材渲染(PyMOL 无头 → fig1/assets/)
  fig1/fig1_assemble.py       组合为整图(本包装等价于直接运行它)

本文件保留习惯入口:python fig1.py → out/fig1_overview.{pdf,png}
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fig1"))
runpy.run_path(os.path.join(HERE, "fig1", "fig1_assemble.py"),
               run_name="__main__")
