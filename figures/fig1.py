"""Fig 1 入口包装:实际代码在 figures/fig1/ 下按子图拆分。"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fig1"))
runpy.run_path(os.path.join(HERE, "fig1", "fig1_assemble.py"),
               run_name="__main__")
