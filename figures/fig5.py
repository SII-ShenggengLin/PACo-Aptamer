"""Fig 5 入口包装:实际代码在 figures/fig5/ 下按子图拆分。"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fig5"))
runpy.run_path(os.path.join(HERE, "fig5", "fig5_assemble.py"),
               run_name="__main__")
