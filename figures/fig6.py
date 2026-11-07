"""Regenerate Figure 6 as figures/out/fig6_case_study.pdf and .png."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fig6"))
runpy.run_path(os.path.join(HERE, "fig6", "fig6_assemble.py"), run_name="__main__")
