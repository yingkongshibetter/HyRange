"""Exact certification of the residual components (Tables II, III and VI).

Runs the module hyrange.certify as a script; its docstring lists the commands.

usage: python experiments/exact_certification.py bench <dataset> <preserved|reduced>
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.certify", run_name="__main__", alter_sys=True)
