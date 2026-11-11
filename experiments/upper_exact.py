"""Exact certification in the upper-bound mode (Table II, row Upper; row HyRange of Table V at t = 5).

Runs the module hyrange.upper_exact as a script; its docstring lists the commands.

usage: python experiments/upper_exact.py job bench <dataset> <mode> | job export <source> 5 | queue bench_pres bench_red export_t5
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.upper_exact", run_name="__main__", alter_sys=True)
