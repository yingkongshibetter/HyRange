"""Forcing in the upper-bound mode on the six benchmarks (paragraph Upper-bound mode, forcing alone).

Runs the module hyrange.upper as a script; its docstring lists the commands.

usage: python experiments/upper_forcing.py queue [--procs 4] | job <ds> <mode> | summary
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.upper", run_name="__main__", alter_sys=True)
