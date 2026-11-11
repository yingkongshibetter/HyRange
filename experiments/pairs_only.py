"""Certified share from the pair counts alone (Table II, row W; paragraph Dense hypergraphs).

Runs the module hyrange.pairs_only as a script; its docstring lists the commands.

usage: python experiments/pairs_only.py
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.pairs_only", run_name="__main__", alter_sys=True)
