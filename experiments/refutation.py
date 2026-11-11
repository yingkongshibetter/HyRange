"""MARIOH + refutation (Table III, row MARIOH + Ref.; paragraph Refutation), on the six benchmarks.

Runs the module hyrange.refute as a script; its docstring lists the commands.

usage: python experiments/refutation.py <dataset> <preserved|reduced>
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.refute", run_name="__main__", alter_sys=True)
