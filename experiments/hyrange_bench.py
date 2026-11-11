"""HyRange on one benchmark input: forcing, components, rank test and one world per seed (Tables II and III).

Runs the module hyrange.pipeline as a script; its docstring lists the commands.

usage: python experiments/hyrange_bench.py <dataset> <preserved|reduced>
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.pipeline", run_name="__main__", alter_sys=True)
