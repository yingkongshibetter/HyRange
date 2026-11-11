"""The 13 random samples of the dense benchmarks with known certain hyperedges (input of paragraph Local
certification).

Runs the module hyrange.dense_samples as a script; its docstring lists the commands.

usage: python experiments/dense_samples.py plan | one <sample id> | drive [<sample id> ...] [--workers 6]
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.dense_samples", run_name="__main__", alter_sys=True)
