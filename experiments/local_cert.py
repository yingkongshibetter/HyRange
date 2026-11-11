"""Local certification by the levels (paragraph Local certification).

Runs the module hyrange.levels as a script; its docstring lists the commands.

usage: python experiments/local_cert.py m4 <dataset> <preserved|reduced> <exact|upper> | m4all
       python experiments/local_cert.py sample <sample id> [--levels node|all] | drive [<sample id> ...]
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.levels", run_name="__main__", alter_sys=True)
