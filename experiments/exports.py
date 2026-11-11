"""Co-authorship exports with a document threshold (Table V, Fig. 2).

Runs the module hyrange.exports as a script; its docstring lists the commands.

usage: python experiments/exports.py queue [--procs 4] | job <source> <t> <upper|cliques> | summary
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    runpy.run_module("hyrange.exports", run_name="__main__", alter_sys=True)
