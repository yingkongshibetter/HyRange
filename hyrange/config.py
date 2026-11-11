"""Paths of the inputs and outputs.  Each one can be set by an environment variable; the defaults are relative to the
repository root.

  HYRANGE_DATA            folder of the inputs                      (default: <repo>/data)
  HYRANGE_MARIOH          clone of the MARIOH repository             (default: <repo>/data/MARIOH)
  HYRANGE_MARIOH_PYTHON   Python of the MARIOH environment          (default: the running interpreter)
  HYRANGE_OPENALEX        folder of the three OpenAlex samples       (default: <repo>/data/openalex)
  HYRANGE_ZENODO          folder of the Zenodo deposit               (default: <repo>/data/zenodo)
  HYRANGE_RESULTS         folder of all outputs                      (default: <repo>/results)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("HYRANGE_DATA", ROOT / "data"))
MARIOH = Path(os.environ.get("HYRANGE_MARIOH", DATA / "MARIOH"))
MARIOH_PYTHON = Path(os.environ.get("HYRANGE_MARIOH_PYTHON", sys.executable))
OPENALEX = Path(os.environ.get("HYRANGE_OPENALEX", DATA / "openalex"))
ZENODO = Path(os.environ.get("HYRANGE_ZENODO", DATA / "zenodo"))
RESULTS = Path(os.environ.get("HYRANGE_RESULTS", ROOT / "results"))
LOGS = RESULTS / "logs"


def module_cmd(module: str) -> list:
    """Command line that runs a module of this package as a script in a fresh interpreter (isolated mode, no
    bytecode files, UTF-8), with the repository root on sys.path.  Arguments are appended by the caller."""
    boot = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import runpy; "
            f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)")
    return [sys.executable, "-I", "-B", "-X", "utf8", "-c", boot]


def kill_tree(pid: int) -> None:
    """Kill a process and its children: taskkill on Windows; elsewhere psutil if installed, else the process alone."""
    import subprocess
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return
    try:
        import psutil
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            c.kill()
        p.kill()
    except Exception:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
