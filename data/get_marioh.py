"""Fetch the MARIOH repository (code and the ten benchmark datasets) at the commit used for the experiments into
data/MARIOH (or the folder given as argument), and check the 40 data files against data/marioh_sha256.txt.

MARIOH: K. Lee, G. Lee, K. Shin, "MARIOH: Multiplicity-Aware Hypergraph Reconstruction", ICDE 2025.
Repository: https://github.com/KyuhanLee/MARIOH (MIT license), commit 134affb3e6d8ea81f905d0021b195f7ed75c1c3b.

usage: python data/get_marioh.py [target folder]
Uses git if it is installed, else downloads the GitHub archive of the commit.
"""
import hashlib
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/KyuhanLee/MARIOH"
COMMIT = "134affb3e6d8ea81f905d0021b195f7ed75c1c3b"
HERE = Path(__file__).resolve().parent


def fetch(dst):
    if dst.exists() and any(dst.iterdir()):
        print(f"{dst} exists; checking it")
        return
    if shutil.which("git"):
        subprocess.run(["git", "clone", URL, str(dst)], check=True)
        subprocess.run(["git", "-C", str(dst), "checkout", COMMIT], check=True)
        return
    url = f"{URL}/archive/{COMMIT}.zip"
    print("downloading", url)
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        tmp = dst.parent / (dst.name + "_tmp")
        z.extractall(tmp)
        shutil.move(str(tmp / top), str(dst))
        shutil.rmtree(tmp, ignore_errors=True)


def check(dst):
    bad = 0
    for line in (HERE / "marioh_sha256.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        h, rel = line.split()
        p = dst / "data" / rel
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        if got != h:
            bad += 1
            print("MISMATCH", rel, got)
    print("all 40 data files match" if not bad else f"{bad} files differ")
    return bad == 0


if __name__ == "__main__":
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "MARIOH"
    fetch(dst)
    sys.exit(0 if check(dst) else 1)
