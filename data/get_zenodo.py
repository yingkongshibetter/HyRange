"""Download the VOSviewer co-authorship deposit (map file and network file) of Zenodo record 22729673 into data/zenodo,
and check its sha256.  The deposit is CC-BY-4.0; it is not copied into this repository.

  zenodo22729673_authors  https://doi.org/10.5281/zenodo.22729673  (Web of Science Core Collection, authors)

usage: python data/get_zenodo.py [target folder]
Then: python -I -B -X utf8 experiments/zenodo_exports.py zenodo22729673_authors
          --map data/zenodo/zenodo22729673_authors/vosviewer_authors_map.txt
          --net data/zenodo/zenodo22729673_authors/vosviewer_authors_network.txt
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPOSITS = {
    "zenodo22729673_authors": ("22729673", {
        "vosviewer_authors_map.txt": ("vosviewer_authors_map.txt",
                                      "e1b00afbf9439f631d36d196d482126e3e0fff32db878e861517dee25f750bcb"),
        "vosviewer_authors_network.txt": ("vosviewer_authors_network.txt",
                                          "f386e38666ea2831289ceea7173c844c6dcaaa2aa693dcbddeffa7f8cb9b71b5"),
    }),
}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=180).read()


def main(root):
    ok = True
    for name, (rec, files) in DEPOSITS.items():
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        meta = json.loads(get(f"https://zenodo.org/api/records/{rec}"))
        links = {f["key"]: f["links"]["self"] for f in meta["files"]}
        for key, (local, sha) in files.items():
            p = out / local
            if not p.exists():
                p.write_bytes(get(links[key]))
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            print(name, local, "ok" if h == sha else "SHA256 DIFFERS " + h)
            ok &= h == sha
    return ok


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "zenodo"
    sys.exit(0 if main(root) else 1)
