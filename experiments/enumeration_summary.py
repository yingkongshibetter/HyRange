"""Merge results/enumeration/parts/*.json into results/enumeration/enum_results.json and print the HyRange side of
paragraph Enumeration: inputs per series and size, and the inputs on which HyRange decides every copy.
usage: python -I -B -X utf8 experiments/enumeration_summary.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
OUTD = CFG.RESULTS / "enumeration"


def decided(r):
    return r["B"]["labels_seed0"]["undecided"] == 0 and r["B"]["uncovered_components"] == 0


def main():
    allrecs = []
    for p in sorted((OUTD / "parts").glob("*.json")):
        allrecs += json.load(open(p, encoding="utf-8"))["records"]
    allrecs.sort(key=lambda r: (float(r["series"][3:]), r["n"], r["i"]))
    tot = dict(instances=len(allrecs), every_copy_decided=sum(1 for r in allrecs if decided(r)))
    json.dump(dict(script="experiments/enumeration.py + experiments/enumeration_summary.py", totals=tot,
                   records=allrecs), open(OUTD / "enum_results.json", "w", encoding="utf-8"), indent=1)
    print("| series | n | inputs | HyRange decides every copy |\n|---|---|---|---|")
    keys = sorted({(r["series"], r["n"]) for r in allrecs}, key=lambda k: (float(k[0][3:]), k[1]))
    for series, n in keys:
        R = [r for r in allrecs if r["series"] == series and r["n"] == n]
        print(f"| {series} | {n} | {len(R)} | {sum(1 for r in R if decided(r))} |")
    big6 = [r for r in allrecs if r["series"] == "deg6" and r["n"] >= 30]
    print(f"\nmean degree six, 30 to 100 nodes: HyRange decides every copy on {sum(1 for r in big6 if decided(r))} of "
          f"{len(big6)} inputs")


if __name__ == "__main__":
    main()
