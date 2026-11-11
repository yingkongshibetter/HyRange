"""Table I: statistics of the target hypergraphs of MARIOH's benchmark (preserved setting), from MARIOH's files with
MARIOH's loader rules (hyrange.bench.target and target_stats, the function behind results/bench/summary.json
records[].target).

usage: python -I -B -X utf8 experiments/table1_targets.py   -> results/bench/targets.json and the table on stdout
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyrange import bench as MB  # noqa: E402

SIX = ["crime", "hosts", "directors", "foursquare", "dblp", "topcs"]


def main():
    rows = []
    print("| Dataset | Nodes | Hyperedges | Mean size |\n|---|---|---|---|")
    for mode in MB.MODES:
        for ds in MB.DATASETS:
            lines, truth = MB.target(ds, mode)
            st = MB.target_stats(lines, truth)
            rows.append(dict(ds=ds, name=MB.DISP[ds], mode=mode, **st))
            if mode == "preserved" and ds in SIX:
                print(f"| {MB.DISP[ds]} | {st['nodes']:,} | {st['hyperedge_copies']:,} | {st['mean_size']:.1f} |")
    MB.save_json(dict(records=rows), MB.OUT / "targets.json")


if __name__ == "__main__":
    main()
