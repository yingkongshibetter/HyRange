"""Job queue for MARIOH's benchmark (Table III; paragraph Dense hypergraphs).  Three kinds of jobs:
  ours    HyRange on one input (experiments/hyrange_bench.py: forcing, components, rank test, one world per seed);
  marioh  the rerun of MARIOH (experiments/marioh_runner.py, seeds 0, 1, 2);
  learned MARIOH on the residual pair counts after forcing (marioh_runner.py --query), only on the four dense
          benchmarks, where it completes the residual component without a world (label learned, experiments/hybrid.py).
At most --slots (8) processes at once: a MARIOH run is one process (one torch thread); an 'ours' run counts 2 slots
because it may start one child process for a very large MILP (pipeline.hsolve_guarded) while it waits.  Jobs whose
output exists are skipped, so the queue can be restarted.  A learned run waits for the residual written by the 'ours'
run of the same dataset and mode.

usage: python -I -B -X utf8 experiments/run_benchmark.py --modes preserved reduced [--slots 8] [--only ours,marioh,learned]
       [--datasets enron,crime] [--marioh-timeout 14400] [--ours-timeout 3600]
Status lines: results/bench/queue_status.jsonl; per-job logs: results/logs/bench/<job>.log
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

PY = sys.executable
LOGS = CFG.LOGS / "bench"
FLAGS = ["-I", "-B", "-X", "utf8"]
# long jobs first so that they start early
ORDER = ["dblp", "topcs", "hschool", "pschool", "eu", "enron", "foursquare", "hosts", "crime", "directors"]
DENSE = ("enron", "pschool", "hschool", "eu")      # the learned completion runs on these four only


def ours_done(ds, mode):
    p = MB.OUT / "ours" / f"{ds}_{mode}.json"
    if not p.exists():
        return False
    return json.load(open(p, encoding="utf-8")).get("stage") == "done"


def jobs_for(modes, kinds, datasets, a):
    J = []
    for mode in modes:
        for ds in [d for d in ORDER if d in datasets]:
            if "ours" in kinds:
                J.append(dict(name=f"ours_{ds}_{mode}", kind="ours", w=2, timeout=a.ours_timeout,
                              cmd=[PY] + FLAGS + [str(HERE / "hyrange_bench.py"), ds, mode],
                              done=lambda ds=ds, mode=mode: ours_done(ds, mode), ready=lambda: True))
            for s in MB.SEEDS:
                if "marioh" in kinds:
                    out = MB.OUT / "marioh" / f"{ds}_{mode}_s{s}.json"
                    J.append(dict(name=f"marioh_{ds}_{mode}_s{s}", kind="marioh", w=1, timeout=a.marioh_timeout,
                                  cmd=[str(MB.MAR_PY)] + FLAGS + [str(HERE / "marioh_runner.py"), "--data", ds, "--mode", mode,
                                                                  "--seed", str(s), "--out", str(out),
                                                                  "--pred", str(MB.OUT / "marioh" / f"{ds}_{mode}_s{s}_pred.json")],
                                  done=lambda out=out: out.exists(), ready=lambda: True))
                if "learned" in kinds and ds in DENSE:
                    out = MB.OUT / "learned" / f"{ds}_{mode}_s{s}.json"
                    q = MB.OUT / "forced" / f"{ds}_{mode}_residual.json"
                    J.append(dict(name=f"learned_{ds}_{mode}_s{s}", kind="learned", w=1, timeout=a.marioh_timeout,
                                  cmd=[str(MB.MAR_PY)] + FLAGS + [str(HERE / "marioh_runner.py"), "--data", ds, "--mode", mode,
                                                                  "--seed", str(s), "--query", str(q), "--out", str(out),
                                                                  "--pred", str(MB.OUT / "learned" / f"{ds}_{mode}_s{s}_pred.json")],
                                  done=lambda out=out: out.exists(), ready=lambda q=q: q.exists(),
                                  dep=f"ours_{ds}_{mode}"))
    return J


def kill_tree(p):
    CFG.kill_tree(p.pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["preserved"])
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--only", default="ours,marioh,learned")
    ap.add_argument("--datasets", default=",".join(MB.DATASETS))
    ap.add_argument("--marioh-timeout", type=float, default=14400.0)
    ap.add_argument("--ours-timeout", type=float, default=3600.0)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    MB.OUT.mkdir(parents=True, exist_ok=True)
    status_path = MB.OUT / "queue_status.jsonl"
    pending = [j for j in jobs_for(a.modes, a.only.split(","), a.datasets.split(","), a) if not j["done"]()]
    finished = {}
    running = []

    def record(j, st, wall):
        finished[j["name"]] = st
        with open(status_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(job=j["name"], status=st, wall_s=round(wall, 1),
                                    time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
        print(time.strftime("%H:%M:%S ") + f"{j['name']}: {st} ({wall:.0f}s)", flush=True)

    print(time.strftime("%H:%M:%S ") + f"{len(pending)} jobs", flush=True)
    while pending or running:
        for r in list(running):
            j, p, t0, fh = r
            rc = p.poll()
            wall = time.time() - t0
            if rc is not None:
                fh.close()
                running.remove(r)
                record(j, "ok" if rc == 0 and j["done"]() else f"exit {rc}", wall)
            elif wall > j["timeout"]:
                kill_tree(p)
                p.wait()
                fh.close()
                running.remove(r)
                record(j, f"killed: time limit {j['timeout']:.0f}s", wall)
        used = sum(r[0]["w"] for r in running)
        for j in list(pending):
            dep = j.get("dep")
            if not j["ready"]():
                if dep and dep in finished:            # the ours run ended without a residual
                    pending.remove(j)
                    record(j, f"skipped: {dep} {finished[dep]} wrote no residual", 0.0)
                continue
            if used + j["w"] > a.slots:
                continue
            fh = open(LOGS / f"{j['name']}.log", "w", encoding="utf-8")
            p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((j, p, time.time(), fh))
            pending.remove(j)
            used += j["w"]
            print(time.strftime("%H:%M:%S ") + f"start {j['name']} (slots {used}/{a.slots})", flush=True)
        if pending and not running and all(not j["ready"]() for j in pending):
            for j in pending:
                record(j, "skipped: dependency never ready", 0.0)
            pending = []
        time.sleep(2)
    print(time.strftime("%H:%M:%S ") + "queue empty", flush=True)


if __name__ == "__main__":
    main()
