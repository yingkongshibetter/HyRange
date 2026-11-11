"""Paragraph Enumeration: runs the authors' TwinSearch code (github.com/tlarock/twin_search, built into
build/ts_run.exe, see README.md in this folder) on the random inputs written by experiments/enumeration.py
(results/enumeration/parts/*.json), in the order deg2, deg3 (sizes 6 to 100), deg6 (sizes 6 to 30), and compares the
hyperedges common to all hypergraphs that it lists with those that HyRange certifies on the same input.

usage (from the repository root):
  python -I -B -X utf8 experiments/twinsearch/run_ts.py random <stop time HH:MM, no new input starts after it> [--tl 600] [--workers 2]
  python -I -B -X utf8 experiments/twinsearch/run_ts.py table
Writes results/twinsearch/results_random.jsonl and results.json.

Each input is given to TwinSearch with the node degrees on the diagonal (use_diagonal = true), hyperedge sizes 2..n,
isomorphism filter off, one TBB thread.  'finished' = the process returned the full list within the limit.
"""
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from hyrange import config as CFG  # noqa: E402

OUT = CFG.RESULTS / "twinsearch"
OUT.mkdir(parents=True, exist_ok=True)
EXE = Path(os.environ.get("HYRANGE_TWINSEARCH_EXE", HERE / "build" / "ts_run.exe"))
PARTS = CFG.RESULTS / "enumeration" / "parts"
WORK = OUT / "work"
MEM_GUARD = 6 * 1024 ** 3
DUMP_CAP = 2_000_000
LOCK = threading.Lock()


def key(e):
    return "-".join(map(str, e))


def unkey(s):
    return tuple(int(x) for x in s.split("-"))


def wd(H):
    W, d = Counter(), Counter()
    for e in H:
        for u in e:
            d[u] += 1
        for u, v in combinations(sorted(e), 2):
            W[(u, v)] += 1
    return W, d


def run_one(H, tag, tl, use_diag=1, min_k=2, max_k=0, filter_iso=0):
    """H: list of tuples of node ids.  Returns a record; node ids in the record are the ids of H."""
    WORK.mkdir(exist_ok=True)
    nodes = sorted({u for e in H for u in e})
    loc = {u: i for i, u in enumerate(nodes)}
    fin, ftw, ferr = WORK / f"{tag}.txt", WORK / f"{tag}.twins", WORK / f"{tag}.err"
    fin.write_text("".join(",".join(str(loc[u]) for u in sorted(e)) + "\n" for e in H), encoding="ascii")
    if ftw.exists():
        ftw.unlink()
    cmd = [str(EXE), str(fin), str(min_k), str(max_k), str(use_diag), str(filter_iso), "1", str(tl), str(ftw),
           str(DUMP_CAP)]
    env = dict(os.environ)
    env.pop("TWIN_RSS_TRACE", None)
    t0 = time.perf_counter()
    rec = dict(tag=tag, n_nodes_given=len(nodes), m=len(H), use_diag=bool(use_diag), min_k=min_k, limit_s=tl)
    with open(ferr, "wb") as err:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, cwd=str(WORK), env=env)
        ps = psutil.Process(p.pid)
        peak, status = 0, None
        while True:
            try:
                p.wait(timeout=0.05 if time.perf_counter() - t0 < 2 else 0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            try:
                peak = max(peak, ps.memory_info().rss)
            except psutil.Error:
                pass
            if peak > MEM_GUARD:
                status = "memory"
            elif time.perf_counter() - t0 > tl + 20:
                status = "time_limit_after_traversal"
            if status:
                p.kill()
                p.wait()
                break
        out = p.stdout.read().decode("utf-8", "replace")
    rec["wall_s"] = time.perf_counter() - t0
    rec["peak_rss_mb"] = peak / 1048576
    rec["returncode"] = p.returncode
    rec["stderr"] = ferr.read_text(encoding="utf-8", errors="replace")[:500]
    if status is None:
        try:
            j = json.loads(out.strip().splitlines()[-1])
        except Exception:
            status = "crash"
            j = None
        if j is not None:
            rec["ts"] = {k: v for k, v in j.items() if k != "common"}
            rec["t_s"] = (j["ms_setup"] + j["ms_search"]) / 1000.0
            if j["drained"]:
                status = "time_limit_in_traversal"
            elif not j["feasible"]:
                status = "infeasible"
            else:
                status = "complete"
                back = lambda s: tuple(sorted(nodes[i] for i in unkey(s)))
                common = sorted(back(s) for s in j["common"])
                rec["twins"] = j["twins"]
                rec["common"] = [key(e) for e in common]
                if j["dumped"]:
                    # independent check of the list: every listed hypergraph is simple, distinct, has the input W and d
                    W0, d0 = wd(H)
                    seen, inter = set(), None
                    for line in ftw.read_text(encoding="ascii").splitlines():
                        T = tuple(sorted(back(s) for s in line.split("|"))) if line else ()
                        assert len(set(T)) == len(T), "listed hypergraph has a repeated hyperedge"
                        assert T not in seen, "hypergraph listed twice"
                        seen.add(T)
                        W1, d1 = wd(T)
                        assert W1 == W0, "listed hypergraph has other pair counts"
                        if use_diag:
                            assert d1 == d0, "listed hypergraph has other degrees"
                        inter = set(T) if inter is None else inter & set(T)
                    assert len(seen) == j["twins"]
                    assert sorted(inter or ()) == common, "intersection mismatch"
                    rec["list_verified"] = True
                    rec["truth_listed"] = tuple(sorted(tuple(sorted(e)) for e in H)) in seen
                else:
                    rec["list_verified"] = False
    rec["status"] = status
    rec["finished"] = status == "complete"
    for f in (fin, ftw):
        if f.exists():
            f.unlink()
    return rec


# ------------------------------------------------------------------ random inputs of experiments/enumeration.py
ORDER = ([("deg2", n) for n in (6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100)]
         + [("deg3", n) for n in (6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100)]
         + [("deg6", n) for n in (6, 8, 10, 12, 15, 20, 25, 30)])


def one_random(series, n, r, tl):
    H = [unkey(s) for s in r["hyperedges"]]
    rec = run_one(H, f"{series}_n{n}_i{r['i']}", tl)
    rec.update(series=series, n=n, i=r["i"], seed=r["seed"])
    B = r["B"]["certified"]
    rec["hyrange_certified"] = len(B)
    rec["hyrange_t_s"] = r["B"]["t_s"]
    if rec["finished"]:
        common = set(rec["common"])
        assert all(m == 1 for m in B.values())
        rec["check_i_violations"] = sorted(set(B) - common) if rec["twins"] > 0 else []
        rec["check_ii_equal"] = set(B) == common
        rec["common_minus_hyrange"] = len(common - set(B))
    return rec


def random_runs(stop_hhmm, tl, workers):
    hh, mm = map(int, stop_hhmm.split(":"))
    lt = time.localtime()
    stop = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    outf = OUT / "results_random.jsonl"
    done = set()
    if outf.exists():
        for line in open(outf, encoding="utf-8"):
            j = json.loads(line)
            done.add((j["series"], j["n"], j["i"]))
    jobs = []
    for series, n in ORDER:
        R = json.load(open(PARTS / f"{series}_n{n}.json", encoding="utf-8"))["records"]
        jobs += [(series, n, r) for r in R if (series, n, r["i"]) not in done]
    it = iter(jobs)

    def worker():
        while True:
            with LOCK:
                if time.time() > stop:
                    return
                job = next(it, None)
            if job is None:
                return
            series, n, r = job
            rec = one_random(series, n, r, tl)
            with LOCK:
                with open(outf, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(time.strftime("%H:%M:%S"), series, n, r["i"], rec["status"], "twins", rec.get("twins"),
                      "t %.3f" % rec.get("t_s", -1), "wall %.2f" % rec["wall_s"],
                      "viol", rec.get("check_i_violations"), "eq", rec.get("check_ii_equal"), flush=True)
    th = [threading.Thread(target=worker) for _ in range(workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()


def table():
    recs = [json.loads(line) for line in open(OUT / "results_random.jsonl", encoding="utf-8")]
    rows = []
    for series, n in ORDER:
        R = [r for r in recs if r["series"] == series and r["n"] == n]
        if not R:
            continue
        F = [r for r in R if r["finished"]]
        rows.append(dict(series=series, n=n, inputs_run=len(R), finished=len(F),
                         not_finished_status=dict(Counter(r["status"] for r in R if not r["finished"])),
                         certified_in_every_listed=sum(1 for r in F if not r["check_i_violations"]),
                         common_equals_certified=sum(1 for r in F if r["check_ii_equal"]),
                         list_verified=sum(1 for r in F if r.get("list_verified"))))
    json.dump(dict(script="run_ts.py", table=rows, records=recs), open(OUT / "results.json", "w", encoding="utf-8"),
              indent=1)
    low = [r for r in rows if r["series"] in ("deg2", "deg3")]
    print(f"mean degree two and three: TwinSearch finishes within the limit on {sum(r['finished'] for r in low)} of "
          f"{sum(r['inputs_run'] for r in low)} inputs; the hyperedges common to its lists equal those HyRange "
          f"certifies on {sum(r['common_equals_certified'] for r in low)}")
    for r in rows:
        if r["series"] == "deg6" and r["n"] == 10:
            print(f"mean degree six, 10 nodes: TwinSearch finishes on {r['finished']} of {r['inputs_run']} inputs")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "random":
        tl = float(sys.argv[sys.argv.index("--tl") + 1]) if "--tl" in sys.argv else 600.0
        wk = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
        random_runs(sys.argv[2], tl, wk)
    elif what == "table":
        table()
