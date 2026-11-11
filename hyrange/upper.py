"""Forcing in the upper-bound mode on the six benchmarks (paragraph Upper-bound mode: 94.1% to 100% by forcing alone).

In exported co-occurrence files the diagonal also counts one-member groups, so D_u is only an UPPER bound on the
number d_u of multi-member hyperedges through u.  Forcer: forcing_upper.WDForcer(adj, D, singleton=True) (proofs in
the experiments), rules D1, A, B, P (= rule E of the experiments; rule L is not used in this mode), wall limit 900 s;
U_u = min(D_u, r_u), ell_u = max_v W_uv.

Input per benchmark and setting: (W, d) of the target lines exactly as pipeline.py (bench.target, forcing.build_wdeg
over local ids = rank of the original node id), with the diagonal D = d read as an upper bound.
Ground truth is used only after the forcing, to count false forced copies (multiset: sum_e max(0, m_forced(e) -
m_target(e))).  Volume = sum of sizes of the target's copies (= sum_u d_u).

usage: python -I -B -X utf8 experiments/upper_forcing.py job <ds> <mode>
       python -I -B -X utf8 experiments/upper_forcing.py queue [--procs 4]
       python -I -B -X utf8 experiments/upper_forcing.py summary
Writes results/upper_forcing/jobs/<ds>_<mode>.json and results/upper_forcing/upper_diag.json.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402

OUT = CFG.RESULTS / "upper_forcing"
JOBS = OUT / "jobs"
LOGS = CFG.LOGS / "upper_forcing"
SIX = ["crime", "hosts", "directors", "foursquare", "dblp", "topcs"]
FORCE_TL = 900.0          # pipeline.py --force-tl default
KILL_S = 960.0            # external wall cap per job (15 min + 60 s to write the record)
UPPER_RULES = ("D1", "A", "B", "P")
RULE_NAMES = ("D1", "A", "B", "P", "L")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def load_forcing_upper():
    """The forcer of the upper-bound mode (module forcing_upper)."""
    from hyrange import forcing_upper
    return forcing_upper


def counting(cls):
    """Subclass that also records the forced volume per rule."""
    class Counting(cls):
        def subtract(self, e, m, rule, touched):
            super().subtract(e, m, rule, touched)
            if m > 0:
                self.vol_by_rule[rule] = self.vol_by_rule.get(rule, 0) + m * len(e)
    return Counting


def exc_info(ex):
    a = ex.args[0] if ex.args else None
    kind = a[0] if isinstance(a, tuple) and a and isinstance(a[0], str) else type(ex).__name__
    rule = a[-1] if isinstance(a, tuple) and a and isinstance(a[-1], str) and a[-1] in RULE_NAMES else None
    return dict(type=type(ex).__name__, kind=str(kind), rule=rule, msg=repr(ex)[:300])


def job(ds, mode):
    from hyrange import forcing as WC          # forcing.py (the one pipeline.py uses)
    outp = JOBS / f"{ds}_{mode}.json"
    res = dict(ds=ds, name=MB.DISP[ds], mode=mode, stage="start")
    t_all = time.perf_counter()
    lines, truth = MB.target(ds, mode)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    lines_loc = [tuple(loc[v] for v in e) for e in lines]
    true_mult = Counter(lines_loc)
    adj, deg = WC.build_wdeg(lines_loc, n)
    vol = int(sum(deg))
    res.update(n=n, lines=len(lines), vol=vol)
    MB.save_json(res, outp)
    W2 = load_forcing_upper()
    F = counting(W2.WDForcer)(adj, list(deg), rules=UPPER_RULES, time_limit=FORCE_TL, copy_adj=True, singleton=True)
    F.vol_by_rule = {}
    t0 = time.perf_counter()
    try:
        F.run()
        exc = None
    except AssertionError as ex:          # forcing_upper Inconsistent: must not happen, the input has a world
        exc = ex
    t = time.perf_counter() - t0
    forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items() if m > 0})
    fv = int(sum(len(e) * m for e, m in forced.items()))
    false = int(sum(max(0, m - true_mult.get(e, 0)) for e, m in forced.items()))
    res.update(stage="forced", forcer="forcing_upper.WDForcer(singleton=True)", rules=list(UPPER_RULES),
               t_force_s=t, force_complete=bool(F.complete) and exc is None,
               exception=exc_info(exc) if exc else None, forced_copies=int(sum(forced.values())),
               forced_vol=fv, forced_vol_share=fv / vol, false_copies=false, sound=(false == 0 and exc is None))
    if not res["sound"]:
        log(f"SOUNDNESS VIOLATION {ds} {mode}: {false} false copies, exc {exc}")
    log(f"{ds} {mode} upper: {t:.1f}s share {res['forced_vol_share']:.4f} copies {res['forced_copies']} "
        f"false {false}")
    res["stage"] = "done"
    res["t_total_s"] = time.perf_counter() - t_all
    MB.save_json(res, outp)


def run_queue(procs):
    LOGS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    pending = [(ds, mode) for ds in SIX for mode in MB.MODES]
    running = []
    while pending or running:
        for rjob in list(running):
            p_, fh, k, t0 = rjob
            rc = p_.poll()
            if rc is None and time.time() - t0 > KILL_S:
                p_.kill()
                p_.wait()
                rc = "killed"
            if rc is not None:
                fh.close()
                running.remove(rjob)
                log(f"end {k} rc {rc} ({time.time() - t0:.0f}s)")
        while pending and len(running) < procs:
            ds, mode = pending.pop(0)
            k = f"{ds}_{mode}"
            jp = JOBS / f"{k}.json"
            if jp.exists() and json.load(open(jp, encoding="utf-8")).get("stage") == "done":
                continue
            fh = open(LOGS / f"{k}.log", "w", encoding="utf-8")
            cmd = [*CFG.module_cmd("hyrange.upper"), "job", ds, mode]
            running.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE)), fh, k,
                            time.time()))
        time.sleep(0.5)
    summary()
    log("queue done")


def summary():
    recs = []
    for mode in MB.MODES:
        for ds in SIX:
            jp = JOBS / f"{ds}_{mode}.json"
            r = json.load(open(jp, encoding="utf-8")) if jp.exists() else None
            if r is None or r.get("stage") != "done":
                print(f"{MB.DISP[ds]} {mode}: missing")
                continue
            recs.append(dict(ds=ds, name=MB.DISP[ds], mode=mode, forced_vol_share=r["forced_vol_share"],
                             false_copies=r["false_copies"], sound=r["sound"]))
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), all_sound=all(r["sound"] for r in recs), records=recs)
    MB.save_json(out, OUT / "upper_diag.json")
    for r in recs:
        print(f"{r['name']} {r['mode']}: forced {100 * r['forced_vol_share']:.1f}% of the volume, "
              f"false copies {r['false_copies']}")
    if recs:
        sh = [r["forced_vol_share"] for r in recs]
        print(f"forcing alone in the upper-bound mode: {100 * min(sh):.1f}% to {100 * max(sh):.1f}%")
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "job":
        job(sys.argv[2], sys.argv[3])
    elif cmd == "queue":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
        run_queue(procs)
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
