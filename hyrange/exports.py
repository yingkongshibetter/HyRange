"""Co-authorship exports with a document threshold (VOSviewer style, full counting) with ground truth (Table V,
Fig. 2).

Hypothesis.  A co-authorship export with full counting that keeps the authors S with at least t documents and ignores
documents with more than CAP = 25 authors is an exact upper-bound-mode view of the TRACE of the documents on S:
  * network file : W_uv = number of (kept) documents that contain both u and v, u, v in S;
  * map file     : D_u  = number of (kept) documents of u, u in S (also those in which u is the only retained author);
  * trace        : the multiset {e & S : e a kept document, |e & S| >= 2}.
W is exactly the pair-count matrix of the trace and D_u >= d_u (number of multi-member trace hyperedges of u), the
difference being the documents whose trace is the singleton {u}.  So the trace is a world of (W, D) in upper-bound mode
and every copy certified in that mode must be a trace copy.

Per (source, t) two jobs:
  upper  : build the export; upper-bound-mode forcing = forcing_upper.WDForcer(adj, D, singleton=True), rules D1, A,
           B, P (= rule E of the experiments), no L, wall limit 900 s (exactly as upper.py); soundness check against the
           trace multiset (false copies); pair counts alone: closed form pairs_only.certified_pairs.
  cliques: the maximal cliques of the link graph (networkx.find_cliques, 600 s limit) with three or more members, and
           how many of them are the set of retained authors of a document (Fig. 2).
HyRange at t = 5 (exact certification in the upper-bound mode) runs in upper_exact.py (job export <source> 5).

usage: python -I -B -X utf8 experiments/exports.py job <source> <t> <upper|cliques>
       python -I -B -X utf8 experiments/exports.py queue [--procs 4]
       python -I -B -X utf8 experiments/exports.py summary
Writes results/exports/jobs/<source>_t<t>_<run>.json and results/exports/sim_export.json.
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
from hyrange import upper as UP  # noqa: E402  (load_forcing_upper, counting, exc_info; importing runs nothing)

OUT = CFG.RESULTS / "exports"
JOBS = OUT / "jobs"
LOGS = CFG.LOGS / "exports"
CAP = 25
THRESH = (1, 2, 3, 5, 10)
SOURCES = ["dblp", "openalex_db", "topcs", "openalex_cs_size3plus", "openalex_bio"]
DISP = {"dblp": "DBLP", "topcs": "MAG-TopCS", "openalex_db": "OpenAlex-DB", "openalex_cs_size3plus": "OpenAlex-CS-3+",
        "openalex_bio": "OpenAlex-Bio"}
FORCE_TL = 900.0
CLIQUE_TL = 600.0
KILL_S = {"upper": 1500.0, "cliques": 700.0}
RUNS = ("upper", "cliques")
UPPER_RULES = ("D1", "A", "B", "P")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ------------------------------------------------------------------ sources and export
def source_docs(src):
    """(documents as sorted tuples of original ids, repeats = separate documents; source facts)."""
    if src in ("dblp", "topcs"):
        path = MB.target_file(src, "preserved")
        raw_sizes = Counter()
        with open(path, "r") as f:
            for line in f:
                raw_sizes[len(line.split())] += 1
        docs = MB.read_lines(path)
        info = dict(origin=f"MARIOH target {path.name} ({src}, preserved setting), one document per line",
                    raw_lines=int(sum(raw_sizes.values())), raw_lines_size_le1=int(raw_sizes.get(0, 0) + raw_sizes.get(1, 0)))
    else:
        from hyrange import views
        D, docs = views.load(src)
        info = dict(origin=f"views.load('{src}') (incidence rows = works, columns = authors)",
                    raw_lines=int(D.H.shape[0]), raw_lines_size_le1=int((D.sizes < 2).sum()))
    sz = Counter(len(e) for e in docs)
    info.update(documents=len(docs), authors=len(set(v for e in docs for v in e)), min_size=min(sz), max_size=max(sz),
                single_author_documents=int(sz.get(1, 0)), documents_over_cap=int(sum(v for k, v in sz.items() if k > CAP)),
                repeated_documents=len(docs) - len(set(docs)))
    return docs, info


def build_export(docs, t):
    """Export at threshold t: S, local ids (rank of original id in S), trace multiset (local ids), adj (W), d, D."""
    kept = [e for e in docs if len(e) <= CAP]
    Dall = Counter()
    for e in kept:
        for u in e:
            Dall[u] += 1
    S = sorted(u for u, c in Dall.items() if c >= t)
    loc = {u: i for i, u in enumerate(S)}
    n = len(S)
    trace = Counter()
    single = empty = 0
    for e in kept:
        T = tuple(loc[u] for u in e if u in loc)      # e sorted and loc monotone => T sorted
        if len(T) >= 2:
            trace[T] += 1
        elif len(T) == 1:
            single += 1
        else:
            empty += 1
    adj = [dict() for _ in range(n)]
    d = [0] * n
    for e, m in trace.items():
        for x in e:
            d[x] += m
            ax = adj[x]
            for y in e:
                if y != x:
                    ax[y] = ax.get(y, 0) + m
    D = [Dall[u] for u in S]
    facts = dict(cap=CAP, documents_kept=len(kept), documents_dropped_by_cap=len(docs) - len(kept), S=n,
                 S_linked=sum(1 for a in adj if a), S_isolated=sum(1 for a in adj if not a),
                 trace_copies=int(sum(trace.values())), trace_distinct=len(trace),
                 trace_vol=int(sum(len(e) * m for e, m in trace.items())),
                 documents_trace_singleton=single, documents_trace_empty=empty,
                 pairs=sum(len(a) for a in adj) // 2, pair_weight=int(sum(sum(a.values()) for a in adj) // 2))
    assert sum(D) - sum(d) == single, "sum_u D_u - sum_u d_u must equal the number of singleton traces"
    assert all(Dx >= dx for Dx, dx in zip(D, d))
    return S, trace, adj, d, D, facts


def multiset_false(forced, truth):
    return int(sum(max(0, m - truth.get(e, 0)) for e, m in forced.items()))


# ------------------------------------------------------------------ jobs
def jkey(src, t, run):
    return f"{src}_t{t:02d}_{run}"


def job(src, t, run):
    outp = JOBS / (jkey(src, t, run) + ".json")
    t_all = time.perf_counter()
    docs, info = source_docs(src)
    S, trace, adj, d, D, facts = build_export(docs, t)
    n = len(S)
    res = dict(source=src, name=DISP[src], t=t, run=run, stage="built", source_info=info, **facts,
               t_build_s=time.perf_counter() - t_all)
    log(f"{src} t {t} {run}: |S| {n}, trace copies {facts['trace_copies']}, vol {facts['trace_vol']}")
    MB.save_json(res, outp)
    vol = facts["trace_vol"]
    if run == "upper":
        W2 = UP.load_forcing_upper()
        F = UP.counting(W2.WDForcer)(adj, D, rules=UPPER_RULES, time_limit=FORCE_TL, copy_adj=True, singleton=True)
        F.vol_by_rule = {}
        t0 = time.perf_counter()
        try:
            F.run()
            exc = None
        except AssertionError as ex:          # forcing_upper Inconsistent: must not happen, the trace is a world
            exc = ex
        tf = time.perf_counter() - t0
        forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items() if m > 0})
        fv = int(sum(len(e) * m for e, m in forced.items()))
        false = multiset_false(forced, trace)
        res.update(stage="forced", forcer="forcing_upper.WDForcer(singleton=True)", rules=list(UPPER_RULES), t_force_s=tf,
                   force_complete=bool(F.complete) and exc is None, rounds=getattr(F, "rounds", None),
                   exception=UP.exc_info(exc) if exc else None,
                   cert_copies=int(sum(forced.values())), cert_vol=fv, cert_vol_share=fv / vol if vol else None,
                   false_copies=false, sound=(false == 0 and exc is None))
        if not res["sound"]:
            bad = sorted((e, m, trace.get(e, 0)) for e, m in forced.items() if m > trace.get(e, 0))
            ex_nodes = sorted(set(x for e, _, _ in bad[:5] for x in e))
            nb = sorted(set(ex_nodes) | set(y for x in ex_nodes for y in adj[x]))
            res["violation_examples"] = [[[S[i] for i in e], m, tm] for e, m, tm in bad[:20]]
            res["violation_local_instance"] = dict(
                nodes_original=[S[i] for i in nb], D=[D[i] for i in nb],
                W=[[S[i], S[j], adj[i][j]] for i in nb for j in adj[i] if i < j and j in set(nb)])
            log(f"SOUNDNESS VIOLATION {src} t {t}: {false} false copies, exc {exc}")
        # W-only closed form (pair counts alone)
        from hyrange.pairs_only import certified_pairs
        t1 = time.perf_counter()
        wo = certified_pairs({u: a for u, a in enumerate(adj)})
        two = time.perf_counter() - t1
        wo_c = Counter({tuple(p): c for p, c in wo.items()})
        res.update(w_only_copies=int(sum(wo.values())), w_only_vol=2 * int(sum(wo.values())),
                   w_only_vol_share=(2 * sum(wo.values()) / vol) if vol else None,
                   w_only_false=multiset_false(wo_c, trace), w_only_s=two)
        res["stage"] = "done"
        res["t_total_s"] = time.perf_counter() - t_all
        MB.save_json(res, outp)
        log(f"upper {tf:.1f}s share {res['cert_vol_share']:.4f} copies {res['cert_copies']} false {false}; "
            f"W-only {res['w_only_vol_share']:.4f}")
        return
    if run == "cliques":
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(u for u in range(n) if adj[u])
        G.add_edges_from((u, v) for u in range(n) for v in adj[u] if u < v)
        t0 = time.perf_counter()
        tot = ge3n = ge3true = 0
        sizes = Counter()
        finished = True
        for c in nx.find_cliques(G):
            tot += 1
            if len(c) >= 3:
                ge3n += 1
                sizes["5+" if len(c) >= 5 else str(len(c))] += 1
                if tuple(sorted(c)) in trace:
                    ge3true += 1
            if (tot & 1023) == 0 and time.perf_counter() - t0 > CLIQUE_TL:
                finished = False
                break
        tc = time.perf_counter() - t0
        res.update(clique=dict(finished=finished, seconds=tc, maximal_cliques=tot, maximal_cliques_ge3=ge3n,
                               maximal_cliques_ge3_true_distinct=ge3true,
                               size_hist={k: int(sizes.get(k, 0)) for k in ("3", "4", "5+")}),
                   stage="done", t_total_s=time.perf_counter() - t_all)
        MB.save_json(res, outp)
        log(f"cliques {tc:.1f}s finished {finished}: {tot} maximal, {ge3n} >= 3 ({ge3true} true)")
        return
    raise SystemExit(run)


# ------------------------------------------------------------------ queue
def all_jobs():
    return [(src, t, run) for t in THRESH for src in SOURCES for run in RUNS]


def run_queue(procs):
    LOGS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    pending = all_jobs()
    running = []
    last_sum = time.time()
    while pending or running:
        for rj in list(running):
            p_, fh, k, t0, spec = rj
            rc = p_.poll()
            if rc is None and time.time() - t0 > KILL_S[spec[2]]:
                p_.kill()
                p_.wait()
                rc = "killed"
            if rc is not None:
                fh.close()
                running.remove(rj)
                jp = JOBS / (k + ".json")
                rec = json.load(open(jp, encoding="utf-8")) if jp.exists() else dict(source=spec[0], t=spec[1],
                                                                                         run=spec[2])
                if rec.get("stage") != "done":
                    rec.update(timeout=(rc == "killed"), returncode=str(rc), stage_reached=rec.get("stage"),
                               stage="failed" if rc != "killed" else "timeout")
                    MB.save_json(rec, jp)
                log(f"end {k} rc {rc} ({time.time() - t0:.0f}s)")
                summary()
        # at most 2 DBLP jobs at once (memory)
        while pending and len(running) < procs:
            n_dblp = sum(1 for r in running if r[4][0] == "dblp")
            idx = next((i for i, s in enumerate(pending) if s[0] != "dblp" or n_dblp < 2), None)
            if idx is None:
                break
            spec = pending.pop(idx)
            k = jkey(*spec)
            jp = JOBS / (k + ".json")
            if jp.exists() and json.load(open(jp, encoding="utf-8")).get("stage") == "done":
                continue
            fh = open(LOGS / f"{k}.log", "w", encoding="utf-8")
            cmd = [*CFG.module_cmd("hyrange.exports"), "job", spec[0], str(spec[1]), spec[2]]
            p_ = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((p_, fh, k, time.time(), spec))
            log(f"start {k}")
        if time.time() - last_sum > 120:
            summary()
            last_sum = time.time()
        time.sleep(0.5)
    summary()
    log("queue done")


# ------------------------------------------------------------------ summary
def load_job(src, t, run):
    jp = JOBS / (jkey(src, t, run) + ".json")
    if not jp.exists():
        return None
    try:
        return json.load(open(jp, encoding="utf-8"))
    except (json.JSONDecodeError, PermissionError):
        return None


def summary():
    recs = []
    for src in SOURCES:
        for t in THRESH:
            row = dict(source=src, name=DISP[src], t=t)
            for run in RUNS:
                row[run] = load_job(src, t, run)
            recs.append(row)
    viol = [dict(source=r["source"], t=r["t"], false_copies=r["upper"]["false_copies"],
                 exception=r["upper"].get("exception"), examples=r["upper"].get("violation_examples"))
            for r in recs if r["upper"] and r["upper"].get("stage") == "done" and not r["upper"]["sound"]]
    done = sum(1 for r in recs for run in RUNS if r[run] and r[run].get("stage") == "done")
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), jobs_done=done, jobs_total=len(RUNS) * len(recs),
               complete=done == len(RUNS) * len(recs),
               protocol=dict(
                   export=("keep documents with <= 25 authors; D_u = number of kept documents of u; S = {u : D_u >= t}; "
                           "W_uv = number of kept documents containing u and v (u, v in S); truth = multiset of traces "
                           "e & S with |e & S| >= 2; local ids = rank of the original id in S"),
                   thresholds=list(THRESH), cap=CAP,
                   upper_mode="forcing_upper.WDForcer(adj, D, singleton=True), rules D1, A, B, P (rule E), no L, 900 s",
                   w_only="pairs_only.certified_pairs: copies of {u,v} = max(0, W_uv - sum_z min(W_uz, W_vz))",
                   volume="sum of sizes of the multi-member trace copies",
                   false_copies="sum_e max(0, m_forced(e) - m_trace(e))",
                   cliques=("networkx.find_cliques on the link graph (W > 0), 600 s limit; maximal cliques with >= 3 "
                            "members; 'true' = the clique is the set of retained authors of a document")),
               records=recs, upper_soundness_violations=viol)
    MB.save_json(out, OUT / "sim_export.json")
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "job":
        job(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == "queue":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
        run_queue(procs)
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
