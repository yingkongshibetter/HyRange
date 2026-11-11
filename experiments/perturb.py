"""Paragraph Inconsistent inputs: one wrong count in the input (W, d).

One pair count or one degree of a benchmark input is changed by one (kinds P+, P-, D+, D-), with 200 random trials
for each of the 12 inputs, each kind of entry and each direction (9,600 trials).  Counted: the trials in which HyRange
meets a contradiction and rejects the changed input, and, among the others, those in which a world of the changed
input was found and checked exactly (so the changed input is consistent and no check can reject it).

Pipeline per trial = the exact mode as run for the experiments (pipeline.py + certify.py), nothing modified:
  1. forcing: pipeline.WDForcerTL (forcing.WDForcer, rules D1, A, B, P(= paper's E), L; 900 s, 10 s per rule-L call);
  2. residual components (engine._components, smallest first), candidates.build_comp_wp (cap 300,000 / 60 s, rank test);
     pipeline volume assertion forced_vol + sum(C.vol) == vol;
  3. world search per non-pinned component: pipeline.world_of (HiGHS, random objective rng(0), 60 s), checked exactly;
  4. exact certification per component with a found world: certify.certify_component on the one found world
     (seed 0), with reduced per-component budgets (BUDGETS below); certified = forced + pinned + lb.
Existing checks that can fire (all of them existed before this script; none is added to the pipeline):
  forcing   : forcing.Inconsistent raised in subtract / node_rules / pair_rule / local_rule;
  components: AssertionError in candidates.build_comp_wp ("residual node without degree", "r_u < d_u on the residual",
              "negative slack on the residual" in the pair-core filter); the pipeline volume assertion;
              the rank-test status rank_full_no_world;
  worlds    : pipeline.world_of status "nosol:Infeasible" (HiGHS: no world of the component) or "unverified".
Caching (outcome-equivalent): the certification record of a component is reused when a component with the identical
  residual (nodes, rows, entries) was certified before in this process with every candidate decided (then the
  certified copies equal F^cert of the component and do not depend on the found world).

usage: python -I -B -X utf8 experiments/perturb.py run <ds> <mode> [--trials 200] [--deadline EPOCH]
       python -I -B -X utf8 experiments/perturb.py summary
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import config as CFG  # noqa: E402
from hyrange import bench as MB  # noqa: E402
from hyrange import engine as EN  # noqa: E402
from hyrange import forcing as WC  # noqa: E402
from hyrange import candidates as WP  # noqa: E402
from hyrange import pipeline as MO  # noqa: E402
from hyrange import certify as CX  # noqa: E402

OUTD = CFG.RESULTS / "perturb"
RES = OUTD / "results"
DS6 = ["crime", "hosts", "directors", "foursquare", "dblp", "topcs"]
KINDS = ["P+", "P-", "D+", "D-"]
SEED_BASE = 20261111
FORCE_TL = 900.0
MILP_TL = 60.0
BUDGETS = dict(ip_s=10.0, ip_comp_s=60.0, dfs_comp_s=60.0, dfslp_s=20.0)   # certify.py defaults: 30 / 900 / 900 / 60


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


SUBTRACT_TAGS = {"negative pair residual", "negative degree residual", "degree exhausted with pairs left",
                 "pairs exhausted with degree left", "r_x < d_x"}


def inconsistent_label(ex):
    a = ex.args[0] if ex.args else ("?",)
    tag = a[0] if isinstance(a, tuple) else str(a)
    if tag in SUBTRACT_TAGS:
        rule = a[-1]
        where = f"subtract after rule {rule}"
    elif tag == "degree without pairs":
        where = "node rules (D1/A/B)"
    elif tag == "negative slack":
        where = "pair rule P/E"
    elif tag == "no local configuration":
        where = "rule L (empty list)"
    else:
        where = "?"
    return tag, where


def comp_sig(C):
    return (tuple(int(x) for x in C.nodes), tuple(C.rows), tuple(int(x) for x in C.Rint))


class Pipeline:
    def __init__(self):
        self.cache = {}           # comp_sig -> dict(all_decided, cert=[(global tuple, lb)])
        self.cache_hits = 0

    def run(self, adj, deg, n, vol):
        """adj/deg are copied by the forcer.  Returns a record (outcome, stage, ...) and the certified Counter."""
        rec = dict()
        t0 = time.perf_counter()
        try:
            F = MO.WDForcerTL(adj, deg, rules=MO.RULES, time_limit=FORCE_TL, copy_adj=True).run()
        except WC.Inconsistent as ex:
            tag, where = inconsistent_label(ex)
            rec.update(outcome="detected", stage="forcing", check=f"Inconsistent: {tag}", where=where,
                       t_s=time.perf_counter() - t0)
            return rec, None
        rec["t_force_s"] = time.perf_counter() - t0
        rec["force_complete"] = bool(F.complete)
        rec["l_call_stops"] = F.l_call_stops
        forced = Counter({tuple(sorted(int(v) for v in e)): int(m) for e, m in F.forced.items()})
        forced_vol = sum(len(e) * m for e, m in forced.items())
        rec["forced_vol"] = int(forced_vol)

        def det(stage, check, where=""):
            rec.update(outcome="detected", stage=stage, check=check, where=where, t_s=time.perf_counter() - t0)
            return rec, None

        comp_nodes = EN._components(F.adj, n)
        comp_nodes.sort(key=len)
        comps = []
        try:
            for cn in comp_nodes:
                comps.append(WP.build_comp_wp(cn, F.adj, F.deg, cap=EN.CAP, enum_s=EN.ENUM_S))
        except AssertionError as ex:
            a = ex.args[0] if ex.args else "?"
            tag = a[0] if isinstance(a, tuple) else str(a)
            return det("components", f"build_comp_wp assertion: {tag}")
        if forced_vol + sum(C.vol for C in comps) != vol:
            return det("components", "pipeline volume assertion (forced + residual volume != vol)")
        if any(getattr(C, "rank_full_no_world", False) for C in comps):
            return det("components", "rank test: full column rank but no world (rank_full_no_world)")
        rec["comps"] = len(comps)
        rec["comps_over_cap"] = sum(1 for C in comps if C.cands is None)
        # world search, seed 0, as pipeline
        rng = np.random.default_rng(0)
        cert = Counter(forced)
        pinned_vol = 0
        uncovered = 0
        found = []
        wstat = Counter()
        for C in comps:
            if C.cands is None:
                uncovered += int(C.vol)
                wstat["not_enumerated:" + C.enum] += 1
                continue
            if C.unique and C.xu is not None:
                pinned_vol += int(C.vol)
                for j in np.nonzero(C.xu)[0]:
                    cert[tuple(sorted(int(C.nodes[i]) for i in C.cands[j]))] += int(C.xu[j])
                wstat["pinned"] += 1
                continue
            if len(C.cands) == 0:
                return det("worlds", "component without candidates (no world)")
            x, st = MO.world_of(C, rng, MILP_TL)
            wstat[st] += 1
            if x is None:
                if st == "nosol:Infeasible" or st == "unverified":
                    return det("worlds", f"world search: {st}")
                uncovered += int(C.vol)
                continue
            found.append((C, x))
        rec["world_status"] = dict(wstat)
        rec["uncovered_vol"] = uncovered
        # exact certification
        undecided = 0
        cert_new = 0
        for C, x in found:
            sig = comp_sig(C)
            hit = self.cache.get(sig)
            if hit is not None and hit["all_decided"]:
                self.cache_hits += 1
                for e, lb in hit["cert"]:
                    cert[e] += lb
                    cert_new += lb
                continue
            r = CX.certify_component(C, [x], budgets=BUDGETS)
            lst = []
            for c in r["cands"]:
                if c["lb"] > 0:
                    e = tuple(sorted(int(C.nodes[i]) for i in C.cands[c["j"]]))
                    lst.append((e, int(c["lb"])))
                undecided += int(c["x"][0]) - int(c["lb"]) if not c["decided"] else 0
            for e, lb in lst:
                cert[e] += lb
                cert_new += lb
            self.cache[sig] = dict(all_decided=bool(r["all_decided"]), cert=lst)
        rec.update(outcome="not_detected", stage="", check="", pinned_vol=int(pinned_vol),
                   cert_copies=int(sum(cert.values())), cert_vol=int(sum(len(e) * m for e, m in cert.items())),
                   cert_new_copies=int(cert_new), undecided_copies=int(undecided), t_s=time.perf_counter() - t0)
        return rec, cert


def stored(ds, mode):
    ou = json.load(open(MB.OUT / "ours" / f"{ds}_{mode}.json", encoding="utf-8"))
    ce = json.load(open(CFG.RESULTS / "certification" / "cert_exact.json", encoding="utf-8"))
    b = [x for x in ce["benchmark"] if x["ds"] == ds and x["mode"] == mode][0]
    return dict(vol=ou["vol"], forced_vol=ou["forced_vol"], forced_vol_share=ou["forced_vol_share"],
                cert_before=b["certified_vol_share_before"], cert_after=b["certified_vol_share_after"],
                ours_cert_vol_share_s0=ou["seeds"][0]["certified_vol_share"])


def cmd_run(a):
    ds, mode = a.ds, a.mode
    RES.mkdir(parents=True, exist_ok=True)
    outp = RES / (f"{ds}_{mode}.rev.jsonl" if a.reverse else f"{ds}_{mode}.jsonl")
    fwd = RES / f"{ds}_{mode}.jsonl"
    lines, truth = MB.target(ds, mode)
    nodes = sorted(set(v for e in lines for v in e))
    loc = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    lines_loc = [tuple(loc[v] for v in e) for e in lines]
    line_mult = Counter(tuple(sorted(e)) for e in lines_loc)
    adj, deg = WC.build_wdeg(lines_loc, n)
    vol = int(sum(deg))
    P = Pipeline()
    fo = open(outp, "w", encoding="utf-8")

    def write(r):
        fo.write(json.dumps(r) + "\n")
        fo.flush()

    # ---------------- sanity: zero perturbation
    st = stored(ds, mode)
    t0 = time.perf_counter()
    rec, cert = P.run(adj, deg, n, vol)
    ok = rec["outcome"] == "not_detected"
    san = dict(kind="none", ds=ds, mode=mode, n=n, vol=vol, stored=st, **rec)
    if ok:
        false_cert = int(sum(max(0, m - line_mult.get(e, 0)) for e, m in cert.items()))
        san["forced_vol_share"] = rec["forced_vol"] / vol
        san["cert_before_share"] = (rec["forced_vol"] + rec["pinned_vol"]) / vol
        san["cert_after_share"] = rec["cert_vol"] / vol
        san["match_forced"] = rec["forced_vol"] == st["forced_vol"] and vol == st["vol"]
        san["match_before"] = abs(san["cert_before_share"] - st["cert_before"]) < 1e-12
        san["match_after"] = abs(san["cert_after_share"] - st["cert_after"]) < 1e-12
        san["sanity_ok"] = bool(san["match_forced"] and san["match_before"] and san["match_after"]
                                and false_cert == 0 and rec["undecided_copies"] == 0)
    else:
        san["sanity_ok"] = False
    san["t_sanity_s"] = time.perf_counter() - t0
    write(san)
    log(f"{ds} {mode} sanity ok={san['sanity_ok']} forced {san.get('forced_vol_share')} vs {st['forced_vol_share']}, "
        f"before {san.get('cert_before_share')} vs {st['cert_before']}, after {san.get('cert_after_share')} vs "
        f"{st['cert_after']}, {san['t_sanity_s']:.1f}s")
    if not san["sanity_ok"]:
        log("SANITY FAILED: stop")
        fo.close()
        sys.exit(2)

    # ---------------- perturbation plan (fixed seed, uniform without replacement, before any trial runs)
    pairs = sorted((u, v) for u in range(n) for v in adj[u] if u < v)
    nodes_d = [u for u in range(n) if deg[u] >= 1]
    plan = {}
    for ki, kind in enumerate(KINDS):
        rng = np.random.default_rng([SEED_BASE, DS6.index(ds), MB.MODES.index(mode), ki])
        pool = pairs if kind in ("P+", "P-") else nodes_d
        k = min(a.trials, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        plan[kind] = [pool[int(i)] for i in idx]
    # interleave kinds (trial i of every kind before trial i + 1), so a time-out leaves balanced partial results
    order = [(kind, i) for i in range(max(len(v) for v in plan.values())) for kind in KINDS if i < len(plan[kind])]
    if a.reverse:
        # second worker on the same plan: walks the same trial list from its end and stops where the forward worker
        # (results/<ds>_<mode>.jsonl) already is; the summary merges both files and drops duplicates
        order = order[::-1]
    log(f"{ds} {mode}: {len(order)} trials planned, pairs {len(pairs)}, nodes {len(nodes_d)}"
        + (" (reverse worker)" if a.reverse else ""))
    done = 0
    for kind, i in order:
        if a.deadline and time.time() > a.deadline:
            log(f"deadline reached after {done} trials")
            break
        if a.reverse:
            seen = set()
            for x in open(fwd, encoding="utf-8"):
                if x.strip():
                    try:
                        q = json.loads(x)
                    except json.JSONDecodeError:
                        continue
                    seen.add((q["kind"], q.get("i")))
            if (kind, i) in seen:
                log(f"reverse worker met the forward worker after {done} trials")
                break
        ent = plan[kind][i]
        t1 = time.perf_counter()
        # apply the perturbation in place; the forcer copies adj/deg in its constructor; undo right after
        if kind in ("P+", "P-"):
            u, v = ent
            old = adj[u][v]
            new = old + (1 if kind == "P+" else -1)
            entry = dict(pair=[nodes[u], nodes[v]], old=old, new=new)
            if new == 0:
                del adj[u][v]
                del adj[v][u]
            else:
                adj[u][v] = adj[v][u] = new
            pvol = vol
        else:
            u = ent
            old = deg[u]
            new = old + (1 if kind == "D+" else -1)
            entry = dict(node=nodes[u], old=old, new=new)
            deg[u] = new
            pvol = vol + (new - old)
        try:
            rec, cert = P.run(adj, deg, n, pvol)
        except Exception as ex:  # an unexpected error is recorded, not hidden
            rec, cert = dict(outcome="error", stage="", check=repr(ex)[:300],
                             tb=traceback.format_exc()[-1500:]), None
        finally:
            if kind in ("P+", "P-"):
                adj[u][v] = adj[v][u] = old
            else:
                deg[u] = old
        r = dict(kind=kind, i=i, ds=ds, mode=mode, entry=entry, vol=pvol, **rec)
        r["t_trial_s"] = time.perf_counter() - t1
        write(r)
        done += 1
        if done % 50 == 0:
            log(f"{ds} {mode}: {done} trials, cache hits {P.cache_hits}")
    fo.close()
    # check that the base input was restored exactly
    adj2, deg2 = WC.build_wdeg(lines_loc, n)
    assert adj2 == adj and deg2 == deg, "base input not restored"
    log(f"{ds} {mode}: finished {done} trials")


# ------------------------------------------------------------------ summary
def load(ds, mode):
    p = RES / f"{ds}_{mode}.jsonl"
    if not p.exists():
        return None, []
    rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    san = rows[0] if rows and rows[0]["kind"] == "none" else None
    out = [r for r in rows if r["kind"] != "none"]
    prev = RES / f"{ds}_{mode}.rev.jsonl"
    rrows = [json.loads(x) for x in open(prev, encoding="utf-8") if x.strip()] if prev.exists() else []
    if rrows:
        assert rrows[0]["kind"] == "none" and rrows[0]["sanity_ok"], "reverse worker sanity"
        have = {(r["kind"], r["i"]) for r in out}
        for r in rrows[1:]:
            if (r["kind"], r["i"]) not in have:     # duplicates (same trial run by both workers): keep the forward one
                out.append(r)
                have.add((r["kind"], r["i"]))
    return san, out


def agg(rows):
    det = [r for r in rows if r["outcome"] == "detected"]
    nd = [r for r in rows if r["outcome"] == "not_detected"]
    return dict(trials=len(rows), detected=len(det), not_detected=len(nd),
                errors=sum(1 for r in rows if r["outcome"] == "error"),
                # a world of the changed input was found and checked exactly (forced copies + one checked world per
                # component, nothing over the cap): the changed input is consistent, so no check can reject it
                not_detected_with_world=sum(1 for r in nd if r.get("uncovered_vol", 0) == 0
                                            and set(r.get("world_status", {})) <= {"found", "pinned"}),
                detected_stage=dict(Counter(r["stage"] for r in det)))


def cmd_summary(a):
    S = dict(generated=time.strftime("%Y-%m-%d %H:%M"), kinds=KINDS, budgets=BUDGETS, seed_base=SEED_BASE, inputs={})
    allrows = []
    for mode in MB.MODES:
        for ds in DS6:
            san, rows = load(ds, mode)
            if san is None:
                continue
            S["inputs"][f"{ds}_{mode}"] = dict(sanity_ok=san.get("sanity_ok"), all=agg(rows),
                                               per_kind={kind: agg([r for r in rows if r["kind"] == kind])
                                                         for kind in KINDS})
            allrows += rows
    S["pooled"] = agg(allrows)
    MB.save_json(S, OUTD / "summary.json")
    g = S["pooled"]
    print(f"{len(S['inputs'])} inputs, sanity (unchanged input reproduces the stored shares): "
          f"{all(v['sanity_ok'] for v in S['inputs'].values())}")
    print(f"HyRange rejects {g['detected']:,} of the {g['trials']:,} changed inputs; the other {g['not_detected']} "
          f"have a world (checked exactly) in {g['not_detected_with_world']} cases; errors {g['errors']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("ds", nargs="?")
    ap.add_argument("mode", nargs="?")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.what == "run":
        cmd_run(a)
    else:
        cmd_summary(a)


if __name__ == "__main__":
    main()
