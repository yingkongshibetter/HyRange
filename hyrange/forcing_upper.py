"""Forcing in the upper-bound mode.  Same code as forcing.py, plus the option singleton=True: the diagonal D_u is
read as an upper bound on the degree d_u (the groups of one member also count in D), so every rule reads the
bounds ell_u = max_v W_uv <= d_u <= U_u = min(D_u, r_u) in place of the degree: D1 applies if U_u = 1, A if
W_uv = U_u for every neighbour v, B if r_u = ell_u, and E (P in the code) uses U in the intersection bound and
the slack r_x - R(|K| - 1) - max(0, ell_x - R).  Rule L is not used in this mode.  With singleton=False the
class behaves as forcing.WDForcer.  Rule E is called P in the code.

View.  For a multihypergraph H (hyperedges of >= 2 members, repeats allowed):
    R_uv = #{e in H : u, v in e}   (u != v)       pair counts (the multiplicity graph W)
    d_u  = #{e in H : u in e}                      hyperdegree (the diagonal)
Worlds of a release (R, d): all multihypergraphs with sizes >= 2 and exactly these entries.
Derived per node: r_u = sum_v R_uv = sum_{e ni u} (|e| - 1)   (so sum_{e ni u} |e| = r_u + d_u),
                  vol = sum_u d_u = sum_e |e| (fixed in every world).

Everything is exact integer arithmetic.  The forcing works on the residual release (release minus the forced
multiset F).  If F lies in every world, the worlds of the residual are exactly {H - F : H a world}, so a rule that
is sound on the residual is sound on the original release (induction over applications).

Rules of the exact mode (proofs in the experiments; P is rule E of the experiments):
  D1  d_u = 1                                  -> 1 copy of S_u = {u} + N(u)
  A   R_uv = d_u for every v in N(u)           -> d_u copies of S_u
  B   r_u = d_u                                -> R_uv copies of {u, v} for every v in N(u)
  P   pair rule on (u, v), R = R_uv > 0:
        K := {u, v}; repeat: a common neighbour w of u, v adjacent to all of K joins K if
             LB(w) := max(R_uw + R_vw - d_w,  max_{x in K} (R + R_xw - d_x)) >= R
        (every hyperedge through u, v contains w); X := common neighbours adjacent to all of K, not in K.
        budget := min( min_{x in K} [r_x - R(|K|-1) - (d_x - R)],  sum_{w in X} min(R, min_{x in K} R_xw) )
                  (0 if X is empty)
        -> max(0, R - budget) copies of K
  L   local star rule on u with d_u <= LMAX_D, |N(u)| <= LMAX_N: enumerate all multisets of d_u nonempty
      subsets T_1..T_d of N(u) with  #{i: v in T_i} = R_uv,  LB_vw <= #{i: v, w in T_i} <= R_vw  and
      sum_{i: v in T_i} |T_i| <= r_v - (d_v - R_uv); a set T that appears >= m times in every enumerated
      multiset gives m forced copies of {u} + T.
  G   (support only): {u, v} with no common neighbour -> 1 copy (complete for G).
  W   (pair counts only): copies of {u, v} >= R_uv - sum_{w common} min(R_uv, R_uw, R_vw)   (sound and
      complete in this multiset model; no hyperedge of >= 3 members is forced under W or G at all).
"""
from __future__ import annotations

import time
from collections import Counter

LMAX_D = 4
LMAX_N = 18
LMAX_LEAVES = 20000


# ------------------------------------------------------------------ view
def build_wdeg(edges, n):
    """adj[u] = {v: R_uv}, deg[u] = d_u from a list of hyperedges (tuples; repeats are copies)."""
    adj = [dict() for _ in range(n)]
    deg = [0] * n
    for e, m in Counter(tuple(sorted(e)) for e in edges).items():
        for x in e:
            deg[x] += m
            ax = adj[x]
            for y in e:
                if y != x:
                    ax[y] = ax.get(y, 0) + m
    return adj, deg


class Inconsistent(AssertionError):
    pass


# ------------------------------------------------------------------ W+deg forcing
class WDForcer:
    def __init__(self, adj, deg, rules=("D1", "A", "B", "P", "L"), time_limit=None, lmax_d=LMAX_D,
                 lmax_n=LMAX_N, lmax_leaves=LMAX_LEAVES, copy_adj=True, singleton=False):
        self.singleton = singleton
        self.adj = [dict(a) for a in adj] if copy_adj else adj
        self.deg = list(deg)
        self.n = len(deg)
        self.r = [sum(a.values()) for a in self.adj]
        self.ub = [min(d,r) for d,r in zip(self.deg,self.r)] if singleton else None
        self.lb = [max(a.values(),default=0) for a in self.adj] if singleton else None
        self.rules = set(rules)
        self.forced = Counter()
        self.counts = Counter()
        self.time_limit = time_limit
        self.complete = True
        self.lmax_d, self.lmax_n, self.lmax_leaves = lmax_d, lmax_n, lmax_leaves
        self.l_skipped = 0
        self.l_runs = 0

    def upper(self, u):
        return self.ub[u] if self.singleton else self.deg[u]

    def lower(self, u):
        return self.lb[u] if self.singleton else self.deg[u]

    # -------------------------------------------------------------- residual update
    def subtract(self, e, m, rule, touched):
        adj, deg, r = self.adj, self.deg, self.r
        k = len(e)
        if m <= 0:
            return
        for x in e:
            ax = adj[x]
            for y in e:
                if y == x:
                    continue
                v = ax.get(y, 0) - m
                if v < 0:
                    raise Inconsistent(("negative pair residual", x, y, e, m, rule))
                if v == 0:
                    del ax[y]
                else:
                    ax[y] = v
            deg[x] -= m
            r[x] -= m * (k - 1)
            if self.singleton:
                self.ub[x] = min(deg[x],r[x])
                self.lb[x] = max(ax.values(),default=0)
            if deg[x] < 0 or r[x] < 0:
                raise Inconsistent(("negative degree residual", x, e, m, rule))
            touched.add(x)
        for x in e:
            if deg[x] == 0 and adj[x]:
                raise Inconsistent(("degree exhausted with pairs left", x, rule))
            if not self.singleton and deg[x] > 0 and not adj[x]:
                raise Inconsistent(("pairs exhausted with degree left", x, rule))
            if not self.singleton and r[x] < deg[x]:
                raise Inconsistent(("r_x < d_x", x, rule))
        self.forced[e] += m
        self.counts[rule] += m

    # -------------------------------------------------------------- node rules
    def node_rules(self, u, touched):
        d = self.upper(u)
        if d == 0:
            return False
        a = self.adj[u]
        if not a:
            raise Inconsistent(("degree without pairs", u))
        if "D1" in self.rules and d == 1:
            self.subtract(tuple(sorted(list(a) + [u])), 1, "D1", touched)
            return True
        if "A" in self.rules and all(x == d for x in a.values()):
            self.subtract(tuple(sorted(list(a) + [u])), d, "A", touched)
            return True
        if "B" in self.rules and self.r[u] == self.lower(u):
            for v, m in list(a.items()):
                self.subtract((min(u, v), max(u, v)), m, "B", touched)
            return True
        return False

    # -------------------------------------------------------------- pair rule
    def pair_core(self, u, v):
        """(K, X, R) for the pair u, v of the residual."""
        adj, deg = self.adj, self.deg
        au, av = adj[u], adj[v]
        R = au[v]
        if len(au) > len(av):
            small, big = av, au
        else:
            small, big = au, av
        cand = [w for w in small if w in big and w != u and w != v]
        K = [u, v]
        changed = True
        while changed and cand:
            changed = False
            keep = []
            for w in cand:
                aw = adj[w]
                ok = True
                lb = au[w] + av[w] - self.upper(w)
                for x in K:
                    rxw = aw.get(x, 0)
                    if rxw == 0:
                        ok = False
                        break
                    t = R + rxw - self.upper(x)
                    if t > lb:
                        lb = t
                if not ok:
                    changed = True
                    continue
                if lb >= R:
                    K.append(w)
                    changed = True
                else:
                    keep.append(w)
            cand = keep
        # final filter: X must be adjacent to all of K (K may have grown after w was checked)
        X = []
        for w in cand:
            aw = adj[w]
            if all(x in aw for x in K):
                X.append(w)
        return K, X, R

    def pair_rule(self, u, v, touched):
        if "P" not in self.rules:
            return False
        K, X, R = self.pair_core(u, v)
        r, deg, adj = self.r, self.deg, self.adj
        kk = len(K)
        slack = min(r[x] - R * (kk - 1) - max(0, self.lower(x) - R) for x in K)
        if slack < 0:
            raise Inconsistent(("negative slack", u, v, K))
        if not X:
            budget = 0
        else:
            ub = 0
            for w in X:
                aw = adj[w]
                ub += min([R] + [aw[x] for x in K])
                if ub >= slack:
                    break
            budget = min(slack, ub)
        m = R - budget
        if m > 0:
            self.subtract(tuple(sorted(K)), m, "P", touched)
            return True
        return False

    # -------------------------------------------------------------- local star rule
    def local_rule(self, u, touched):
        if "L" not in self.rules:
            return False
        d = self.upper(u)
        a = self.adj[u]
        if d < 2 or len(a) > self.lmax_n or d > self.lmax_d:
            return False
        sols = self.local_enum(u)
        if sols is None:
            self.l_skipped += 1
            return False
        self.l_runs += 1
        if not sols:
            raise Inconsistent(("no local configuration", u))
        common = None
        for s in sols:
            c = Counter(s)
            common = c if common is None else (common & c)
            if not common:
                return False
        fired = False
        for T, m in sorted(common.items(), key=lambda t: sorted(t[0])):
            if not T:
                continue  # padding slots are not forced singleton copies
            e = tuple(sorted(list(T) + [u]))
            # every hyperedge in common is a true hyperedge of every world; subtract it
            self.subtract(e, m, "L", touched)
            fired = True
        return fired

    def local_enum(self, u):
        """All local configurations at u (multisets of d_u frozensets), or None over the leaf budget."""
        adj, deg, r = self.adj, self.deg, self.r
        d = self.upper(u)
        au = adj[u]
        N = sorted(au, key=lambda v: (-au[v], v))
        idx = {v: i for i, v in enumerate(N)}
        nn = len(N)
        # pair bounds among neighbours
        lo = [[0] * nn for _ in range(nn)]
        hi = [[0] * nn for _ in range(nn)]
        for i, v in enumerate(N):
            av = adj[v]
            for j in range(i):
                w = N[j]
                rvw = av.get(w, 0)
                hi[i][j] = hi[j][i] = min(rvw, au[v], au[w])
                lb = max(0, au[v] + rvw - self.upper(v), au[w] + rvw - self.upper(w), au[v] + au[w] - d)
                lo[i][j] = lo[j][i] = lb
                if lb > hi[i][j]:
                    return []
        cap = [r[v] - max(0, self.lower(v) - au[v]) for v in N]
        slots = [0] * d          # bitmask of members (by neighbour index)
        sizes = [0] * d
        assign = [None] * nn     # tuple of slot indices per neighbour
        out = set()
        leaves = [0]
        # Once even a subset of admissible stars has no common nonempty edge,
        # the intersection over ALL stars is necessarily empty. Exact early exit.
        common_nonempty = [None]
        no_common = [False]

        def classes():
            """Groups of slots with identical content so far (slot order)."""
            seen = {}
            for s in range(d):
                seen.setdefault(slots[s], []).append(s)
            return list(seen.values())

        def choose(cls, R):
            """Compositions: pick k_c from each class (first k_c slots), sum k_c = R."""
            res = []

            def rec(ci, left, cur):
                if ci == len(cls):
                    if left == 0:
                        res.append(tuple(cur))
                    return
                rem = sum(len(c) for c in cls[ci + 1:])
                for k in range(min(left, len(cls[ci])), -1, -1):
                    if left - k > rem:
                        break
                    rec(ci + 1, left - k, cur + cls[ci][:k])
            rec(0, R, [])
            return res

        def rec(i):
            if leaves[0] > self.lmax_leaves or no_common[0]:
                return
            if i == nn:
                leaves[0] += 1
                if not self.singleton and any(sizes[s] == 0 for s in range(d)):
                    return
                for j in range(nn):
                    if sum(sizes[s] for s in assign[j]) > cap[j]:
                        return
                key = tuple(sorted(tuple(sorted(N[b] for b in range(nn) if slots[s] >> b & 1)) for s in range(d)))
                out.add(key)
                c = Counter(t for t in key if t)
                common_nonempty[0] = c if common_nonempty[0] is None else common_nonempty[0] & c
                no_common[0] = not common_nonempty[0]
                return
            # remaining-capacity prune: slots still empty must be filled by later neighbours
            empty = sum(1 for s in range(d) if sizes[s] == 0)
            if not self.singleton and empty > sum(au[N[j]] for j in range(i, nn)):
                return
            R = au[N[i]]
            for S in choose(classes(), R):
                ok = True
                Sset = set(S)
                for j in range(i):
                    c = sum(1 for s in assign[j] if s in Sset)
                    if c < lo[i][j] or c > hi[i][j]:
                        ok = False
                        break
                if not ok:
                    continue
                for s in S:
                    slots[s] |= 1 << i
                    sizes[s] += 1
                # partial size check for already placed neighbours (sizes only grow)
                bad = False
                for j in range(i + 1):
                    sj = assign[j] if j < i else S
                    if sum(sizes[s] for s in sj) > cap[j]:
                        bad = True
                        break
                if not bad:
                    assign[i] = S
                    rec(i + 1)
                    assign[i] = None
                for s in S:
                    slots[s] &= ~(1 << i)
                    sizes[s] -= 1
                if leaves[0] > self.lmax_leaves or no_common[0]:
                    return

        rec(0)
        if leaves[0] > self.lmax_leaves:
            return None
        return [tuple(frozenset(t) for t in key) for key in out]

    # -------------------------------------------------------------- driver
    def run(self, order=None, log=None):
        t0 = time.perf_counter()
        n = self.n
        pos = list(range(n))
        if order is not None:
            for i, u in enumerate(order):
                pos[u] = i
        pending = sorted((u for u in range(n) if self.deg[u] > 0), key=lambda u: pos[u])
        rounds = 0
        use_L = "L" in self.rules
        while pending:
            rounds += 1
            touched_all = set()
            progress_at = time.perf_counter()
            for ui, u in enumerate(pending):
                if log and time.perf_counter() - progress_at >= 30:
                    log(f"  progress round {rounds}: node {ui}/{len(pending)}, forced {sum(self.forced.values())}, "
                        f"local runs {self.l_runs}, skips {self.l_skipped}, elapsed {time.perf_counter()-t0:.1f}s")
                    progress_at = time.perf_counter()
                if self.time_limit is not None and time.perf_counter() - t0 > self.time_limit:
                    self.complete = False
                    break
                if self.deg[u] == 0:
                    continue
                touched = set()
                self.node_rules(u, touched)
                if self.deg[u] > 0:
                    for v in list(self.adj[u]):
                        if self.deg[u] == 0:
                            break
                        if v in self.adj[u]:
                            self.pair_rule(u, v, touched)
                if use_L and self.deg[u] > 0 and not touched:
                    self.local_rule(u, touched)
                touched_all |= touched
            if not self.complete:
                break
            nb = set(touched_all)
            for x in touched_all:
                nb.update(self.adj[x])
            pending = sorted((u for u in nb if self.deg[u] > 0), key=lambda u: pos[u])
            if log:
                log(f"  round {rounds}: forced {sum(self.forced.values())} copies, pending {len(pending)}, "
                    f"{time.perf_counter() - t0:.1f}s")
        self.rounds = rounds
        self.t = time.perf_counter() - t0
        return self


# ------------------------------------------------------------------ G and W forcing
def force_G(adj):
    """{u, v} with no common neighbour in the support: 1 copy (complete for the unweighted view G)."""
    forced = Counter()
    for u, a in enumerate(adj):
        for v in a:
            if u < v:
                av = adj[v]
                small, big = (a, av) if len(a) <= len(av) else (av, a)
                if not any(w in big for w in small if w != u and w != v):
                    forced[(u, v)] += 1
    return forced


def force_W(adj, time_limit=None):
    """Pair-count rule with cascade: copies of {u, v} >= R_uv - sum_{w common} min(R_uv, R_uw, R_vw)."""
    adj = [dict(a) for a in adj]
    n = len(adj)
    forced = Counter()
    t0 = time.perf_counter()
    pending = list(range(n))
    complete = True
    while pending:
        touched = set()
        for u in pending:
            if time_limit is not None and time.perf_counter() - t0 > time_limit:
                complete = False
                break
            for v in list(adj[u]):
                if v not in adj[u]:
                    continue
                au, av = adj[u], adj[v]
                R = au[v]
                small, big = (au, av) if len(au) <= len(av) else (av, au)
                ub = 0
                for w in small:
                    if w != u and w != v and w in big:
                        ub += min(R, au[w], av[w])
                        if ub >= R:
                            break
                m = R - ub
                if m > 0:
                    for x, y in ((u, v), (v, u)):
                        nv = adj[x][y] - m
                        if nv == 0:
                            del adj[x][y]
                        else:
                            adj[x][y] = nv
                    forced[(min(u, v), max(u, v))] += m
                    touched.update((u, v))
        if not complete:
            break
        nb = set(touched)
        for x in touched:
            nb.update(adj[x])
        pending = sorted(nb)
    return forced, adj, complete


# ------------------------------------------------------------------ checks
def verify_forced(forced, true_edges):
    true = Counter(tuple(sorted(e)) for e in true_edges)
    invalid = sum(max(0, m - true.get(e, 0)) for e, m in forced.items())
    return dict(forced_copies=sum(forced.values()), invalid=invalid, ok=invalid == 0)
