"""HyRange: hyperedge certification from pair counts and node degrees.

Modules (Sections III and IV of the experiments):
  forcing        exact-mode forcing rules D1, A, B, E (called P in the code) and L
  forcing_upper  forcing rules D1, A, B, E of the upper-bound mode
  candidates     candidate hyperedges, their filters and the rank test of a residual component
  engine         HiGHS calls, clique enumeration, components, exact world check, rigorous dual bound
  pipeline       forcing + components + world search (one world per seed) on a benchmark input
  certify        exact certification of a component: enumeration, dual bound, complete (refutation) search
  upper_exact    exact certification in the upper-bound mode (slack columns)
  levels         local certification (level-B systems, node and edge levels)
  refute         refutation of the copies of another reconstruction
  pairs_only     the closed form of the hyperedges certain under the pair counts alone
  quick          HyRange on a small input in one call
  bench, exports, upper, dense_samples, views, config   inputs and drivers of the experiments
  brute, levels_check   brute-force checks on small random inputs (used by the tests)
"""
