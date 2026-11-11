# Running the experiments

Use Python 3.12 and install `requirements.txt`. Run the commands below from the repository root. Results are written to `results/` by default. Set `HYRANGE_RESULTS` to use a different directory.

## Start with one benchmark

```bash
python data/get_marioh.py
python -I -B -X utf8 experiments/hyrange_bench.py hosts preserved
python -I -B -X utf8 experiments/exact_certification.py bench hosts preserved
```

The first command fetches a pinned upstream revision and checks the input files against their SHA-256 checksums. The next two write `results/bench/ours/hosts_preserved.json` and `results/certification/cert_exact/bench_hosts_preserved.json`. Use `reduced` for the setting without repeated copies. Run `python -I -B -X utf8 tests/test_soundness.py` to check certification against brute-force enumeration on small inputs.

## Other experiments

The commands below use the prefix `python -I -B -X utf8 experiments/`. Replace `<dataset>`, `<mode>`, and `<source>` with the values described in the script's docstring. Scripts that summarize stored outputs run after the corresponding jobs.

| Experiment | Scripts and arguments |
|---|---|
| Target statistics | `table1_targets.py` |
| Reconstruction and dense benchmarks | `run_benchmark.py --modes preserved reduced`, `hybrid.py all`, `bench_summary.py` |
| Pair-count-only certification | `pairs_only.py` |
| Exact certification | `exact_certification.py bench <dataset> <mode>`, `same_input_summary.py cert` |
| Upper-bound certification | `upper_forcing.py queue`, `upper_forcing.py summary`, `upper_exact.py queue bench_pres bench_red`, `upper_exact_summary.py` |
| Refutation | `refutation.py <dataset> <mode>`, `bench_summary.py` |
| Integer program, CP-SAT, and D1 plus MARIOH | `ip_baseline.py <dataset> <mode>`, `cpsat_baseline.py run <dataset> <mode> a:0,a:1,a:2`, `cpsat_summary.py`, `d1_marioh.py` (`prep`, `jobs`, `summary`), `same_input_summary.py same` |
| Rule ablations | `fewer_rules.py <dataset> <mode> none`, `fewer_rules.py <dataset> <mode> d1`, `fewer_rules.py <dataset> <mode> full`, `fewer_rules_summary.py`, `rule_ablation.py all`, `fig4_rule_steps.py` |
| Enumeration | `enumeration.py run deg2 6,8,10,12,15,20,25,30,40,60,100` (also `deg3` and `deg6`), `enumeration_summary.py` |
| Co-authorship exports | `exports.py queue`, `exports.py summary`, `upper_exact.py job export <source> 5`, `upper_exact_summary.py`, `exports_table.py`, `fig2_false_teams.py` |
| Deposited co-authorship map | `python data/get_zenodo.py`, then `zenodo_exports.py` (see the script's usage) |
| Local certification | `dense_samples.py plan`, `dense_samples.py drive`, `local_cert.py m4all`, `local_cert.py drive --levels node --wall 900 --tag _node`, `local_cert.py drive --levels all --wall 1800 --tag _lv`, `local_cert_summary.py` |
| Perturbation and label precision | `perturb.py run <dataset> <mode>`, `perturb.py summary`, `label_precision.py` |

The OpenAlex samples are included. [Data setup](data/README.md) describes the other sources. To run the MARIOH baseline, install `requirements-marioh.txt` in a separate environment and set `HYRANGE_MARIOH_PYTHON` to that environment's Python. Use `requirements-cpsat.txt` for CP-SAT. For TwinSearch, follow [its build instructions](experiments/twinsearch/README.md).

The benchmark configuration uses seeds 0, 1, and 2 and one CPU thread per run. Solver and per-job limits are defined in the source. The more expensive baseline runs can take an hour or longer.
