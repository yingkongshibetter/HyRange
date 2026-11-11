"""Fig. 2: teams read off a co-authorship export as maximal cliques that are not real teams.

Source: results/exports/sim_export.json (records[].cliques.clique and records[].upper.false_copies).
usage: python -I -B -X utf8 experiments/fig2_false_teams.py
For each of the five sources, at t = 1 (all authors) and t = 5 (authors with at least five documents):
  false share = 1 - (maximal cliques with >= 3 members that are the set of retained authors of a document) /
                (maximal cliques with >= 3 members).
The certified copies of the upper-bound mode have 0 false copies in all 25 exports (asserted below).
Writes results/figures/motiv_export_c.pdf and .png.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyrange import config as CFG  # noqa: E402

EXPORTS_JSON = str(CFG.RESULTS / "exports" / "sim_export.json")
FIGS = str(CFG.RESULTS / "figures")
Path(FIGS).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})

d = json.load(open(EXPORTS_JSON, encoding="utf-8"))
assert d["complete"] and not d["upper_soundness_violations"]
recs = {(r["name"], int(r["t"])): r for r in d["records"]}
assert all(int(r["upper"]["false_copies"]) == 0 for r in d["records"]) and len(d["records"]) == 25

names = ["DBLP", "MAG-TopCS", "OpenAlex-DB", "OpenAlex-CS-3+", "OpenAlex-Bio"]
short = ["DBLP", "MAG-TopCS", "OA-DB", "OA-CS", "OA-Bio"]
TS = [(1, "all authors", "#9ecae1"), (5, "authors with at least five documents", "#2171b5")]


def false_share(name, t):
    c = recs[(name, t)]["cliques"]["clique"]
    assert c["finished"]
    n, true = c["maximal_cliques_ge3"], c["maximal_cliques_ge3_true_distinct"]
    return 100.0 * (n - true) / n, n, true


fig, ax = plt.subplots(figsize=(3.45, 1.36))
w = 0.36
for j, (t, lab, col) in enumerate(TS):
    for i, nm in enumerate(names):
        v, n, true = false_share(nm, t)
        x = i + (j - 0.5) * (w + 0.03)
        ax.bar(x, v, w, color=col, linewidth=0, zorder=3, label=lab if i == 0 else None)
        ax.text(x, v + 1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#222222")
        print(nm, t, f"{v:.2f}", n, true)
ax.axhline(0, color="#111111", linewidth=0.8, zorder=4)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short)
ax.set_ylim(0, 46)
ax.set_yticks([0, 20, 40])
ax.set_ylabel("False teams (%)")
ax.yaxis.grid(True, color="#dddddd", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", length=0)
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc="lower left", frameon=False, handlelength=1.0, handletextpad=0.4, borderaxespad=0.0,
          bbox_to_anchor=(0.0, 1.03), ncol=2, columnspacing=1.0)
fig.subplots_adjust(left=0.115, right=0.995, top=0.85, bottom=0.13)
fig.savefig(FIGS + "/motiv_export_c.pdf")
fig.savefig(FIGS + "/motiv_export_c.png", dpi=300)
t5 = [false_share(nm, 5)[0] for nm in names]
print(f"t = 5: {min(t5):.1f}% to {max(t5):.1f}% of the maximal cliques of three or more authors are no document set")
print("written")
