"""Fig. 4: share of the 2-core certified by each step (paragraphs Inside the 2-core, Rules E and L).

One stacked horizontal bar per benchmark with a nonempty 2-core in both settings (Hosts, DBLP, MAG-TopCS), preserved
and reduced.  A bar is the volume of the 2-core (100 percent), split by the step that certifies it: rules A and B,
rule E (rule P in the files), rule L, exact certification, and the rest, which exact certification labels ambiguous
(Table VI has no undecided copy).  Rule D1 certifies exactly the volume outside the 2-core (asserted), so it has no
segment.
  segment of a step = (share after the step - share before it) / (1 - share after D1)
  shares after D1, +A,B, +E, +L : results/bench/ablation_summary.json records[].shares
  share after exact certification: results/certification/cert_exact.json benchmark[].certified_vol_share_after
Writes results/figures/rules_c.pdf and .png and prints the numbers of the two paragraphs.

usage: python -I -B -X utf8 experiments/fig4_rule_steps.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyrange import config as CFG  # noqa: E402

ABL = CFG.RESULTS / "bench" / "ablation_summary.json"
CERT = CFG.RESULTS / "certification" / "cert_exact.json"
FIGS = CFG.RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})

SHOWN = ["Hosts", "DBLP", "MAG-TopCS"]
MODES = ["preserved", "reduced"]
SETS = ["D1", "D1+A+B", "D1+A+B+P", "D1+A+B+P+L"]


abl = {(r["name"], r["mode"]): r for r in json.load(open(ABL, encoding="utf-8"))["records"]}
cert = {(r["name"], r["mode"]): r for r in json.load(open(CERT, encoding="utf-8"))["benchmark"]}

seg = {}
for k in [(name, mode) for mode in MODES for name in SHOWN]:
    a = abl[k]
    assert a["d1_equals_two_core"] and abs(a["shares"]["D1"] - a["two_core_outside_share"]) < 1e-12
    v = [a["shares"][s] for s in SETS] + [cert[k]["certified_vol_share_after"]]
    assert all(v[i] <= v[i + 1] + 1e-12 for i in range(4)) and v[0] < 1
    core = 1.0 - v[0]
    parts = [100 * (v[i + 1] - v[i]) / core for i in range(4)]
    parts.append(100.0 - sum(parts))
    seg[k] = parts
    print(f"{k[0]} {k[1]}: 2-core {100 * core:.1f}% of the volume, {sum(parts[:4]):.1f}% of it certified; "
          f"A,B {parts[0]:.1f}, E {parts[1]:.1f}, L {parts[2]:.1f}, exact {parts[3]:.1f}, ambiguous {parts[4]:.1f}")
e = [seg[(nm, md)][1] for nm in ("DBLP", "MAG-TopCS") for md in MODES]
lmax = max((seg[k][2], k) for k in seg)
print(f"rule E: {min(e):.1f}% to {max(e):.1f}% of the 2-core on DBLP and MAG-TopCS; rule L: up to {lmax[0]:.1f}% "
      f"({lmax[1][0]}, {lmax[1][1]})")

COLS = ["#1baf7a", "#eb6834", "#2a78d6", "#33256b", "#d4d4d4"]
LABS = ["A, B", "E", "L", "Exact", "Ambiguous"]
INK = ["white", "white", "white", "white", "#333333"]
H = 0.62
ROWS = [(nm, md) for md in MODES for nm in SHOWN]
YPOS = [5.6, 4.8, 4.0, 2.4, 1.6, 0.8]

fig, ax = plt.subplots(figsize=(3.45, 1.62))
for (nm, md), y in zip(ROWS, YPOS):
    left = 0.0
    for j, p in enumerate(seg[(nm, md)]):
        if p <= 0:
            continue
        ax.barh(y, p, H, left=left, color=COLS[j], edgecolor="white", linewidth=0.8, zorder=3)
        if p >= 9.0:
            ax.text(left + p / 2, y - 0.02, "%.0f" % p, ha="center", va="center", fontsize=8, color=INK[j], zorder=4)
        left += p
ax.set_yticks(YPOS)
ax.set_yticklabels([nm for nm, _ in ROWS])
ax.set_ylim(0.3, 6.1)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of the 2-core (%)", labelpad=2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0, pad=3)
ax.tick_params(axis="x", length=2, pad=2)
for y, lab in ((4.8, "Preserved"), (1.6, "Reduced")):
    ax.text(-0.345, y, lab, transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center", fontsize=8,
            fontstyle="italic")

handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.5) for c in COLS]
fig.legend(handles, LABS, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, handlelength=1.0,
           handleheight=0.8, handletextpad=0.35, columnspacing=1.0, borderaxespad=0.15)
fig.subplots_adjust(left=0.265, right=0.975, top=0.875, bottom=0.215)
fig.savefig(FIGS / "rules_c.pdf")
fig.savefig(FIGS / "rules_c.png", dpi=300)
print("written")
