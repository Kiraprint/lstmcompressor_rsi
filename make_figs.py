"""Generate figures for the report (PNG, 200 dpi, no external deps beyond matplotlib)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figs")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

res = json.loads(Path("final_results.json").read_text())
base_bytes = Path("bench_baseline/data/enwik5_compressed.dat").stat().st_size
N = 100000


def name(r):
    return r["name"].replace("_", " ")


# ---- 1. ratio vs time: full landscape + zoom on the feasible region ---------
SHORT = {"baseline_reference": "baseline", "G_LLM_GPT2_prior_fast": "GPT-2 prior"}


def short(r):
    n = r["name"]
    return SHORT.get(n, n.split("_")[0])


ref = N / base_bytes
tgt = ref * 1.10
budget = 27.0 * 1.5

fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.3))


def rules(ax):
    ax.axhline(ref, color="crimson", ls="--", lw=1.1,
               label=f"baseline {ref:.3f}")
    ax.axhline(tgt, color="green", ls=":", lw=1.1, label="+10 % target")
    ax.axvline(budget, color="orange", ls=":", lw=1.1, label="+50 % time")


# left: everything, log time so the 8 s and 483 s runs both fit
for r in res:
    axL.scatter(r["total_s"], r["ratio"], s=46, zorder=3,
                marker="o" if r["ok"] else "x")
    axL.annotate(short(r), (r["total_s"], r["ratio"]), fontsize=8,
                 xytext=(4, 3), textcoords="offset points")
axL.scatter(27.0, ref, s=90, marker="*", color="crimson", zorder=4)
axL.annotate("baseline", (27.0, ref), fontsize=8, xytext=(4, -9),
             textcoords="offset points")
axL.set_xscale("log")
axL.set_xlim(6, 800)
axL.set_ylim(2.85, 7.25)
rules(axL)
axL.set_xlabel("total time, encode + decode (s, log)")
axL.set_ylabel("compression ratio")
axL.set_title("all runs")

# right: only the runs that fit the time budget, y zoomed on the real spread
for r in res:
    if r["total_s"] > 60 or r["ratio"] < 2.95:
        continue
    axR.scatter(r["total_s"], r["ratio"], s=46, zorder=3,
                marker="o" if r["ok"] else "x")
    axR.annotate(f"{short(r)}  {r['ratio']:.4f}",
                 (r["total_s"], r["ratio"]), fontsize=8,
                 xytext=(4, 3), textcoords="offset points")
axR.scatter(27.0, ref, s=90, marker="*", color="crimson", zorder=4)
axR.annotate(f"baseline  {ref:.4f}", (27.0, ref), fontsize=8,
             xytext=(4, -10), textcoords="offset points")
axR.set_xlim(0, 58)
axR.set_ylim(2.96, 3.14)
rules(axR)
axR.set_xlabel("total time (s)")
axR.set_title("zoom: runs within the +50 % time budget")
axR.legend(fontsize=8, loc="lower left")

fig.suptitle("Accuracy / cost trade-off", y=1.02, fontsize=11.5)
fig.tight_layout()
fig.savefig(OUT / "pareto.png")
plt.close(fig)

# ---- 2. bits-per-byte convergence ------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.1))
for r in [res[0]] + [x for x in res if x["name"].endswith("FINAL")]:
    c = r.get("curve") or []
    if c:
        a = np.array(c)
        ax.plot(a[:, 0], a[:, 1], lw=2.0, label=name(r))
ax.set_xlabel("coding step (byte position within a stream)")
ax.set_ylabel("cumulative bits / byte")
ax.set_title("Online convergence, no pre-training")
ax.legend(fontsize=9)
fig.savefig(OUT / "curve.png")
plt.close(fig)

# ---- 3. bar chart of sizes --------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.1))
labels = ["reference"] + [name(r) for r in res]
vals = [base_bytes] + [r["bytes"] for r in res]
cols = ["crimson"] + ["steelblue"] * len(res)
ax.barh(labels, vals, color=cols)
ax.axvline(base_bytes * 0.9, color="green", ls=":", lw=1.5, label="target size")
ax.set_xlabel("compressed bytes")
ax.invert_yaxis()
ax.legend(fontsize=9)
fig.savefig(OUT / "bars.png")
plt.close(fig)
print("figures ->", OUT)
