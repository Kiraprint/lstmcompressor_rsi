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
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

res = json.loads(Path("final_results.json").read_text())
base_bytes = Path("bench_baseline/data/enwik5_compressed.dat").stat().st_size
N = 100000


def name(r):
    return r["name"].replace("_", " ")


# ---- 1. ratio vs time scatter ------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for r in res:
    ok = r["ok"]
    ax.scatter(r["total_s"], r["ratio"], s=55,
               marker="o" if ok else "x", zorder=3)
    ax.annotate(name(r), (r["total_s"], r["ratio"]), fontsize=6.5,
                xytext=(4, 4), textcoords="offset points")
ax.axhline(N / base_bytes, color="crimson", ls="--", lw=1.2,
           label=f"reference {N/base_bytes:.3f}")
ax.axhline(N / base_bytes * 1.10, color="green", ls=":", lw=1.2,
           label="+10 % target")
ax.axvline(27.0 * 1.5, color="orange", ls=":", lw=1.2, label="+50 % time budget")
ax.set_xlabel("total time, encode + decode (s)")
ax.set_ylabel("compression ratio")
ax.set_title("Accuracy / cost trade-off")
ax.legend(fontsize=7)
fig.savefig(OUT / "pareto.png")
plt.close(fig)

# ---- 2. bits-per-byte convergence ------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for r in [res[0]] + [x for x in res if x["name"].endswith("FINAL")]:
    c = r.get("curve") or []
    if c:
        a = np.array(c)
        ax.plot(a[:, 0], a[:, 1], lw=1.6, label=name(r))
ax.set_xlabel("coding step (byte position within a stream)")
ax.set_ylabel("cumulative bits / byte")
ax.set_title("Online convergence, no pre-training")
ax.legend(fontsize=7)
fig.savefig(OUT / "curve.png")
plt.close(fig)

# ---- 3. bar chart of sizes --------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 2.6))
labels = ["reference"] + [name(r) for r in res]
vals = [base_bytes] + [r["bytes"] for r in res]
cols = ["crimson"] + ["steelblue"] * len(res)
ax.barh(labels, vals, color=cols)
ax.axvline(base_bytes * 0.9, color="green", ls=":", lw=1.2, label="target size")
ax.set_xlabel("compressed bytes")
ax.invert_yaxis()
ax.legend(fontsize=7)
fig.savefig(OUT / "bars.png")
plt.close(fig)
print("figures ->", OUT)
