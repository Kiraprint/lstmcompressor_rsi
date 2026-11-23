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
_stats = Path("timing_stats.json")
stats = json.loads(_stats.read_text()) if _stats.exists() else {}


def name(r):
    return r["name"].replace("_", " ")


# ---- 1. ratio vs time --------------------------------------------------------
SHORT = {
    "baseline_reference": "baseline",
    "A_LSTM_Adam_n128_s13": "A  Adam n128 s13",
    "B_LSTM_Adam_n48_s32": "B  Adam n48 s32",
    "C_LSTM_Muon_n128_s32": "C  Muon n128 s32",
    "D_LSTM_Muon_n48_s32": "D  Muon n48 s32",
    "E_LSTM_Muon_n80_s32_cosine_FINAL": "E  Muon n80 s32 cos",
    "F_GDN_Muon_n48_s32": "F  GDN-2 n48 s32",
    "G_LLM_GPT2_prior_fast": "G  GPT-2 prior",
}
STYLE = {"baseline": ("crimson", "*"), "adam": ("#3b6fb6", "o"),
         "muon": ("#1f9d8f", "o"), "gdn": ("#8e44ad", "s"),
         "gpt2": ("#c0392b", "*")}
FAM_LABEL = {"baseline": "базовый (эталон)", "adam": "LSTM + Adam",
             "muon": "LSTM + Muon", "gdn": "GDN-2 (лин. внимание)",
             "gpt2": "GPT-2 приор (замороженный)"}


def family(r):
    n = r["name"]
    if n == "baseline_reference":
        return "baseline"
    if "GDN" in n:
        return "gdn"
    if "LLM" in n or "GPT2" in n:
        return "gpt2"
    return "adam" if "Adam" in n else "muon"


def stat_of(r):
    key = "baseline" if r["name"] == "baseline_reference" else r["name"]
    return stats.get(key)


def pt(r):
    """(ratio, x, (xerr_lo, xerr_hi) or None, repeats or None)."""
    st = stat_of(r)
    if not st:
        return r["ratio"], r["total_s"], None, None
    m = st["median_total_s"]
    return r["ratio"], m, (m - st["min_total_s"], st["max_total_s"] - m), st["reps_total_s"]


ref = N / base_bytes
tgt = ref * 1.10
budget = stats.get("_budget", {}).get("budget_50pct_s") or 27.0 * 1.5

fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.4))


def rules(ax):
    ax.axhline(ref, color="crimson", ls="--", lw=1.1, label=f"бейзлайн {ref:.3f}")
    ax.axhline(tgt, color="green", ls=":", lw=1.1, label=f"цель +10 % ({tgt:.3f})")
    ax.axvline(budget, color="orange", ls=":", lw=1.1,
               label=f"бюджет +50 % ({budget:.1f} с)")


def place(ax, fig, x, y, text, taken, fontsize=8):
    """Put ``text`` near (x, y), trying offsets until it clears other labels."""
    cands = [(6, 3, "left"), (6, -10, "left"), (-6, 3, "right"), (-6, -10, "right"),
             (0, 9, "center"), (0, -15, "center"), (10, 8, "left"), (-10, 8, "right"),
             (10, -16, "left"), (-10, -16, "right")]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    for dx, dy, ha in cands:
        t = ax.annotate(text, (x, y), fontsize=fontsize, zorder=6, ha=ha,
                        xytext=(dx, dy), textcoords="offset points")
        bb = t.get_window_extent(r)
        inside = (bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1
                  and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1)
        if inside and not any(bb.overlaps(o) for o in taken):
            taken.append(bb)
            return t
        t.remove()
    t = ax.annotate(text, (x, y), fontsize=fontsize, zorder=6, ha="left",
                    xytext=(6, 3), textcoords="offset points")
    taken.append(t.get_window_extent(r))
    return t


takenL, takenR, seen = [], [], set()
for r in res:
    fam = family(r)
    col, mk = STYLE[fam]
    y, x, err, reps = pt(r)
    axL.errorbar(x, y, xerr=None if err is None else [[err[0]], [err[1]]],
                 fmt=mk, ms=12 if mk == "*" else 7, color=col, zorder=4,
                 ecolor=col, elinewidth=1.2, capsize=3,
                 label=None if fam in seen else FAM_LABEL[fam])
    seen.add(fam)
    if reps:  # individual repeats of the measured runs
        axL.plot(reps, [y] * len(reps), "|", ms=8, color="0.35", alpha=0.8, zorder=3)
axL.set_xscale("log")
axL.set_xlim(6, 800)
axL.set_ylim(2.85, 7.3)
rules(axL)
axL.set_xlabel("суммарное время, кодирование + декодирование (с, лог.)")
axL.set_ylabel("коэффициент сжатия")
axL.set_title("все прогоны")
axL.legend(fontsize=7, loc="upper left", framealpha=0.92)

# right panel: runs that fit the time budget, y zoomed on the real spread
for r in res:
    y, x, err, reps = pt(r)
    if x > 60 or not (2.95 <= y <= 3.2):
        continue
    col, mk = STYLE[family(r)]
    axR.errorbar(x, y, xerr=None if err is None else [[err[0]], [err[1]]],
                 fmt=mk, ms=12 if mk == "*" else 7, color=col, zorder=4,
                 ecolor=col, elinewidth=1.2, capsize=3)
    if reps:
        axR.plot(reps, [y] * len(reps), "|", ms=8, color="0.35", alpha=0.8, zorder=3)
axR.set_xlim(0, 58)
axR.set_ylim(2.978, 3.142)
rules(axR)
axR.set_xlabel("суммарное время (с)")
axR.set_title("зум: прогоны в бюджете по времени")
axR.legend(fontsize=7, loc="lower right", framealpha=0.92)

# labels last, so the layout (limits, legend) is already final
for r in res:  # left: only the outliers, the cluster is labelled in the zoom
    y, x, _, _ = pt(r)
    if x > 60 or y > 3.5:
        place(axL, fig, x, y, SHORT[r["name"]], takenL)
place(axL, fig, 22.1, 3.02, "A–E: LSTM,\nсм. зум", takenL)
_zoom = sorted((pt(r) + (r,) for r in res if pt(r)[1] <= 60 and 2.95 <= pt(r)[0] <= 3.2),
               key=lambda t: (min(t[0] - 2.985, 3.135 - t[0]), t[1]))
for y, x, _e, _s, r in _zoom:
    place(axR, fig, x, y, SHORT[r["name"]], takenR)

axR.set_xlim(0, 58)
axR.set_ylim(2.978, 3.142)
rules(axR)
axR.set_xlabel("суммарное время (с)")
axR.set_title("зум: прогоны в бюджете по времени")
axR.legend(fontsize=7, loc="lower right", framealpha=0.92)

fig.suptitle("Коэффициент сжатия от времени; точки — медиана, штрихи — отдельные прогоны",
             y=1.03, fontsize=10.5)
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
ax.set_xlabel("шаг кодирования (позиция байта в потоке)")
ax.set_ylabel("накопленные биты / байт")
ax.set_title("Онлайновая сходимость, без предобучения")
ax.legend(fontsize=9)
fig.savefig(OUT / "curve.png")
plt.close(fig)

# ---- 3. bar chart of sizes --------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.1))
labels = ["эталон"] + [SHORT.get(r["name"], name(r)) for r in res]
vals = [base_bytes] + [r["bytes"] for r in res]
cols = ["crimson"] + ["steelblue"] * len(res)
ax.barh(labels, vals, color=cols)
ax.axvline(base_bytes * 0.9, color="green", ls=":", lw=1.5, label="цель −10 %")
ax.set_xlabel("сжатый размер, байт")
ax.invert_yaxis()
ax.legend(fontsize=9)
fig.savefig(OUT / "bars.png")
plt.close(fig)
print("figures ->", OUT)