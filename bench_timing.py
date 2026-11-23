#!/usr/bin/env python3
"""Repeated, paired timing measurements for the three reported codecs.

Bytes are deterministic (identical MD5 on every run), so only wall time varies.
The host shares the GPU with unrelated processes, and a single baseline reading
on this machine ranges from 16 s to 27 s, so absolute times are not comparable
across sessions. Runs are therefore interleaved (baseline, LSTM, GPT-2, repeat)
and the reported statistic is the *paired* time ratio of each codec to the
baseline measured in the same repetition.

    .venv/bin/python bench_timing.py [reps]   # -> timing_stats.json
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = str(ROOT / ".venv" / "bin" / "python")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SRC = str(ROOT / "data" / "enwik5")

LLM_CMD = """
import sys, json, hashlib
from llm_codec_fast import LLMCodecFast
data = open(sys.argv[1], 'rb').read()
c = LLMCodecFast()
r = c.compress(data)
d = c.decompress(r['frame'])
print(json.dumps({'encode_s': r['encode_s'], 'decode_s': d['decode_s'],
                  'bytes': len(r['frame']), 'ok': d['data'] == data,
                  'md5': hashlib.md5(d['data']).hexdigest()}))
"""


def run(cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"failed: {cmd}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout


def baseline():
    run([PY, "main.py"], ROOT / "bench_baseline")
    d = json.loads((ROOT / "bench_baseline" / "results.json").read_text())
    return {"total_s": d["time_spent"], "bytes": None, "ok": d["success"]}


def lstm():
    out = run([PY, "-m", "codec.cli", "--config", str(ROOT / "config.json"),
               "--input", SRC])
    d = json.loads(out[out.index("{"):])
    return {"enc_s": d["encode_s"], "dec_s": d["decode_s"],
            "total_s": d["total_s"], "bytes": d["compressed"], "ok": d["success"]}


def llm():
    out = run([PY, "-c", LLM_CMD, SRC])
    d = json.loads(out[out.index("{"):])
    return {"enc_s": round(d["encode_s"], 2), "dec_s": round(d["decode_s"], 2),
            "total_s": round(d["encode_s"] + d["decode_s"], 2),
            "bytes": d["bytes"], "ok": d["ok"]}


CASES = [("baseline", baseline),
         ("E_LSTM_Muon_n80_s32_cosine_FINAL", lstm),
         ("G_LLM_GPT2_prior_fast", llm)]

samples = {name: [] for name, _ in CASES}
for rep in range(1, REPS + 1):
    for name, fn in CASES:          # interleaved: same host state for all three
        r = fn()
        r["rep"] = rep
        samples[name].append(r)
        print(f"rep {rep}/{REPS}  {name:<36} {r['total_s']:6.2f} s  "
              f"bytes={r['bytes']} ok={r['ok']}", flush=True)

base_by_rep = {r["rep"]: r["total_s"] for r in samples["baseline"]}
stats = {}
for name, _ in CASES:
    reps = samples[name]
    tot = [r["total_s"] for r in reps]
    paired = [r["total_s"] / base_by_rep[r["rep"]] for r in reps]
    ent = {
        "reps_total_s": tot,
        "median_total_s": round(statistics.median(tot), 2),
        "min_total_s": round(min(tot), 2),
        "max_total_s": round(max(tot), 2),
        "spread_pct": round((max(tot) / min(tot) - 1) * 100, 1),
        "paired_time_ratio": [round(p, 3) for p in paired],
        "median_paired_ratio": round(statistics.median(paired), 3),
        "bytes": reps[-1]["bytes"],
        "ok": all(r["ok"] for r in reps),
    }
    if "enc_s" in reps[0]:
        ent["median_enc_s"] = round(statistics.median([r["enc_s"] for r in reps]), 2)
        ent["median_dec_s"] = round(statistics.median([r["dec_s"] for r in reps]), 2)
    stats[name] = ent

bm = stats["baseline"]["median_total_s"]
stats["_meta"] = {
    "reps": REPS,
    "interleaved": True,
    "baseline_median_s": bm,
    "budget_50pct_s": round(bm * 1.5, 2),
    "note": ("Прогоны чередуются (baseline, LSTM, GPT-2), поэтому дрейф хоста "
             "влияет на все три одинаково; paired_time_ratio — отношение времени "
             "кодеков к бейзлайну в том же повторе."),
}
(ROOT / "timing_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
print("\n" + json.dumps(stats, indent=2))
print("-> timing_stats.json")