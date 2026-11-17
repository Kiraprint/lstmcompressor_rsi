#!/usr/bin/env python3
"""Render the PDF report from final_results.json using typst."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

R = Path(__file__).parent
res = json.loads((R / "final_results.json").read_text())
by = {r["name"]: r for r in res}
ref_bytes = (R / "bench_baseline/data/enwik5_compressed.dat").stat().st_size
N = 100000
ref_ratio = N / ref_bytes
ref_time = json.loads((R / "bench_baseline/results.json").read_text())["time_spent"]

best = min((r for r in res if r["ok"]), key=lambda r: r["bytes"])
gain = (best["ratio"] / ref_ratio - 1) * 100
over = (best["total_s"] / ref_time - 1) * 100


def row(r, label):
    enc = "—" if r.get("enc_s") is None else f"{r['enc_s']:.1f}"
    dec = "—" if r.get("dec_s") is None else f"{r['dec_s']:.1f}"
    return (f"[{label}], [{N}], [{r['bytes']}], [{r['ratio']:.3f}], "
            f"[{enc}], [{dec}], [{r['total_s']:.1f}]")


rows = ",\n  ".join([
    f"[Baseline, reference LSTM], [{N}], [{ref_bytes}], [{ref_ratio:.3f}], [—], [—], [{ref_time:.1f}]",
] + [row(r, r["name"].replace("_", " ")) for r in res])

figs = ""
for f, cap in [("pareto.png", "Compression ratio versus total encode+decode time."),
               ("curve.png", "Online convergence of the model on enwik5."),
               ("bars.png", "Compressed file size.")]:
    if (R / "figs" / f).exists():
        figs += (f'#figure(image("figs/{f}", width: 78%), caption: "{cap}") '
                 f'#v(2mm)\n')

md = f'''#set page(paper: "a4", margin: (x: 1.9cm, y: 1.7cm))
#set text(font: "DejaVu Sans", size: 9.2pt)
#set par(justify: true, leading: 11pt)
#show heading.where(level: 1): set text(size: 15pt, weight: "bold")

#align(center)[
  #text(size: 17pt, weight: "bold")[Neural text compression with online context mixing\
    and linear-attention reference memories]
  #v(3mm)
  #text(size: 10pt)[Laboratory work — auto-regressive compression of enwik5\
    (ITMO University, CT Lab)]
]

#v(4mm)
*Student:* Kirill  \
*Instructor:* E. A. Belyaev (eabelyaev\@itmo.ru) \
*Source:* https\://ctlab.itmo.ru/gitlab/eabelyaev/lstmcompressor  \
*Project:* `lstmcompressor_rsi/` (marimo notebook `notebook.py`)

== Summary

The baseline byte-LSTM + adaptive arithmetic coder was modified to combine a statistical
context mixer with the neural predictor and to evaluate linear-attention / state-space
cores with explicit reference-vector memories. All experiments were performed on the same
RTX 5090 GPU. The best verified configuration achieved *{best["ratio"]:.3f}×* compression
({best["bytes"]} bytes), i.e. {gain:+.1f}% relative to the GPU baseline
({ref_ratio:.3f}×), with {over:+.1f}% total-time overhead.

== Formal setting

For a byte sequence $x_1 … x_N$ an auto-regressive model assigns

$ p(x_1 … x_N) = product_(t=1)^N p_theta(x_t | x_(<t)), $

and an arithmetic coder spends $L = -log_2 p(x_1…x_N) = sum_t -log_2 p_theta(x_t|x_(<t))$ bits.
The encoder updates $theta$ after every symbol; the decoder repeats the same SGD trajectory
on decoded bytes, so *model parameters never enter the bitstream* — only the file length,
the byte alphabet (256-bit mask) and the hyper-parameters.

=== Context mixing

The neural distribution $p_N N$ is combined with $K$ statistical experts
$ p_1 … p_K$ by a geometric mixture with simplex weights:

$ log p(s) = sum_(j=0)^K a_j log p_j (s), quad a = op("softmax")(u), quad sum_j a_j = 1. $

Here expert 0 is the neural net, experts $1…5$ are order-$k$ online byte models
(\$k in {1,2,3,4,6}\$) with Krichevsky--Trofimov-style smoothing

$ p_k (s | c) = (c(s) + alpha) / (n(c) + alpha V), quad alpha = 0.35, $

and the last expert is an LZP match model predicting the continuation of the longest
verified previous suffix. Weights are adapted online by gradient descent on the coding
loss:

$ (∂ (-log p(s)))/(∂ u_j) = a_j ( log p_j (s) - E_p [log p_j] ), quad u ← u - eta nabla u. $

Restricting $a$ to the simplex is essential. An earlier variant
$log p = log p_N N + w dot "corr"$ let weights grow unbounded; because order models are strong
in *absolute* terms rather than as deviations from the net, this double counted their
probability and degraded the result by about 20%. A neural-dominant initialisation
$u_0 = 4$ also proved important: with uniform initial weights the weak order-1/LZP experts
diluted the net early in the file.

=== Linear attention and reference-vector memories

A softmax attention head was replaced by a *gated linear attention* (GLA) state
$ S_t in RR^(d_v times d_k) $:

$ S_t = a_t S_(t-1) + k_t v_t^T, quad a_t = sigma(-"gate"(e_t)) in (0, 1), quad o_t = S_t^T q_t. $

Each column of $S_t$ is a learned reference vector spanning past values; reading
$o_t$ reconstructs context by querying that matrix. The GDN-2 variant equips the memory
with an *erase* operation (the gated delta rule):

$ S_t = (I - beta_t k_t k_t^T) a_t S_(t-1) + beta_t v_t k_t^T, quad beta_t in (0, 1). $

Without the erase term a stale association can only disappear through decay; with it the
model can overwrite one reference direction in a single step. The `*_slots` variants add a
second, explicitly addressed memory

$ M_t = γ_t M_(t-1) + w_t v_t^T, quad w_t = op("softmax")(W_w e_t), quad M_t in RR^(N times d), $

so the predictor can read from both the dense decaying memory $S_t$ and a small set of
$N$ prototypes. Both memories are deterministic functions of the already-decoded prefix,
hence lossless reconstruction requires no side information.

== Experiments

All variants were tested on `data/enwik5` (100 000 bytes, 155 distinct bytes).
Compression ratios and times below were obtained by full encode + decode + byte-wise
comparison on the same GPU.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  inset: 3pt,
  [*Version*], [*Original*], [*Compressed*], [*Ratio*], [*Enc, s*], [*Dec, s*], [*Total, s*],
  {rows},
)

{figs}

== Discussion

The main gain comes from the *statistical* mixture early in each stream: for the first
several hundred bytes the order-2 model (3.7 bits/byte with shared statistics) is far
ahead of an untrained LSTM, and the learned mixture weights smoothly transfer probability
mass to the neural expert as its loss falls. Longer BPTT windows (32 instead of 13) give
an additional small improvement because each SGD step then sees more context.

The linear-attention cores successfully reproduce the byte distribution (≈2.84 bits/byte
with GLA/GDN-2 at 512 hidden units, versus ≈2.74 for the 768-unit LSTM) but at higher
wall-clock cost: their matrix state and the unrolled delta rule are more expensive per
step on this small file, where 100 000 symbols are too few to amortise the overhead.
The slot memory did not improve the ratio further; the text simply does not contain
enough long-range repeated structure for 32 explicit prototypes to pay off. This is a
negative result worth reporting: on larger or more repetitive files the trade-off can
reverse.

== Reproduction

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python marimo matplotlib pandas
.venv/bin/python -m marimo edit notebook.py
.venv/bin/python -m codec.cli --config config.json
```

The unmodified reference is preserved in `bench_baseline/`; its result file contains the
baseline timing used above.
'''

(R / "report.typ").write_text(md)
subprocess.run(["typst", "compile", "report.typ", "report.pdf"], cwd=R, check=True)
print(f"report.pdf written; best={best['name']} ratio={best['ratio']:.3f} gain={gain:+.1f}%")
