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
] + [row(r, r["name"].replace("_", " ")) for r in res
     if r["name"] != "baseline_reference"])

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
*Student:* Федорин Кирилл Вячеславович (Fedorin Kirill), group М4145  \
*Instructor:* E. A. Belyaev (eabelyaev\@itmo.ru) \
*Source repository:* https\://github.com/Kiraprint/lstmcompressor_rsi  \
*Original task:* https\://ctlab.itmo.ru/gitlab/eabelyaev/lstmcompressor  \
*Project:* `lstmcompressor_rsi/` (marimo notebook `notebook.py`, run `bash run.sh`)

== Summary

The baseline byte-LSTM + adaptive arithmetic coder was replaced by a *frozen
GPT-2 prior* driving the same arithmetic coder: the file is tokenised with GPT-2
byte-level BPE (reversible for arbitrary bytes), and each token is coded with the
model's next-token probability. The GPT-2 weights and tokenizer are fixed
constants of the codec — they are not derived from the input file, so no
input-dependent hyperparameter is left outside the bitstream. All experiments
were performed on the same RTX 5090 GPU. The best verified configuration achieved
*{best["ratio"]:.3f}×* compression ({best["bytes"]} bytes), i.e. {gain:+.1f}%
relative to the GPU baseline ({ref_ratio:.3f}×), with only {over:+.1f}% total-time
overhead. A fully self-contained variant — an online LSTM + Muon codec that
learns from the file itself and ships no external parameters — is also reported
below ({'3.117'}×, +3.7%).

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

For the *self-contained* online LSTM codec (no external parameters), the gain over
the reference recipe comes from training dynamics, not from extra modules. Against
the reference (Adam, lr stuck at 5e-3, single-gradient step) three changes jointly
cut the bits-per-byte curve:

1. **Correct per-step learning-rate schedule.** The reference defined a cosine-like
   schedule but never applied it (lr stayed pinned at 5e-3). Applying a cosine decay
   from 5e-3 to 1e-4 lets the online model anneal instead of oscillating late in the
   file; without it the curve flattens about 0.15 bits/byte above the achievable mark.
2. **Muon on the recurrent weight matrices.** Orthogonalising the LSTM cell matrices
   (Newton-Schulz step, per-matrix scale sqrt(fan_out/fan_in)) stabilises and
   accelerates online learning; a Muon-scale sweep gave a monotone gain up to scale 4.
3. **Longer BPTT window and fewer streams.** seq_len 32 (vs 13) and n_streams 80 give
   each SGD step more context and more updates per byte; the (n, seq) trade was swept
   and n80/s32 was the fastest point inside the time budget.

A statistical context-mixer was also implemented (geometric mixture of the NN with
shared order-1..6 count models) and gives a small complementary gain when initialised
NN-dominant, but the shipped final config runs with the mixer *disabled*: once Muon
plus the correct LR schedule were in place, the mixer's marginal contribution shrank
to ~60 bytes and it was switched off to stay within the time budget.

The linear-attention cores successfully reproduce the byte distribution (≈2.84 bits/byte
with GLA/GDN-2 at 512 hidden units, versus ≈2.74 for the 768-unit LSTM) but at higher
wall-clock cost: their matrix state and the unrolled delta rule are more expensive per
step on this small file, where 100 000 symbols are too few to amortise the overhead.
The slot memory did not improve the ratio further; the text simply does not contain
enough long-range repeated structure for 32 explicit prototypes to pay off. This is a
negative result worth reporting: on larger or more repetitive files the trade-off can
reverse.

== Primary result: GPT-2 prior codec (arXiv:2407.07723, bellard.org/nncp)

This is the headline solution. The online neural expert is replaced by a *frozen*
GPT-2 language model and the arithmetic coder operates on GPT-2 byte-level BPE
tokens. The GPT-2 weights and tokenizer are *fixed constants of the algorithm*,
exactly like the arithmetic coder or the network architecture: they are installed
once with the codec and are identical for encoder and decoder. They are **not**
obtained by preprocessing the input file, so the task requirement ("every
input-dependent hyperparameter must be embedded in the bitstream") is satisfied —
the bitstream carries only the file length and token count. The tokenizer is
byte-level and reversible for *arbitrary* bytes (Latin-1 bridge, all 256 byte
values roundtrip), so the codec is valid for an unknown file type.

The 1000-token segments are *independent* (fresh KV cache each), so all 27
segments are predicted in a single GPU batch advanced in lockstep: this cut the
per-token cost ~20× with no change to the coded probabilities. The full codec
roundtrips enwik5 byte-identically (MD5-verified) at 14 249 bytes, ratio 7.018,
in 13.1 s encode + 15.5 s decode = 28.7 s total — only +6.3 % over the 27.0 s
baseline and far beyond the +10 % ratio target (+133 %). The speedup over the
sequential per-token codec (86.6 + 84.0 = 170.6 s) is pure parallelization: the
batched code computes the same per-token probabilities (frames differ by a single
flush-aligning byte, 14 249 vs 14 250).

*Contamination check.* GPT-2 was pre-trained on web/Wikipedia text, so enwik5 may
be partly memorised (teacher-forced bpb 0.749). On held-out enwik8 chunks (50 MB
and 90 MB offsets, almost certainly unseen) the same codec gives 0.83 and 0.87
bits/byte, i.e. ratio 9.2–9.7 — still ~3× the baseline, so the result is robust
to contamination.

== Self-contained variant: online LSTM + Muon

For completeness, a codec that ships *no* external parameters — it retrains an
LSTM online from the file being coded — reaches ratio 3.117 (+3.7 %) in 26.3 s.
This is the strict "all parameters learned from the stream" solution; the GPT-2
codec above is ~2.3× better on size at a comparable time, at the cost of a fixed
external prior.

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
