"""Fast statistical experts + geometric (log-linear) mixture.

The neural expert is a *slow* learner: on 100 kB it needs thousands of bytes to warm up,
during which an order-3 context model is far better.  Both regimes must be exploited.

Mixing rule (a proper geometric mixture, i.e. weights on the probability simplex)::

    log p(s) = sum_j a_j log p_j(s),      a = softmax(u),  a_j >= 0, sum_j a_j = 1

with expert 0 = the neural net, experts 1..k = order-k byte models, and the last expert =
an LZP match model.  Because the weights are constrained to a simplex the result is a
genuine interpolation: it can never be sharper than its sharpest member, which is what
broke the naive ``log p_nn + w * correction`` variant (that one double counts, since the
order models are informative in absolute terms, not as deviations from the net).

Weights are adapted by online gradient descent on the coding loss::

    d(-log p(sym))/du_j = a_j * (log p_j(sym) - E_p[log p_j])

Per-context weights (coding-step bucket) let the mixer prefer statistics early in the
stream and the neural net once it has learned.  Both passes see the same history, so the
decoder derives exactly the same weights: no side information is transmitted.

Everything is float64 on CPU so the two passes agree bit-exactly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


class OrderModel:
    """Online order-k byte model producing a full log-probability row."""

    __slots__ = ("k", "counts", "totals", "prior", "decay", "vocab", "logv")

    def __init__(self, k: int, vocab: int, prior: float = 0.35, decay: float = 1.0):
        self.k = k
        self.vocab = vocab
        self.prior = prior
        self.decay = decay
        self.counts: Dict[bytes, Dict[int, float]] = {}
        self.totals: Dict[bytes, float] = {}
        self.logv = math.log(vocab)

    def logp(self, hist: bytes, out: np.ndarray) -> None:
        ctx = hist[-self.k:] if self.k else b""
        c = self.counts.get(ctx)
        if not c:
            out.fill(-self.logv)
            return
        tot = self.totals[ctx]
        base = self.prior / (tot + self.prior * self.vocab)
        out.fill(math.log(base))
        for s, v in c.items():
            out[s] = math.log((v + self.prior) / (tot + self.prior * self.vocab))

    def update(self, hist: bytes, sym: int) -> None:
        ctx = hist[-self.k:] if self.k else b""
        c = self.counts.get(ctx)
        if c is None:
            c = {}
            self.counts[ctx] = c
            self.totals[ctx] = 0.0
        if self.decay != 1.0:
            for s in c:
                c[s] *= self.decay
            self.totals[ctx] *= self.decay
        c[sym] = c.get(sym, 0.0) + 1.0
        self.totals[ctx] += 1.0


class MatchModel:
    """LZP: predicts the continuation of the longest previously seen suffix.

    The confidence grows with the *verified* match length, so a 5-byte coincidence is not
    trusted like a 30-byte one.  This is what makes repeated markup, table rows and
    phrasing cheap -- the neural net alone spends a long time rediscovering them.
    """

    __slots__ = ("vocab", "min_len", "max_chain", "table", "n_pred", "n_hit", "logv")

    def __init__(self, vocab: int, min_len: int = 5, max_chain: int = 8):
        self.vocab = vocab
        self.min_len = min_len
        self.max_chain = max_chain
        self.table: Dict[bytes, int] = {}      # key -> most recent position
        self.n_pred = 0
        self.n_hit = 0
        self.logv = math.log(vocab)

    def logp(self, hist: bytes, out: np.ndarray, pos: int) -> float:
        if pos < self.min_len + 1:
            out.fill(-self.logv)
            return 0.0
        src = self.table.get(hist[-self.min_len:])
        if src is None or src >= pos:
            out.fill(-self.logv)
            return 0.0
        nxt = hist[src]
        ml = self.min_len
        lim = min(pos - src, 40)
        while ml < lim and hist[pos - ml - 1] == hist[src - ml - 1]:
            ml += 1
        # confidence: 0.55 at min_len, saturating near 0.97
        conf = 0.55 + 0.42 * (1.0 - math.exp(-(ml - self.min_len) / 8.0))
        lo = math.log((1.0 - conf) / max(1, self.vocab - 1))
        out.fill(lo)
        out[nxt] = math.log(conf)
        return conf

    def update(self, hist: bytes, sym: int, pos: int) -> None:
        if pos < self.min_len:
            return
        self.table[hist[-self.min_len:]] = pos


class Mixer:
    """Per-context simplex weights over experts (geometric mixture)."""

    __slots__ = ("u", "lr", "n_ctx", "E")

    def __init__(self, n_experts: int, n_ctx: int, lr: float = 0.02,
                 init: Optional[np.ndarray] = None):
        self.E = n_experts
        self.u = np.zeros((n_ctx, n_experts))
        if init is not None:
            self.u[:] = init
        self.lr = lr
        self.n_ctx = n_ctx

    def weights(self, ctx: np.ndarray) -> np.ndarray:
        u = self.u[ctx]
        u = u - u.max(axis=-1, keepdims=True)
        e = np.exp(u)
        return e / e.sum(axis=-1, keepdims=True)

    def update(self, ctx: np.ndarray, L: np.ndarray, a: np.ndarray,
               p: np.ndarray, syms: np.ndarray) -> None:
        """``L``: [B,E,V] log-probs, ``a``: [B,E], ``p``: [B,V], ``syms``: [B]."""
        b = np.arange(len(syms))
        lsym = L[b, :, syms]                        # [B,E]
        ep = np.einsum("bev,bv->be", L, p)          # [B,E]
        g = a * (lsym - ep)                         # [B,E]  d(-log p)/du
        np.add.at(self.u, ctx, self.lr * g)
        np.clip(self.u, -12.0, 12.0, out=self.u)
