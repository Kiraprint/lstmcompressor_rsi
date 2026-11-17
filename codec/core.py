"""Online neural + statistical byte codec.

Frame layout (self-describing, no assumption about file type)::

    magic   b"NSCC1"                      5 bytes
    u8      header length
    header  JSON {"len": N, "alphabet": [..], "cfg": {...}}
    AC      arithmetic-coded payload

The file is split into ``n_streams`` contiguous chunks (one batch slot each, exactly like
the reference implementation).  Every slot codes exactly ``ceil(N/n)`` symbols; positions
past EOF are coded as a deterministic dummy (index 0) so both passes issue *identical*
arithmetic-coder operations.

Each slot keeps its own recurrent state and byte history.  All slots are predicted in one
batched forward pass; the neural distribution is geometrically mixed with fast statistical
experts (order-1..k models and an LZP match model), arithmetic coded, and then one SGD step
is taken.  The decoder replays the same computation: no model weights are transmitted.
"""

from __future__ import annotations

import io
import json
import os
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ArithmeticCoder import (ArithmeticDecoder, ArithmeticEncoder, BitInputStream,
                             BitOutputStream)
from codec.config import Config
from codec.mixer import MatchModel, Mixer, OrderModel
from codec import nn as nnm
from codec.optim import build_optimizer

MAGIC = b"NSCC1"


def encode_frame(payload: bytes, length: int, alphabet: Sequence[int],
                 cfg: Config) -> bytes:
    import struct
    mask = bytearray(32)
    for b in alphabet:
        mask[b >> 3] |= 1 << (b & 7)
    fields = (cfg.n_streams, cfg.seq_len, cfg.hidden, cfg.emb, cfg.layers,
              cfg.heads, cfg.slots, cfg.seed,
              int(cfg.mixer), int(cfg.freq_scale),
              int(cfg.mix_lr * 1e6), int(cfg.mix_init * 1000))
    head = length.to_bytes(5, "big") + bytes(mask) + struct.pack(">12I", *fields)            + cfg.arch.encode().ljust(16, b"\x00")
    return MAGIC + len(head).to_bytes(2, "big") + head + payload


def decode_frame(blob: bytes) -> Tuple[Dict, bytes]:
    import struct
    if blob[:5] != MAGIC:
        raise ValueError("bad magic")
    n = int.from_bytes(blob[5:7], "big")
    h = blob[7:7 + n]
    length = int.from_bytes(h[:5], "big")
    mask = h[5:37]
    alphabet = [b for b in range(256) if mask[b >> 3] & (1 << (b & 7))]
    f = struct.unpack(">12I", h[37:85])
    arch = h[85:101].rstrip(b"\x00").decode()
    head = {"len": length, "alphabet": alphabet, "arch": arch,
            "n_streams": f[0], "seq_len": f[1], "hidden": f[2],
            "emb": f[3], "layers": f[4], "heads": f[5], "slots": f[6],
            "seed": f[7], "mixer": bool(f[8]), "freq_scale": f[9],
            "mix_lr": f[10] / 1e6, "mix_init": f[11] / 1000}
    return head, blob[7 + n:]


def _cumfreq(probs: np.ndarray, scale: int) -> np.ndarray:
    q = (probs * scale).astype(np.int64) + 1   # reference: +1 after truncation
    return np.cumsum(q, dtype=np.int64)


class Experts:
    """Order-k models (tables shared across streams) + per-stream LZP."""

    def __init__(self, cfg: Config, vocab: int, n_streams: int):
        self.orders = [OrderModel(k, vocab, cfg.count_prior, cfg.mix_decay)
                       for k in cfg.orders]
        self.match = [MatchModel(vocab) for _ in range(n_streams)]
        self.E = 1 + len(self.orders) + 1
        init = np.zeros(self.E)
        init[0] = cfg.mix_init    # the neural net starts dominant
        self.mixer = Mixer(self.E, n_ctx=64, lr=cfg.mix_lr, init=init)
        self.V = vocab
        self.L = np.zeros((n_streams, self.E, vocab))

    def logp(self, hists: Sequence[bytes], poss: Sequence[int]) -> np.ndarray:
        L = self.L
        for i, h in enumerate(hists):
            L[i, 0].fill(-math.log(self.V))     # placeholder, NN fills row 0
            for j, m in enumerate(self.orders):
                m.logp(h, L[i, 1 + j])
            self.match[i].logp(h, L[i, -1], poss[i])
        return L

    def update(self, hists: Sequence[bytes], syms: Sequence[int],
               poss: Sequence[int]) -> None:
        for i, h in enumerate(hists):
            for m in self.orders:
                m.update(h, syms[i])
            self.match[i].update(h, syms[i], poss[i])


def _lr(cfg: Config, step: int, total: int) -> float:
    total = max(1, total)
    t = min(max(0, step), total) / total
    if cfg.sched == "cosine":
        return cfg.lr_end + (cfg.lr - cfg.lr_end) * 0.5 * (1 + math.cos(math.pi * t))
    if cfg.sched == "const":
        return cfg.lr
    return cfg.lr_end + (cfg.lr - cfg.lr_end) * (1.0 - t)


def run(data, cfg: Config, compress: bool, alphabet: Optional[Sequence[int]] = None,
        length: int = -1, verbose: bool = False) -> Dict:
    """Compress ``data`` (bytes) or decompress ``data`` (the payload)."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    raw = bytes(data)
    if compress:
        length = len(raw)
    elif length < 0:
        raise ValueError("length is required for decompression")
    if alphabet is None:
        alphabet = sorted(set(raw))
    V = len(alphabet)
    lut = np.zeros(256, dtype=np.int64)
    for i, b in enumerate(alphabet):
        lut[b] = i

    n = min(cfg.n_streams, max(1, length))
    slice_len = math.ceil(length / n)
    steps = slice_len   # reference: while pos < ceil(N/n)
    T = cfg.seq_len

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = nnm.build(cfg.arch, V, cfg).to(device)
    # Muon orthogonalises the recurrent matrices; Adam handles embedding/bias/temperature.
    muon_params = ([p for name, p in model.named_parameters()
                    if name in ("rnn.weight_ih_l0", "rnn.weight_hh_l0")
                    or name == "rnn.weight_ih_l0"] if cfg.optimizer == "muon" else [])
    opt = build_optimizer(cfg, model, muon_params)

    if compress:
        out = io.BytesIO()
        coder = ArithmeticEncoder(32, BitOutputStream(out))
    else:
        out = None
        coder = ArithmeticDecoder(32, BitInputStream(io.BytesIO(raw)))
        raw = bytearray(length)

    hists: List[bytearray] = [bytearray() for _ in range(n)]
    cursors = [i * slice_len for i in range(n)]
    experts = Experts(cfg, V, n)
    state = model.init_state(n, device)
    scale = cfg.freq_scale
    uf = _cumfreq(np.full(V, 1.0 / V), scale)

    first = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k = 0
        if cursors[i] < length:
            if compress:
                k = lut[raw[cursors[i]]]
                coder.write(uf, k)
            else:
                k = int(coder.read(uf))
                raw[cursors[i]] = alphabet[k]
            cursors[i] += 1
        first[i] = k
        hists[i].append(k)

    seq = torch.tensor(first, dtype=torch.long, device=device).view(n, 1).repeat(1, T)

    t0 = time.time()
    total_bits = 0.0
    n_coded = 0
    _DBG = [0.0, 0.0]
    curve: List[Tuple[int, float]] = []

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = _lr(cfg, step, steps)
        logits, new_state = model(seq, state)
        probs_nn = torch.softmax(logits, dim=1).detach().cpu().numpy()
        logp_nn = np.log(np.maximum(probs_nn, 1e-300))

        if cfg.mixer:
            L = experts.logp([bytes(h) for h in hists], [len(h) for h in hists])
            L[:, 0] = logp_nn
            ctx = np.full(n, min(step, 63), dtype=np.int64)
            a = experts.mixer.weights(ctx)
            logp = np.einsum("be,bev->bv", a, L)
            logp -= logp.max(axis=1, keepdims=True)
            p = np.exp(logp)
            p /= p.sum(axis=1, keepdims=True)
        else:
            p = np.exp(logp_nn)

        old_h = [bytes(h) for h in hists]
        old_l = [len(h) for h in hists]
        syms = np.zeros(n, dtype=np.int64)
        masks = np.zeros(n, dtype=np.float64)
        for i in range(n):
            pi = p[i]
            fr = _cumfreq(pi, scale)
            if cursors[i] < length:
                if compress:
                    k = lut[raw[cursors[i]]]
                    coder.write(fr, k)
                else:
                    k = int(coder.read(fr))
                    raw[cursors[i]] = alphabet[k]
                masks[i] = 1.0
                total_bits += -math.log2(max(pi[k], 1e-300))
                n_coded += 1
                cursors[i] += 1
            else:
                # reference: out-of-range positions are not coded at all
                k = 0
            syms[i] = k
            hists[i].append(k)

        if cfg.mixer:
            experts.update(old_h, syms, old_l)
            experts.mixer.update(ctx, L, a, p, syms)

        if os.environ.get("CODEC_DEBUG") and masks.sum():
            _DBG[0] += -math.log2(max(float(np.prod(p[np.arange(n), syms][masks > 0])), 1e-300))
            _DBG[1] += masks.sum()
            if step % 32 == 0:
                print(f"  [dbg] {step} coder={total_bits/max(1,n_coded):.4f} "
                      f"model={_DBG[0]/max(1,_DBG[1]):.4f}", flush=True)
        if os.environ.get("CODEC_DEBUG") and step in (7,127,255):
            print("  [dbg2]",step,"bpb",round(total_bits/max(1,n_coded),5),"zero",opt.param_groups[0]["lr"],flush=True)
        if n_coded and step % 128 == 0:
            curve.append((step, total_bits / n_coded))
            if os.environ.get("CODEC_DEBUG"):
                print(f"  [dbg] step={step} coder_bpb={total_bits/n_coded:.4f}", flush=True)

        sym_t = torch.tensor(syms, dtype=torch.long, device=device)
        sel = torch.softmax(logits, dim=1).gather(1, sym_t.view(-1, 1)).view(-1).clamp_min(1e-7)
        m = torch.tensor(masks, dtype=torch.float32, device=device)
        if m.sum() > 0:
            # sum, not mean: this is the reference implementation's gradient scale
            loss = -(torch.log(sel) * m).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()

        seq = torch.cat([seq[:, 1:], sym_t.view(n, 1)], dim=1).detach()
        state = model.detach(new_state)

    if compress:
        coder.finish()
        while coder.output.numbitsfilled != 0:   # pad the trailing partial byte
            coder.output.write(0)
    payload = out.getvalue() if compress else None
    return {"elapsed": time.time() - t0, "bits": total_bits, "n": n_coded,
            "bpb": total_bits / max(1, n_coded), "curve": curve,
            "payload": payload, "alphabet": alphabet, "data": bytes(raw)}
