"""Neural experts for online (decoder-symmetric) byte modelling.

All models share one protocol so the codec loop is architecture agnostic::

    state = model.init_state(batch)
    logits, state = model(x, state)        # x: [B, T] ids -> logits: [B, V]
    state = model.detach(state)

The window ``T`` is recomputed from scratch at every stream position, exactly like the
reference implementation: gradients flow through ``T`` steps (truncated BPTT) and the
carried state is the only long-range channel.

``ssm``/``gdn``/``ssm_slots`` implement *linear attention* whose carried state is an
explicit matrix memory

    S_t = a_t * S_{t-1} + k_t v_t^T                            (gated linear attention)
    S_t = (I - b_t k_t k_t^T) a_t S_{t-1} + b_t v_t k_t^T       (gated delta rule, GDN-2)

``S_t`` is literally a set of ``d_v`` reference vectors ``S_t[:, j]`` spanned by the keys
of the past; the read-out ``S_t^T q_t`` is a *reconstruction query* into that memory.
``*_slots`` variants add ``N`` further explicit reference vectors with soft addressing,
so the model can maintain a compressed set of prototypes rather than pure decay.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_TAU_MIN = math.log(0.9)
LOG_TAU_MAX = math.log(1.1)


class _Base(nn.Module):
    """Shared temperature/readout plumbing of the reference implementation."""

    def __init__(self, vocab: int = 0, emb: int = 0, hidden: int = 0):
        super().__init__()
        self.vocab = int(vocab)
        self.emb = int(emb)
        self.hidden = int(hidden)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.out = nn.Linear(self.hidden, self.vocab)

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.out(h)
        tau = torch.exp(torch.clamp(self.log_tau, LOG_TAU_MIN, LOG_TAU_MAX))
        return logits / tau

    def init_state(self, batch: int, device: torch.device):
        raise NotImplementedError

    def detach(self, state):
        return state


class LSTMNet(_Base):
    """Reference architecture (byte LSTM + 13-step BPTT)."""

    def __init__(self, vocab, emb, hidden, layers=1):
        super().__init__()
        # RNG draw order must match the reference implementation:
        # embedding -> LSTM -> output projection -> temperature
        self.vocab, self.emb, self.hidden = int(vocab), int(emb), int(hidden)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.embedding = nn.Embedding(vocab, emb)
        self.rnn = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True)
        self.layers = layers
        self.out = nn.Linear(hidden, vocab)

    def init_state(self, batch, device):
        z = torch.zeros(self.layers, batch, self.hidden, device=device)
        return (z.clone(), z.clone())

    def detach(self, state):
        return (state[0].detach(), state[1].detach())

    def forward(self, x, state):
        out, st = self.rnn(self.embedding(x), state)
        return self._head(out[:, -1, :]), st


class GRUNet(_Base):
    def __init__(self, vocab, emb, hidden, layers=1):
        super().__init__()
        self.vocab, self.emb, self.hidden = int(vocab), int(emb), int(hidden)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.embedding = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, num_layers=layers, batch_first=True)
        self.layers = layers
        self.out = nn.Linear(hidden, vocab)

    def init_state(self, batch, device):
        return torch.zeros(self.layers, batch, self.hidden, device=device)

    def detach(self, state):
        return state.detach()

    def forward(self, x, state):
        out, st = self.rnn(self.embedding(x), state)
        return self._head(out[:, -1, :]), st


class _LinearAttn(_Base):
    """Gated linear attention (parallel form) or gated delta rule (sequential form).

    ``mode='gla'``: parallel causal linear attention with per-head exponential decay.
    ``mode='gdn'``: delta-rule recurrence, unrolled over the window (T is small).
    """

    def __init__(self, vocab, emb, hidden, heads=4, mode="gla", conv=3, slots=0):
        super().__init__(vocab, emb, hidden)
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.heads = int(heads)
        self.d = hidden // self.heads
        self.mode = mode
        self.conv_k = int(conv)
        self.n_slots = int(slots)

        self.embedding = nn.Embedding(vocab, emb)
        self.qkv = nn.Linear(emb, 3 * hidden, bias=False)
        assert 3 * hidden == 3 * heads * self.d
        self.gate_in = nn.Linear(emb, self.heads)
        self.alpha = nn.Parameter(torch.zeros(self.heads))
        self.z_gate = nn.Linear(emb, hidden, bias=False)
        self.dwconv = (nn.Conv1d(3 * hidden, 3 * hidden, conv,
                                 padding=conv - 1, groups=3 * hidden)
                       if conv > 1 else None)
        if mode == "gdn":
            self.beta_proj = nn.Linear(emb, self.heads)
            self.beta_bias = nn.Parameter(torch.full((self.heads,), -1.0))
        if self.n_slots:
            self.slot_key = nn.Parameter(torch.randn(self.n_slots, self.d) * 0.02)
            self.slot_write = nn.Linear(emb, self.n_slots, bias=False)
            self.slot_val = nn.Linear(emb, self.d, bias=False)
            self.slot_gate = nn.Linear(emb, 1, bias=False)

    # -- state ---------------------------------------------------------
    def init_state(self, batch, device):
        S = torch.zeros(batch, self.heads, self.d, self.d, device=device)
        M = (torch.zeros(batch, self.n_slots, self.d, device=device)
             if self.n_slots else None)
        return (S, M)

    def detach(self, state):
        return tuple(None if s is None else s.detach() for s in state)

    def _project(self, e, T):
        B = e.shape[0]
        qkv = self.qkv(e).view(B, T, 3, self.heads, self.d)
        if self.dwconv is not None:
            # causal depthwise conv: left-pad k-1, keep the first T frames
            c = self.dwconv(qkv.reshape(B, T, 3 * self.hidden).transpose(1, 2))
            qkv = c[:, :, :T].transpose(1, 2).view(B, T, 3, self.heads, self.d)
        q, k, v = qkv.unbind(dim=2)
        return F.silu(q), F.normalize(k, dim=-1, eps=1e-6), v / math.sqrt(self.d)

    def forward(self, x, state):
        B, T = x.shape
        e = self.embedding(x)
        q, k, v = self._project(e, T)                    # [B,T,H,d]
        # per-head log-decay, strictly negative
        log_a = -F.softplus(self.gate_in(e) + self.alpha)     # [B,T,H]
        z = F.silu(self.z_gate(e))
        S, M = state

        if self.mode == "gla":
            o, S = self._gla(q, k, v, log_a, S)
        else:
            o, S = self._gdn(q, k, v, log_a, e, S)

        if self.n_slots:
            o, M = self._slots(o, q, e, M, log_a)
        out = o.reshape(B, T, self.hidden) * z
        return self._head(out[:, -1, :]), (S, M)

    # -- gated linear attention, parallel over the window ---------------
    def _gla(self, q, k, v, log_a, S):
        """Decay weights are *inclusive*: token t sees itself with weight 1.

        For s <= t:  w[t,s] = exp(cum[t] - cum[s])  (self term: cum[t]-cum[t] = 0).
        """
        B, T, H, d = q.shape
        cum = torch.cumsum(log_a, dim=1)                       # [B,T,H]
        cumh = cum.permute(0, 2, 1)                            # [B,H,T]
        # intra-window attention, causal
        scale = torch.exp(cumh.unsqueeze(-1) - cumh.unsqueeze(-2))      # [B,H,T,T]
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        scale = torch.where(mask, scale, torch.zeros_like(scale))
        attn = torch.einsum("bthd,bshd->bhts", q, k) * scale             # [B,H,T,T]
        o_in = torch.einsum("bhts,bshd->bthd", attn, v)                  # [B,T,H,d]
        # carried state, decayed from step -1 up to t: exp(cum[t])
        q_t = q.permute(0, 2, 1, 3)                                      # [B,H,T,d]
        o_carry = torch.einsum("bhde,bhte->bthd", S, q_t)
        o_carry = o_carry * torch.exp(cumh).permute(0, 2, 1).unsqueeze(-1)
        o = o_in + o_carry
        # state after the window
        total = torch.exp(cumh[:, :, -1]).view(B, H, 1, 1)
        w = torch.exp(cumh[:, :, -1:] - cumh)                            # [B,H,T]
        dS = torch.einsum("bthd,bthe,bht->bhde", v, k, w)
        return o, total * S + dS

    # -- gated delta rule, sequential ----------------------------------
    def _gdn(self, q, k, v, log_a, e, S):
        B, T, H, d = q.shape
        beta = torch.sigmoid(self.beta_proj(e) + self.beta_bias)         # [B,T,H]
        a = torch.exp(log_a)
        outs = []
        for t in range(T):
            at = a[:, t].view(B, H, 1, 1)
            bt = beta[:, t].view(B, H, 1, 1)
            kt = k[:, t].view(B, H, d, 1)
            vt = v[:, t].view(B, H, 1, d)
            qt = q[:, t].view(B, H, d, 1)
            S = at * S
            kS = torch.matmul(kt.transpose(-1, -2), S)                   # [B,H,1,d]
            # k_t v_t^T is [B,H,d,1] @ [B,H,1,d] = [B,H,d,d]
            S = S - bt * torch.matmul(kt, kS) + bt * torch.matmul(kt, vt)
            outs.append(torch.matmul(S, qt).squeeze(-1))
        return torch.stack(outs, dim=1), S

    # -- explicit reference-vector memory ------------------------------
    def _slots(self, o, q, e, M, log_a):
        B, T, H, d = q.shape
        w = torch.softmax(self.slot_write(e), dim=-1)                     # [B,T,N]
        val = self.slot_val(e)                                            # [B,T,d]
        gate = torch.sigmoid(self.slot_gate(e))                           # [B,T,1]
        for t in range(T):
            M = gate[:, t].unsqueeze(-1) * M + w[:, t].unsqueeze(-1) * val[:, t].unsqueeze(1)
        addr = torch.einsum("bthd,nd->bhtn", q, self.slot_key) / math.sqrt(d)
        read = torch.softmax(addr, dim=-1)                                # [B,H,T,N]
        r = torch.einsum("bhtn,bnd->bthd", read, M)
        return o + r, M


class DualLSTM(nn.Module):
    """Two independently initialised byte LSTMs whose probabilities are averaged.

    An ensemble of two experts is the cheapest reliable way to buy accuracy: the
    arithmetic mean of the two softmax distributions is strictly better calibrated
    than either member, and both members are trained online by the same loss.
    """

    def __init__(self, vocab, emb, hidden, layers=1):
        super().__init__()
        self.a = LSTMNet(vocab, emb, hidden, layers)
        self.b = LSTMNet(vocab, emb, hidden, layers)
        # distinct init: perturb the second member so the ensemble is not degenerate
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(0x5EED)
            for p in self.b.parameters():
                if p.requires_grad:
                    p.add_(torch.randn(p.shape, generator=gen) * 0.01)

    def init_state(self, batch, device):
        return (self.a.init_state(batch, device), self.b.init_state(batch, device))

    def detach(self, state):
        return (self.a.detach(state[0]), self.b.detach(state[1]))

    def forward(self, x, state):
        la, sa = self.a(x, state[0])
        lb, sb = self.b(x, state[1])
        pa = torch.softmax(la, dim=1)
        pb = torch.softmax(lb, dim=1)
        p = 0.5 * (pa + pb)
        return torch.log(p.clamp_min(1e-30)), (sa, sb)


def build(arch: str, vocab: int, cfg) -> nn.Module:
    if arch == "lstm":
        return LSTMNet(vocab, cfg.emb, cfg.hidden, cfg.layers)
    if arch == "gru":
        return GRUNet(vocab, cfg.emb, cfg.hidden, cfg.layers)
    if arch == "dual":
        return DualLSTM(vocab, cfg.emb, cfg.hidden, cfg.layers)
    if arch == "ssm":
        return _LinearAttn(vocab, cfg.emb, cfg.hidden, cfg.heads, mode="gla")
    if arch == "gdn":
        return _LinearAttn(vocab, cfg.emb, cfg.hidden, cfg.heads, mode="gdn")
    if arch == "ssm_slots":
        return _LinearAttn(vocab, cfg.emb, cfg.hidden, cfg.heads, mode="gla",
                           slots=cfg.slots)
    if arch == "gdn_slots":
        return _LinearAttn(vocab, cfg.emb, cfg.hidden, cfg.heads, mode="gdn",
                           slots=cfg.slots)
    raise ValueError(f"unknown arch {arch!r}")
