"""Configuration for the online neural text codec.

Every field here is a *decoder-symmetric* hyper-parameter: the same value is used by
the encoder and the decoder, so nothing has to be transmitted except the byte
alphabet (a 256-bit mask) and the file length.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---- loop geometry -------------------------------------------------
    n_streams: int = 128          # interleaved chunks (batch dimension)
    seq_len: int = 13             # BPTT window
    seed: int = 1234

    # ---- neural expert(s) ----------------------------------------------
    arch: str = "lstm"            # lstm | gru | ssm | gdn | ssm_slots | dual
    emb: int = 150
    hidden: int = 768
    layers: int = 1
    heads: int = 4                # SSM heads
    slots: int = 32               # explicit "reference vectors"

    # ---- optimisation ---------------------------------------------------
    lr: float = 5e-3
    lr_end: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.9999
    eps: float = 1e-5
    clip: float = 3.0
    weight_decay: float = 0.0
    sched: str = "linear"         # linear | cosine | const

    # ---- optimiser ------------------------------------------------------
    optimizer: str = "adam"       # adam | muon
    muon_lr_scale: float = 1.0    # extra LR multiplier on Muon matrices
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # ---- statistical context mixing -------------------------------------
    mixer: bool = True
    orders: Tuple[int, ...] = (1, 2, 3, 4, 6)
    mix_lr: float = 0.02
    mix_init: float = 4.0        # initial log-weight in favour of the neural expert
    mix_decay: float = 1.0        # count decay per update (1.0 = none)
    count_prior: float = 0.35     # additive smoothing per symbol
    sse: bool = False             # secondary symbol estimation stage

    # ---- arithmetic coding ----------------------------------------------
    freq_scale: int = 1 << 24

    # ---- runtime ---------------------------------------------------------
    device: str = "cuda"

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["orders"] = list(self.orders)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        if "orders" in d:
            d["orders"] = tuple(d["orders"])
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
