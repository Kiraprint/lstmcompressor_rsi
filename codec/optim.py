"""Muon optimizer (momentum + Newton-Schulz orthogonalisation) for the codec.

Muon is applied to 2-D recurrent / projection matrices; 1-D parameters and the
byte embedding keep Adam, which is the standard mixed recommendation.  The
optimizer is fully deterministic so encoder and decoder replay the same
trajectory.
"""

from __future__ import annotations

import math

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalise a 2-D gradient with the Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonAdam(torch.optim.Optimizer):
    """Muon for matrices, Adam (beta1=0) for everything else."""

    def __init__(self, params, lr, betas=(0.0, 0.9999), eps=1e-5,
                 muon_lr_scale=1.0, momentum=0.95, ns_steps=5,
                 muon_params=None):
        params = list(params)
        muon_set = set(muon_params or [])
        self.muon_set = muon_set
        # group 0: Muon matrices, group 1: Adam params
        groups = [
            {"params": [p for p in params if p in muon_set], "lr": lr,
             "lr_scale": muon_lr_scale},
            {"params": [p for p in params if p not in muon_set], "lr": lr,
             "lr_scale": 1.0},
        ]
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps,
                                      momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if not group["params"]:
                continue
            lr = group["lr"] * group.get("lr_scale", 1.0)
            if group is self.param_groups[0]:
                momentum = group["momentum"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.reshape(g.shape[0], -1)
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    u = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                    # standard Muon per-matrix scale: sqrt(max(rows/cols, 1))
                    scale = math.sqrt(max(1.0, u.shape[0] / u.shape[1]))
                    p.add_(u.reshape(p.shape), alpha=-lr * scale)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    t = st["step"]
                    st["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                    st["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bc2 = 1 - beta2 ** t
                    denom = (st["exp_avg_sq"] / bc2).sqrt().add_(eps)
                    # beta1 == 0 (reference setting): no bias correction needed
                    p.addcdiv_(st["exp_avg"], denom, value=-lr)
        return loss


def build_optimizer(cfg, model, muon_params):
    """Return Adam (reference) or MuonAdam depending on cfg.optimizer."""
    if getattr(cfg, "optimizer", "adam") == "muon":
        return MuonAdam(
            model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, muon_lr_scale=getattr(cfg, "muon_lr_scale", 1.0),
            momentum=getattr(cfg, "muon_momentum", 0.95),
            ns_steps=getattr(cfg, "muon_ns_steps", 5),
            muon_params=muon_params,
        )
    return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
                            weight_decay=cfg.weight_decay)
