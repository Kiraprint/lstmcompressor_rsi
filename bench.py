"""Benchmark multiple codec configurations and persist results + learning curves."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from codec.config import Config
from codec.core import decode_frame, encode_frame, run

DATA = "data/enwik5"


def bench(name: str, cfg: Config, data: bytes, verbose: bool = True) -> dict:
    alphabet = sorted(set(data))
    t0 = time.time()
    r = run(data, cfg, compress=True, alphabet=alphabet)
    blob = encode_frame(r["payload"], len(data), alphabet, cfg)
    t_enc = time.time() - t0
    head, payload = decode_frame(blob)
    t1 = time.time()
    r2 = run(payload, cfg, compress=False, alphabet=head["alphabet"], length=head["len"])
    t_dec = time.time() - t1
    ok = r2["data"] == data
    res = {"name": name, "ok": ok, "config": cfg.to_dict(),
           "ratio": len(data) / len(blob), "bytes": len(blob),
           "enc_s": round(t_enc, 2), "dec_s": round(t_dec, 2),
           "total_s": round(t_enc + t_dec, 2), "bpb": round(r["bpb"], 4),
           "curve": r["curve"]}
    if verbose:
        print(f"{name:22s} ratio={res['ratio']:.4f} bytes={res['bytes']:6d} "
              f"t={res['total_s']:6.2f}s bpb={res['bpb']:.4f} ok={ok}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["quick", "full", "smoke"], default="full")
    ap.add_argument("--subset", type=int, default=0)
    args = ap.parse_args()

    data = open(DATA, "rb").read()
    if args.subset:
        data = data[:args.subset]

    base = Config()

    if args.suite == "smoke":
        cases = [("smoke", base.replace(n_streams=16, hidden=192, emb=64))]
    else:
        cases = [
            ("baseline_reimpl", base.replace(mixer=False, arch="lstm")),
            ("lstm+stats", base.replace(mixer=True, arch="lstm")),
            ("gru+stats", base.replace(mixer=True, arch="gru")),
            ("ssm_gla", base.replace(mixer=True, arch="ssm", hidden=512, heads=4)),
            ("gdn_delta", base.replace(mixer=True, arch="gdn", hidden=512, heads=4)),
            ("ssm_slots", base.replace(mixer=True, arch="ssm_slots", hidden=512, heads=4)),
            ("gdn_slots", base.replace(mixer=True, arch="gdn_slots", hidden=512, heads=4)),
        ]
        if args.suite == "quick":
            cases = cases[:4]

    results = []
    for name, cfg in cases:
        try:
            results.append(bench(name, cfg, data))
        except Exception as e:
            print(f"{name}: FAIL {type(e).__name__}: {e}", flush=True)
            results.append({"name": name, "ok": False, "error": repr(e)})
        with open("experiments.json", "w") as f:
            json.dump(results, f, indent=2)
    print("saved experiments.json")


if __name__ == "__main__":
    main()
