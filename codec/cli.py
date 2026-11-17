#!/usr/bin/env python
"""Command-line entry point: compression + decompression + verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from codec.config import Config
from codec.core import decode_frame, encode_frame, run


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--input", default="data/enwik5")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    src = args.input
    dst = args.output or src + "_rsi.dat"
    back = dst + ".dec"

    raw = open(src, "rb").read()
    t0 = time.time()
    r = run(raw, cfg, compress=True)
    blob = encode_frame(r["payload"], len(raw), r["alphabet"], cfg)
    with open(dst, "wb") as f:
        f.write(blob)
    t_enc = time.time() - t0

    head, payload = decode_frame(blob)
    t1 = time.time()
    r2 = run(payload, cfg, compress=False, alphabet=head["alphabet"],
             length=head["len"])
    out = r2["data"]
    t_dec = time.time() - t1
    with open(back, "wb") as f:
        f.write(out)

    ok = bytes(out) == raw
    ratio = len(raw) / len(blob)
    res = {"success": ok, "ratio": round(ratio, 4),
           "compressed": len(blob), "original": len(raw),
           "encode_s": round(t_enc, 2), "decode_s": round(t_dec, 2),
           "total_s": round(t_enc + t_dec, 2), "bpb": round(r["bpb"], 4),
           "arch": cfg.arch, "md5": md5(back)}
    with open("result.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    if not ok:
        raise SystemExit("decompression mismatch")


if __name__ == "__main__":
    main()
