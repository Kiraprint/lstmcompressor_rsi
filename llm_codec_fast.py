"""Fast LLM-prior codec: independent segments are predicted in one GPU batch.

Semantics are identical to llm_codec.py (segment-local 1000-token context,
fresh KV cache per segment, uniform first-token prior), but all segments are
advanced in lockstep so GPT-2 forwards and 50K-softmax/cumfreq construction are
batched.  The Nayuki arithmetic coder still consumes one symbol at a time from
precomputed cumulative-frequency rows.
"""
from __future__ import annotations
import io, math, struct, time
import torch
from llm_codec import (_tokenizer, VOCAB, MAGIC, cumfreq, GpuFreq, SCALE, SEG)
from ArithmeticCoder import (ArithmeticEncoder, ArithmeticDecoder,
                             BitOutputStream, BitInputStream)

class PairFreq:
    """Minimal frequency view: the encoder only reads freqs[-1], freqs[sym-1], freqs[sym]."""
    __slots__ = ("lo", "hi", "sym")

    def __init__(self, lo: int, hi: int, sym: int):
        self.lo, self.hi, self.sym = lo, hi, sym

    def __getitem__(self, i: int) -> int:
        if i == -1:
            return SCALE  # total: every row's last entry is forced to SCALE
        if i == self.sym:
            return self.hi
        if i == self.sym - 1:
            return self.lo
        raise IndexError(i)


class LLMCodecFast:
    def __init__(self, seg: int = SEG):
        self.tk = _tokenizer()
        self.model = __import__('transformers').GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='sdpa').cuda().eval()
        self.dev = 'cuda'
        self.seg = seg
        self.uniform = cumfreq(torch.full((VOCAB,), 1.0/VOCAB, device=self.dev))

    def _precompute(self, ids: list[int]):
        """Return list of cumfreq tensors: cfs[seg][pos] predicts ids[seg][pos].

        row pos is computed by feeding token pos-1 (the immediately preceding
        symbol); pos==0 is the uniform first-symbol prior with no forward.
        """
        segs = [ids[i:i+self.seg] for i in range(0, len(ids), self.seg)]
        B = len(segs); T = max(len(s) for s in segs)
        # inp[:, k] = token k of each segment; fed to predict token k+1.
        inp = torch.tensor([[(s[k] if k < len(s) else 0) for k in range(T)] for s in segs],
                           device=self.dev, dtype=torch.long)
        # Only the two cumulative frequencies the encoder reads are kept:
        # one gather on the GPU per position, one bulk copy at the end.
        hi_rows, lo_rows = [], []
        past = None
        with torch.no_grad():
            for pos in range(T):
                if pos == 0:
                    # uniform first-symbol prior, no forward (same row the decoder uses)
                    syms = inp[:, 0]
                    hi = self.uniform.gather(0, syms)
                    lo = self.uniform.gather(0, (syms - 1).clamp_min(0)) * (syms > 0)
                else:
                    out = self.model(inp[:, pos-1:pos], past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    p = torch.softmax(out.logits[:, -1].float(), dim=-1)  # [B,VOCAB]
                    c = (torch.cumsum(p.double(), dim=-1) * (SCALE - VOCAB)).long() \
                        + torch.arange(1, VOCAB+1, device=self.dev)
                    c[:, -1] = SCALE
                    syms = inp[:, pos]
                    hi = c.gather(1, syms.unsqueeze(1)).squeeze(1)
                    lo = c.gather(1, (syms - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
                    lo = lo * (syms > 0)
                hi_rows.append(hi)
                lo_rows.append(lo)
        hi_cpu = torch.stack(hi_rows).cpu().tolist()   # [T][B]
        lo_cpu = torch.stack(lo_rows).cpu().tolist()
        return (lo_cpu, hi_cpu), segs

    def compress(self, data: bytes):
        t0 = time.time()
        try:  # valid UTF-8: tokenize as text (fewer tokens on multibyte chars)
            text, mode = data.decode('utf8'), 0
        except UnicodeDecodeError:  # arbitrary bytes: latin-1 is byte-transparent
            text, mode = data.decode('latin1'), 1
        ids = self.tk.encode(text).ids
        (lo_cpu, hi_cpu), segs = self._precompute(ids)
        out = io.BytesIO(); enc = ArithmeticEncoder(32, BitOutputStream(out))
        bits = 0.0
        # write in position-major order (pos 0 of every segment, then pos 1, ...)
        # so arithmetic-coding order matches the decoder.
        for pos in range(max(len(s) for s in segs) if segs else 0):
            for b, s in enumerate(segs):
                if pos >= len(s):
                    continue
                sym = s[pos]
                lo, hi = lo_cpu[pos][b], hi_cpu[pos][b]
                enc.write(PairFreq(lo, hi, sym), sym)
                if pos > 0:
                    bits += -math.log2(max((hi - lo) / SCALE, 1e-300))
        enc.finish()
        while enc.output.numbitsfilled != 0: enc.output.write(0)
        payload = out.getvalue()
        frame = MAGIC + struct.pack('>IIII', len(data), len(ids), len(payload), mode) + payload
        return {'frame': frame, 'bits': bits, 'ntok': len(ids), 'encode_s': time.time()-t0}

    def decompress(self, frame: bytes):
        length, ntok, psz, mode = struct.unpack('>IIII', frame[6:22])
        payload = frame[22:22+psz]
        t0 = time.time()
        dec = ArithmeticDecoder(32, BitInputStream(io.BytesIO(payload)))
        seg_lens = [min(self.seg, max(0, ntok - i)) for i in range(0, ntok, self.seg)]
        B = len(seg_lens); T = max(seg_lens) if seg_lens else 0
        parts = [[] for _ in range(B)]
        past = None
        # activate all segments until the very end: pad short segments by
        # feeding their last token, but only decode real positions.
        for pos in range(T):
            if pos == 0:
                rows = [self.uniform] * B
            else:
                x = torch.tensor([[parts[b][-1] if pos-1 < seg_lens[b] else 0]
                                  for b in range(B)], device=self.dev, dtype=torch.long)
                with torch.no_grad():
                    o = self.model(x, past_key_values=past, use_cache=True)
                past = o.past_key_values
                p = torch.softmax(o.logits[:, -1].float(), dim=-1)
                c = (torch.cumsum(p.double(), dim=-1) * (SCALE - VOCAB)).long() + torch.arange(1, VOCAB+1, device=self.dev)
                c[:, -1] = SCALE
                rows = list(c)
            for b in range(B):
                if pos < seg_lens[b]:
                    parts[b].append(dec.read(GpuFreq(rows[b])))
        ids = [t for part in parts for t in part]
        text = self.tk.decode(ids)
        data = (text.encode('utf8') if mode == 0 else text.encode('latin1'))[:length]
        return {'data': data, 'ntok': len(ids), 'decode_s': time.time()-t0}
