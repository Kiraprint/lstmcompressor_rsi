"""LLM-prior codec: 1000-token segments, each a fresh never-cropped KV cache.

Context for a token is its own segment prefix (window semantics).  Because a
segment never exceeds GPT-2's 1024-token window there is no crop and no manual
position bookkeeping.  Encoder and decoder replay identical segments.
"""
from __future__ import annotations
import io, math, struct, time
import torch
from transformers import GPT2LMHeadModel
from tokenizers import Tokenizer as _TK, pre_tokenizers as _pt, decoders as _dec
from tokenizers.models import BPE as _BPE
from huggingface_hub import hf_hub_download as _hdl
VOCAB = 50257
MAGIC = b"NSLLM1"

def _tokenizer():
    tk = _TK(_BPE(vocab=_hdl("gpt2", "vocab.json"), merges=_hdl("gpt2", "merges.txt")))
    tk.pre_tokenizer = _pt.ByteLevel(add_prefix_space=False)
    tk.decoder = _dec.ByteLevel()
    return tk
from ArithmeticCoder import (ArithmeticEncoder, ArithmeticDecoder,
                             BitOutputStream, BitInputStream)

SCALE = 1 << 24
SEG = 1000

def cumfreq(p: torch.Tensor) -> torch.Tensor:
    n = p.numel()
    c = (torch.cumsum(p.double(), dim=-1) * (SCALE - n)).long() + torch.arange(1, n + 1, device=p.device)
    c[-1] = SCALE
    return c

class GpuFreq:
    def __init__(self, c): self.c=c; self.n=c.numel()
    def __len__(self): return self.n
    def __getitem__(self, i): return int(self.c[i]) if i != -1 else int(self.c[-1])

class LLMCodecSeg:
    def __init__(self):
        self.tk=_tokenizer(); self.model=GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval(); self.dev='cuda'
        self.uniform=cumfreq(torch.full((VOCAB,),1.0/VOCAB,device=self.dev))

    def _step(self, token, past):
        x=torch.tensor([[token]],device=self.dev,dtype=torch.long)
        with torch.no_grad():
            out=self.model(x,past_key_values=past,use_cache=True)
        return torch.softmax(out.logits[0,-1].float(),dim=-1), out.past_key_values

    def compress(self, data: bytes):
        ids=self.tk.encode(data.decode('latin1')).ids
        out=io.BytesIO(); enc=ArithmeticEncoder(32,BitOutputStream(out))
        bits=0.0; t0=time.time()
        N=len(ids)
        for bs in range(0,N,SEG):
            be=min(bs+SEG,N)
            past=None
            for j in range(bs,be):
                sym=ids[j]
                if j==bs:
                    cu=self.uniform; ps=1.0/VOCAB
                else:
                    p,past=self._step(ids[j-1],past); cu=cumfreq(p); ps=float(p[sym])
                enc.write(GpuFreq(cu),sym); bits+=-math.log2(max(ps,1e-300))
        enc.finish()
        while enc.output.numbitsfilled!=0: enc.output.write(0)
        payload=out.getvalue()
        frame=MAGIC+struct.pack('>IIII',len(data),len(ids),len(payload),0)+payload
        return {'frame':frame,'bits':bits,'ntok':len(ids),'encode_s':time.time()-t0}

    def decompress(self, frame: bytes):
        length,ntok,psz,_=struct.unpack('>IIII',frame[6:22]); payload=frame[22:22+psz]
        dec=ArithmeticDecoder(32,BitInputStream(io.BytesIO(payload)))
        ids=[]; t0=time.time()
        for bs in range(0,ntok,SEG):
            be=min(bs+SEG,ntok); part=[]; past=None
            for j in range(bs,be):
                if not part: cu=self.uniform
                else:
                    p,past=self._step(part[-1],past); cu=cumfreq(p)
                k=dec.read(GpuFreq(cu)); part.append(k)
            ids.extend(part)
        data=self.tk.decode(ids).encode('latin1')[:length]
        return {'data':data,'ntok':len(ids),'decode_s':time.time()-t0}
