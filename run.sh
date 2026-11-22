#!/usr/bin/env bash
# Однострочный запуск обучения/сжатия на enwik5.
#   bash run.sh                 # сжать data/enwik5
#   bash run.sh data/файл.bin   # сжать произвольный файл
set -euo pipefail
SRC="${1:-data/enwik5}"

command -v uv >/dev/null || { echo "нужен uv: https://docs.astral.sh/uv"; exit 1; }

# 1) окружение
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
# 2) torch под CUDA (индекс PyTorch)
.venv/bin/python -c "import torch, torch.cuda; assert torch.cuda.is_available()" 2>/dev/null \
  || uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
# 3) остальные зависимости
.venv/bin/python -c "import numpy, marimo, matplotlib, pandas" 2>/dev/null \
  || uv pip install --python .venv -r requirements.txt
# 4) прогон: сжатие + распаковка + сверка MD5
if [ "${2:-}" = "--llm" ]; then
  .venv/bin/python -c "import transformers, tokenizers, huggingface_hub" 2>/dev/null \
    || uv pip install --python .venv transformers tokenizers huggingface_hub
  .venv/bin/python - "$SRC" <<'PY'
import sys, hashlib
from llm_codec_fast import LLMCodecFast
data = open(sys.argv[1], 'rb').read()
c = LLMCodecFast(); r = c.compress(data)
out = c.decompress(r['frame'])['data']
md5 = hashlib.md5(data).hexdigest()
print(f"LLM codec: {len(data)} -> {len(r['frame'])} bytes, "
      f"ratio {len(data)/len(r['frame']):.4f}, "
      f"enc {r['encode_s']:.1f}s, md5 {md5}, "
      f"MATCH {out == data}")
PY
else
  .venv/bin/python -m codec.cli --config config.json --input "$SRC"
fi
