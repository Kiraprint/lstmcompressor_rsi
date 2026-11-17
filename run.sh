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
.venv/bin/python -m codec.cli --config config.json --input "$SRC"
