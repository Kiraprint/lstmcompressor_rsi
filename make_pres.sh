#!/usr/bin/env bash
# Сборка презентации (слайдов) из marimo-ноутбука presentation.py.
#   bash make_pres.sh
# Требуется playwright + chromium: .venv/bin/python -m playwright install chromium
set -euo pipefail
PY=.venv/bin/python
"$PY" -c "import nbformat, nbconvert, playwright" 2>/dev/null \
  || /home/kir/.local/bin/uv pip install --python "$PY" nbformat nbconvert playwright
"$PY" -m playwright install chromium >/dev/null 2>&1 || true
"$PY" make_figs.py
"$PY" -m marimo export pdf presentation.py -o presentation.pdf \
      --as=slides --no-include-inputs
echo "presentation.pdf готов"
