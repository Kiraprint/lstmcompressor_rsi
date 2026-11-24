#!/usr/bin/env bash
# Поднять marimo-ноутбук по LLM-кодеку и открыть к нему публичный туннель.
#   bash serve_llm.sh [порт]
#   MODE=run bash serve_llm.sh          # только просмотр (без правки кода)
# Печатает URL с токеном доступа. Ctrl-C — остановить оба процесса.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-2718}"
MODE="${MODE:-edit}"   # edit = виден и редактируется код; run = read-only приложение
TOKEN="${MARIMO_TOKEN:-$(head -c 9 /dev/urandom | base64 | tr -d '/+=' )}"

command -v cloudflared >/dev/null || [ -x "$HOME/.local/bin/cloudflared" ] || {
  echo "нет cloudflared: положи бинарь в ~/.local/bin/cloudflared"; exit 1; }
CF="$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")"

if [ "$MODE" = "run" ]; then
  .venv/bin/marimo run llm_notebook.py --headless --host 127.0.0.1 --port "$PORT" \
    --token-password "$TOKEN" --session-ttl 3600 > /tmp/marimo_llm.log 2>&1 &
else
  .venv/bin/marimo edit llm_notebook.py --headless --host 127.0.0.1 --port "$PORT" \
    --token-password "$TOKEN" --session-ttl 3600 > /tmp/marimo_llm.log 2>&1 &
fi
MARIMO_PID=$!
trap 'kill $MARIMO_PID $CF_PID 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 60); do
  grep -q "URL:" /tmp/marimo_llm.log && break; sleep 1
done

"$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > /tmp/cf_tunnel.log 2>&1 &
CF_PID=$!
for _ in $(seq 1 60); do
  URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" /tmp/cf_tunnel.log | head -1) || true
  [ -n "${URL:-}" ] && break; sleep 1
done
[ -n "${URL:-}" ] || { echo "туннель не поднялся, см. /tmp/cf_tunnel.log"; exit 1; }

cat <<TXT

  Ноутбук:  $URL/?access_token=$TOKEN
  Токен:    $TOKEN
  Логи:     /tmp/marimo_llm.log, /tmp/cf_tunnel.log

  Туннель публичный: ссылку и токен не публикуй. Ctrl-C — остановить.

TXT
wait $MARIMO_PID
