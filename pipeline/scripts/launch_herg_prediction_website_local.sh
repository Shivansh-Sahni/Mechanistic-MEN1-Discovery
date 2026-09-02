#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
SERVER_SCRIPT="$REPO_ROOT/pipeline/scripts/run_herg_prediction_website.py"
RUN_DIR="$REPO_ROOT/.codex_tmp/herg_website_launch"
SERVER_LOG="$RUN_DIR/server.log"
SERVER_PID_FILE="$RUN_DIR/server.pid"
LAUNCH_INFO="$RUN_DIR/launch_info.txt"
PORT="${HERG_WEBSITE_PORT:-8795}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "Project Python is unavailable: $PYTHON_BIN"
  exit 2
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "Port $PORT is already in use. Stop the previous website launch first."
  exit 2
fi

mkdir -p "$RUN_DIR"
chmod 700 "$RUN_DIR"
cd "$REPO_ROOT"
"$PYTHON_BIN" "$SERVER_SCRIPT" validate >/dev/null
nohup "$PYTHON_BIN" "$SERVER_SCRIPT" serve --host 127.0.0.1 --port "$PORT" \
  --enable-internal-examples \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
print -r -- "$SERVER_PID" >"$SERVER_PID_FILE"

for _ in {1..90}; do
  if curl --fail --silent "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    print -u2 "The localhost website process stopped during startup."
    exit 1
  fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:$PORT/api/health" >/dev/null
curl --fail --silent "http://127.0.0.1:$PORT/" >/dev/null

umask 077
{
  print -r -- "URL=http://127.0.0.1:$PORT/"
  print -r -- "SERVER_PID=$SERVER_PID"
  print -r -- "SCOPE=localhost_only"
} >"$LAUNCH_INFO"
chmod 600 "$LAUNCH_INFO"

print -r -- "hERG localhost website is ready."
print -r -- "URL: http://127.0.0.1:$PORT/"
print -r -- "Scope: localhost only; no tunnel was created."
print -r -- "Stop: pipeline/scripts/stop_herg_prediction_website.sh"
