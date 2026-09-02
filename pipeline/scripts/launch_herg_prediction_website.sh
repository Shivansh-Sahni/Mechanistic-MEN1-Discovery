#!/bin/zsh
set -euo pipefail

if [[ "${HERG_ENABLE_PUBLIC_TUNNEL:-0}" != "1" ]]; then
  print -u2 "Public tunneling is disabled by default to protect local research structures."
  print -u2 "Use launch_herg_prediction_website_local.sh for localhost."
  print -u2 "Set HERG_ENABLE_PUBLIC_TUNNEL=1 only after explicit authorization for a temporary tunnel."
  exit 2
fi

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
SERVER_SCRIPT="$REPO_ROOT/pipeline/scripts/run_herg_prediction_website.py"
SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
RUN_DIR="$REPO_ROOT/.codex_tmp/herg_website_launch"
SERVER_LOG="$RUN_DIR/server.log"
TUNNEL_LOG="$RUN_DIR/tunnel.log"
KNOWN_HOSTS="$RUN_DIR/pinggy_known_hosts"
SERVER_PID_FILE="$RUN_DIR/server.pid"
TUNNEL_PID_FILE="$RUN_DIR/tunnel.pid"
LAUNCH_INFO="$RUN_DIR/launch_info.txt"
PORT="${HERG_WEBSITE_PORT:-8796}"
AUTH_USERNAME="${HERG_WEBSITE_USERNAME:-research}"
AUTH_PASSWORD="${HERG_WEBSITE_PASSWORD:-$(openssl rand -hex 8)}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "Project Python is unavailable: $PYTHON_BIN"
  exit 2
fi
if [[ ! -x "$SSH_BIN" ]]; then
  print -u2 "The SSH tunnel client is unavailable: $SSH_BIN"
  exit 2
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "Port $PORT is already in use. Stop the previous website launch first."
  exit 2
fi

mkdir -p "$RUN_DIR"
chmod 700 "$RUN_DIR"

cleanup_failed_launch() {
  if [[ -f "$TUNNEL_PID_FILE" ]]; then
    local tunnel_pid
    tunnel_pid="$(<"$TUNNEL_PID_FILE")"
    kill "$tunnel_pid" >/dev/null 2>&1 || true
  fi
  if [[ -f "$SERVER_PID_FILE" ]]; then
    local server_pid
    server_pid="$(<"$SERVER_PID_FILE")"
    kill "$server_pid" >/dev/null 2>&1 || true
  fi
  for managed_file in "$LAUNCH_INFO" "$SERVER_PID_FILE" "$TUNNEL_PID_FILE"; do
    [[ ! -f "$managed_file" ]] || rm -f -- "$managed_file"
  done
}
trap cleanup_failed_launch EXIT
trap 'exit 130' INT TERM

cd "$REPO_ROOT"
"$PYTHON_BIN" "$SERVER_SCRIPT" validate >/dev/null
HERG_WEBSITE_USERNAME="$AUTH_USERNAME" HERG_WEBSITE_PASSWORD="$AUTH_PASSWORD" \
  nohup "$PYTHON_BIN" "$SERVER_SCRIPT" serve --host 127.0.0.1 --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
print -r -- "$SERVER_PID" >"$SERVER_PID_FILE"

for _ in {1..60}; do
  if curl --fail --silent "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    print -u2 "The website process stopped during startup."
    exit 1
  fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:$PORT/api/health" >/dev/null

LOCAL_HTTP_STATUS="$(
  curl --silent --output /dev/null --write-out '%{http_code}' "http://127.0.0.1:$PORT/"
)"
if [[ "$LOCAL_HTTP_STATUS" != "401" ]]; then
  print -u2 "The local website did not enforce password protection."
  exit 1
fi
curl --fail --silent --user "$AUTH_USERNAME:$AUTH_PASSWORD" \
  "http://127.0.0.1:$PORT/" >/dev/null

nohup "$SSH_BIN" -T \
  -p 443 \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$KNOWN_HOSTS" \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R "0:127.0.0.1:$PORT" \
  free.pinggy.io >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID="$!"
print -r -- "$TUNNEL_PID" >"$TUNNEL_PID_FILE"

PUBLIC_URL=""
ALTERNATE_URL=""
for _ in {1..60}; do
  PUBLIC_URL="$(
    rg -o 'https://[a-z0-9-]+\.free\.pinggy\.net' "$TUNNEL_LOG" \
      | head -n 1 || true
  )"
  ALTERNATE_URL="$(
    rg -o 'https://[a-z0-9-]+\.run\.pinggy-free\.link' "$TUNNEL_LOG" \
      | head -n 1 || true
  )"
  if [[ -n "$PUBLIC_URL" ]]; then
    break
  fi
  if ! kill -0 "$TUNNEL_PID" >/dev/null 2>&1; then
    print -u2 "The private tunnel stopped during startup."
    exit 1
  fi
  sleep 1
done
if [[ -z "$PUBLIC_URL" ]]; then
  print -u2 "The private website URL was not created in time."
  exit 1
fi

HTTP_STATUS=""
for _ in {1..6}; do
  HTTP_STATUS="$(
    curl --silent --connect-timeout 3 --max-time 5 --output /dev/null \
      --write-out '%{http_code}' "$PUBLIC_URL/" || true
  )"
  if [[ "$HTTP_STATUS" == "401" ]]; then
    break
  fi
  if ! kill -0 "$TUNNEL_PID" >/dev/null 2>&1; then
    print -u2 "The private tunnel stopped before it became reachable."
    exit 1
  fi
  sleep 1
done
PUBLIC_AUTH_VERIFIED=0
if [[ "$HTTP_STATUS" == "401" ]]; then
  AUTHENTICATED=0
  for _ in {1..3}; do
    if curl --fail --silent --connect-timeout 3 --max-time 5 \
      --user "$AUTH_USERNAME:$AUTH_PASSWORD" "$PUBLIC_URL/" >/dev/null; then
      AUTHENTICATED=1
      PUBLIC_AUTH_VERIFIED=1
      break
    fi
    sleep 1
  done
  if [[ "$AUTHENTICATED" != "1" ]]; then
    print -u2 "The public website did not accept the generated review credentials."
    exit 1
  fi
elif [[ -n "$HTTP_STATUS" && "$HTTP_STATUS" != "000" ]]; then
  print -u2 "The public website did not become reachable with password protection."
  exit 1
else
  print -u2 "Warning: the lab command-line network could not reach the assigned HTTPS domain."
  print -u2 "Local password protection passed; verify the public URL in a browser."
fi

umask 077
{
  print -r -- "URL=$PUBLIC_URL"
  print -r -- "ALTERNATE_URL=$ALTERNATE_URL"
  print -r -- "USERNAME=$AUTH_USERNAME"
  print -r -- "PASSWORD=$AUTH_PASSWORD"
  print -r -- "SERVER_PID=$SERVER_PID"
  print -r -- "TUNNEL_PID=$TUNNEL_PID"
  print -r -- "PUBLIC_AUTH_VERIFIED=$PUBLIC_AUTH_VERIFIED"
} >"$LAUNCH_INFO"
chmod 600 "$LAUNCH_INFO"

print -r -- "Website launched successfully."
print -r -- "URL: $PUBLIC_URL"
if [[ -n "$ALTERNATE_URL" ]]; then
  print -r -- "Alternate URL: $ALTERNATE_URL"
fi
print -r -- "Username: $AUTH_USERNAME"
print -r -- "Password: $AUTH_PASSWORD"
print -r -- "Pinggy may show a one-time tunnel notice before the login prompt."
print -r -- "Free Pinggy URLs expire after about 60 minutes."
if [[ "$PUBLIC_AUTH_VERIFIED" == "1" ]]; then
  print -r -- "Public password protection: verified"
else
  print -r -- "Public password protection: browser verification pending (local protection verified)"
fi
print -r -- "Run pipeline/scripts/stop_herg_prediction_website.sh when the review is finished."

while kill -0 "$SERVER_PID" >/dev/null 2>&1 \
  && kill -0 "$TUNNEL_PID" >/dev/null 2>&1; do
  sleep 5
done
print -u2 "The public website stopped because one of its managed processes exited."
exit 1
