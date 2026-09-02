#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
RUN_DIR="$REPO_ROOT/.codex_tmp/herg_website_launch"

stop_saved_process() {
  local name="$1"
  local pid_file="$2"
  local expected_marker="$3"
  if [[ ! -f "$pid_file" ]]; then
    print -r -- "$name is not recorded as running."
    return
  fi
  local saved_pid
  saved_pid="$(<"$pid_file")"
  if [[ "$saved_pid" != <-> ]]; then
    print -u2 "Ignoring invalid $name process identifier."
    return
  fi
  if kill -0 "$saved_pid" >/dev/null 2>&1; then
    local process_command
    process_command="$(ps -p "$saved_pid" -o command= 2>/dev/null || true)"
    if [[ "$process_command" != *"$expected_marker"* ]]; then
      print -u2 "Ignoring $name process identifier because it belongs to another process."
      return
    fi
    kill "$saved_pid"
    print -r -- "$name stopped."
  else
    print -r -- "$name was already stopped."
  fi
}

stop_saved_process "Private tunnel" "$RUN_DIR/tunnel.pid" "free.pinggy.io"
stop_saved_process "Prediction website" "$RUN_DIR/server.pid" "run_herg_prediction_website.py"

for managed_file in "$RUN_DIR/launch_info.txt" "$RUN_DIR/server.pid" "$RUN_DIR/tunnel.pid"; do
  if [[ -f "$managed_file" ]]; then
    rm -f -- "$managed_file"
  fi
done
