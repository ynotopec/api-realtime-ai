#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"
RUN_DIR="$SCRIPT_DIR/.run"
PROXY_PID_FILE="$RUN_DIR/proxy.pid"
STATIC_PID_FILE="$RUN_DIR/static.pid"

IP_ARG="${1:-}"
PORT_ARG="${2:-}"

cd "$SCRIPT_DIR"
mkdir -p "$RUN_DIR"

"$SCRIPT_DIR/install.sh"

set -a
[[ -f .env ]] && source .env
set +a

IP="${IP_ARG:-${HOST:-127.0.0.1}}"
PORT="${PORT_ARG:-${PORT:-8000}}"
STATIC_PORT="${STATIC_PORT:-5173}"

stop_pid() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      sleep 1
    fi
    rm -f "$pid_file"
  fi
}

stop_pid "$PROXY_PID_FILE"
stop_pid "$STATIC_PID_FILE"

nohup env HOST="$IP" PORT="$PORT" "$VENV_DIR/bin/python" proxy.py > "$RUN_DIR/proxy.log" 2>&1 &
echo $! > "$PROXY_PID_FILE"

nohup "$VENV_DIR/bin/python" -m http.server "$STATIC_PORT" --bind "$IP" > "$RUN_DIR/static.log" 2>&1 &
echo $! > "$STATIC_PID_FILE"

echo "Started external-test"
echo "- Proxy:  ws://$IP:$PORT/ws"
echo "- Static: http://$IP:$STATIC_PORT"
echo "Logs:    tail -f $RUN_DIR/*.log"
