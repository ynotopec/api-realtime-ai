#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"

cd "$SCRIPT_DIR"

"$SCRIPT_DIR/install.sh"

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" lock --project "$SCRIPT_DIR" --upgrade
UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" sync --project "$SCRIPT_DIR"

echo "Dependencies upgraded for $PROJECT_NAME"
