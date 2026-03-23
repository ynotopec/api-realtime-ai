#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"

cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --upgrade uv
fi

uv venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

uv pip install --upgrade pip setuptools wheel
uv pip install --upgrade --requirements requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "Upgrade complete. Virtualenv: $VENV_DIR"
