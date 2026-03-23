# Realtime API bridge for Voxtral on vLLM

Minimal FastAPI websocket bridge that accepts OpenAI Realtime-style client events and forwards model requests to an OpenAI-compatible backend.

Default model target is:
- `mistralai/Voxtral-Mini-4B-Realtime-2602`

## What was simplified
- one runtime app (`app.py`)
- deterministic setup scripts (`run.sh`, `upgrade.sh`)
- `uv`-based dependency management
- fixed venv location: `~/venv/<repo-basename>`
- `.env.example` + auto bootstrap `.env`

## Prerequisites
- Python 3.10+
- `ffmpeg` on PATH
- vLLM running with OpenAI-compatible API

Example vLLM launch:

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --host 0.0.0.0 \
  --port 8000
```

## Environment
Copy/edit `.env`:

```bash
cp .env.example .env
```

Important values:
- `OPENAI_API_BASE` (for example `http://127.0.0.1:8000/v1`)
- `OPENAI_API_MODEL=mistralai/Voxtral-Mini-4B-Realtime-2602`
- `OPENAI_API_KEY` (`EMPTY` is common for local vLLM)

## Run

```bash
./run.sh [IP] [PORT]
```

- If `IP`/`PORT` are omitted, `SERVER_NAME` / `SERVER_PORT` from `.env` are used.
- Script is idempotent: safe to rerun.

## Upgrade dependencies

```bash
./upgrade.sh
```

This refreshes the venv and upgrades dependencies with `uv`.
