# external-test

Minimal realtime smoke-test setup with idempotent scripts.

## What this setup gives you

- `uv`-managed dependencies
- virtual environment at `~/venv/<project-name>`
- safe to rerun scripts (idempotent)
- `.env.example` template
- one-command start and upgrade

## Files

- `install.sh` — install `uv` (if needed), sync dependencies, create `.env` if missing
- `start.sh [IP] [PORT]` — start proxy + static server
- `stop.sh` — stop both background processes
- `upgrade.sh` — upgrade lock/dependencies with `uv`
- `.env.example` — starter environment file copied to `.env` by `install.sh`

## Usage

```bash
cd external-test
./install.sh
./start.sh             # default: 127.0.0.1 8000
./start.sh 0.0.0.0 9000
./stop.sh
./upgrade.sh
```

## Environment

Copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE` (optional)
- `HOST` / `PORT` / `STATIC_PORT` (optional defaults)
