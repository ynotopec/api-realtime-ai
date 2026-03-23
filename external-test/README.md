# external-test (minimal + automated)

Simple browser/CLI smoke tests for realtime behavior.

## One-time setup (idempotent)

```bash
cd external-test
make install
```

What `make install` does:
- creates `~/venv/external-test`
- installs dependencies with `uv`
- creates `.env` from `.env.example` if missing

## Start services

```bash
make start                 # defaults: 127.0.0.1 8000 5173
make start 0.0.0.0 8000    # optional IP/PORT override
```

This launches in background:
- websocket bridge: `ws://<IP>:<PORT>/ws`
- static test pages: `http://<IP>:5173`

Pages:
- `/index.html`
- `/echo-cancel.html`
- `/half-duplex.html`

## Stop / status / upgrade

```bash
make status
make stop
make upgrade
```

## Environment

Edit `.env`:
- `OPENAI_API_KEY`
- `OPENAI_API_BASE` (optional)
- `HOST`, `PORT`, `STATIC_PORT`
