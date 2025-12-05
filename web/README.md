# Realtime Chat UI

A lightweight React + Vite front-end for the realtime WebSocket bridge. It renders markdown responses, captures inline feedback, and exposes replay controls so you can re-trigger responses for any message in the transcript.

## Quickstart

```bash
cd web
npm install
npm run dev
```

Environment defaults live in `.env.example`:

- `VITE_REALTIME_API_BASE` – dev WebSocket base URL (defaults to `ws://localhost:8080`).
- `VITE_REALTIME_API_BASE_PROD` – optional production override.
- `VITE_REALTIME_CHANNEL` – optional channel name for group routing (added as a query param).
- `VITE_REALTIME_API_KEY` – passed through the `openai-insecure-api-key` WebSocket subprotocol.

The `src/config.ts` helper picks the prod or dev base depending on `import.meta.env.MODE` so builds automatically target your deployment.
