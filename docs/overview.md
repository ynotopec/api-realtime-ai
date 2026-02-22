# Overview

## Objectif
`api-realtime-ai` expose une passerelle unique pour gérer une conversation vocale en temps réel :

1. réception audio client,
2. transcription (STT),
3. génération de réponse (LLM),
4. synthèse vocale (TTS),
5. streaming de la réponse audio + texte.

## Composants

- `app.py` : API FastAPI, WebSocket `/v1/realtime`, gestion transcript/replay.
- `chat_completions.py` : endpoint `/v1/chat/completions` compatible OpenAI.
- `observability.py` : instrumentation logs/metrics/traces.
- `web/` : UI React pour tester conversations et feedback.

## Contrats d'API

- **Realtime** : `ws://<host>:<port>/v1/realtime`
- **REST Chat** : `POST /v1/chat/completions`
- **Transcripts** : `GET /api/transcripts`, `GET /api/transcripts/{id}`, `POST /api/transcripts/{id}/replay`

## Exécution

- Installation : `make install`
- Lancement : `make run`

## Dépendances explicites

- Python 3.10+
- `ffmpeg` sur le PATH
- Variables `.env` (minimum: `OPENAI_API_KEY`)
- Services STT/LLM/TTS HTTP accessibles
