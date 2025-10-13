# Realtime Speech Bridge (FastAPI + WebSockets)

A lightweight FastAPI service that bridges **browser audio → ASR (Whisper)** → **LLM replies** → **TTS** via a single OpenAI-style realtime WebSocket endpoint.

* **`/v1/realtime` (WebSocket)**: accepts OpenAI Realtime-compatible events, handles optional server-side VAD, streams TTS audio back to the caller, and mirrors conversation state events.

> Uses `ffmpeg` for audio muxing/resampling, `webrtcvad` for server VAD, and upstream Whisper/TTS/LLM services accessible over HTTP.

---

## Table of contents

* [Architecture](#architecture)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
* [Realtime WebSocket Bridge (`/v1/realtime`)](#realtime-websocket-bridge-v1realtime)
* [Examples](#examples)
* [Deployment](#deployment)
* [Security notes](#security-notes)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Architecture

```
Browser mic (24k PCM)  ──► FastAPI WebSocket (/v1/realtime)
         │                    │
         │                    ▼
         │             Server VAD (webrtcvad)
         │                    │ (chunking)
         │                    ▼
         │               Whisper API (HTTP)
         │                    │
         │                    ▼
         │              Transcription text
         │                    │
         │                    ▼
         │         LLM chat completions (HTTP)
         │                    │
         │                    ▼
         │                 TTS API
         │                    │
         ▼                    ▼
     Transcripts        Streamed PCM/Opus back to client
```

---

## Features

* **Realtime transcription pipeline** powered by Whisper.
* **OpenAI-like realtime bridge** (`/v1/realtime`):
  * Accepts `input_audio_buffer.append` events with base64 PCM24k frames.
  * Optional **server-side VAD** (configurable via `session.update`).
  * Streams text deltas and PCM24k audio chunks back to the client.
  * Supports response cancellation and auto-response generation triggered by VAD.
* **Token gate** for realtime endpoints via `API_TOKENS`.

---

## Prerequisites

* Python 3.10+
* **ffmpeg** installed and available on `$PATH` (used for (de)muxing & resampling)
* Reachable **Whisper**, **LLM**, and **TTS** HTTP endpoints

Required Python packages (see `requirements.txt`):

```
fastapi
uvicorn[standard]
requests
numpy
webrtcvad
websockets
```

---

## Quickstart

### 1) Environment variables

Create a `.env` file or export the environment variables described in [Configuration](#configuration). A minimal setup looks like:

```bash
export AUDIO_API_KEY=...          # Whisper backend
export OPENAI_API_KEY=...         # LLM for conversation replies
export OPENAI_API_BASE=https://your-openai-compatible-host
export OPENAI_API_MODEL=gpt-oss
export TTS_API_KEY=...
export TTS_API_URL=https://api-txt2audio.cloud-pi-native.com/v1/audio/speech
export API_TOKENS="token1,token2" # optional auth for realtime WebSocket
export SERVER_NAME=0.0.0.0        # optional host binding
export SERVER_PORT=8080           # optional port binding
```

### 2) Install & run

```bash
pip install -r requirements.txt
uvicorn app:app --host ${SERVER_NAME:-0.0.0.0} --port ${SERVER_PORT:-8080} --ws websockets
```

> Alternatively run `python app.py` which bootstraps `uvicorn` with the same defaults.

---

## Configuration

| Variable            | Required                 | Default                                                             | Description                                                                    |
| ------------------- | ------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `AUDIO_API_KEY`     | ✅                        | –                                                                   | API key for Whisper transcription backend.                                     |
| `OPENAI_API_KEY`    | ✅                        | –                                                                   | API key for OpenAI-compatible chat endpoint (conversation replies).            |
| `OPENAI_API_BASE`   | ✅ (if not default)       | ``                                                                  | Base URL for OpenAI-compatible API (must expose `/chat/completions`).          |
| `OPENAI_API_MODEL`  | ❌                        | `gpt-oss`                                                           | Model name for conversation replies.                                           |
| `WHISPER_URL`       | ❌                        | `https://api-audio2txt.cloud-pi-native.com/v1/audio/transcriptions` | Whisper transcription endpoint.                                                |
| `TTS_API_KEY`       | ❌ (required if TTS used) | –                                                                   | API key for TTS endpoint.                                                      |
| `TTS_API_URL`       | ❌                        | `https://api-txt2audio.cloud-pi-native.com/v1/audio/speech`         | TTS endpoint.                                                                  |
| `REQUEST_TIMEOUT`   | ❌                        | `30`                                                                | HTTP timeout (seconds) for upstream calls.                                     |
| `DEFAULT_SYSTEM_PROMPT` | ❌                    | `You are a concise, upbeat assistant…`                              | Default system prompt enforcing short, focused sentences.                       |
| `DEFAULT_TTS_INSTRUCTIONS` | ❌                | `Speak clearly and positively…`                                     | Default TTS instructions promoting concise speech.                              |
| `DEFAULT_VAD_AGGR`   | ❌                        | `2`                                                                 | Default webrtcvad aggressiveness (0..3).                                        |
| `DEFAULT_VAD_START_MS` | ❌                     | `120`                                                               | Minimum voiced audio (ms) before considering speech started.                    |
| `DEFAULT_VAD_END_MS` | ❌                       | `450`                                                               | Required trailing silence (ms) before cutting a turn.                           |
| `DEFAULT_VAD_PAD_MS` | ❌                       | `180`                                                               | Additional padding (ms) kept around detected speech.                            |
| `DEFAULT_VAD_MAX_MS` | ❌                       | `5000`                                                              | Maximum turn length (ms) before force-closing a chunk.                          |
| `DEFAULT_VAD_AUTORESP` | ❌                     | `1`                                                                 | Auto-trigger a response when VAD commits user speech.                           |
| `DEFAULT_VAD_INTERRUPT_RESPONSE` | ❌          | `0`                                                                 | Interrupt streaming replies when new speech arrives via server VAD.             |
| `API_TOKENS`        | ❌                        | –                                                                   | Comma-separated tokens to allow access to realtime endpoint. Empty = no auth.  |
| `SERVER_NAME`       | ❌                        | `0.0.0.0`                                                           | Bind address for the FastAPI/uvicorn server.                                   |
| `SERVER_PORT`       | ❌                        | `8080`                                                              | Bind port for the FastAPI/uvicorn server.                                      |

### VAD tuning defaults

Server-side VAD is opt-in (via `session.update`), but the defaults above help keep chunks aligned with spoken sentences. Lower `DEFAULT_VAD_END_MS` and `DEFAULT_VAD_MAX_MS` if you need quicker, more granular commits, or raise them when you prefer longer turns. Boolean toggles such as `DEFAULT_VAD_AUTORESP` accept `1/0`, `true/false`, `yes/no`, or `on/off`. All timings are expressed in milliseconds.

**Audio formats**

* **Input** realtime: base64 **PCM16** at **24 kHz** (frame size = 20 ms, 960 bytes).
* Whisper upload: `audio/webm` (Opus) is produced internally from PCM using `ffmpeg`.
* TTS: external API returns WebM/Opus; helper converts to **PCM16 24 kHz** for streaming and **16 kHz** where needed.

---

## Realtime WebSocket Bridge (`/v1/realtime`)

* FastAPI WebSocket endpoint mimicking an OpenAI Realtime flow.
* Auth: optional Bearer token, `?token=` query parameter, or `Sec-WebSocket-Protocol: openai-insecure-api-key.<token>` (when `API_TOKENS` is configured).

**Client → server events**

* `session.update`: `{ "session": { "voice": "shimmer", "modalities":["audio","text"], ... } }`
* `conversation.item.create`: `{ "item": { "id?": "...", "type":"message", "role":"user", "content":[ {"type":"input_text", "text":"..."} ] } }`
* `input_audio_buffer.append`: `{ "audio": "<base64 pcm24k>" }`
* `input_audio_buffer.commit`: no payload → triggers transcription
* `input_audio_buffer.clear`: clears buffered audio / VAD state
* `conversation.item.retrieve` / `conversation.item.delete`
* `response.create`: no payload → generates reply + audio
* `response.cancel`: cancel the currently streaming response (if IDs match)

**Server → client events**

* `session.created`, `session.updated`
* `conversation.item.created` / `retrieved` / `deleted`
* `conversation.item.input_audio_transcription.completed`: Whisper transcription result for committed audio
* `response.output_text.delta`: text token(s)
* `response.audio.delta`: base64 PCM24k chunks
* `response.done`: end of response
* `response.cancelled`: emitted when the server cancels a response because of client request or VAD interruption
* `error`: `{ error: { message: "..." } }`

---

## Examples

### Node.js WebSocket client

```js
import WebSocket from "ws";
const ws = new WebSocket("ws://localhost:8080/v1/realtime", {
  headers: { Authorization: "Bearer TOKEN" }
});
ws.on("open", () => {
  ws.send(JSON.stringify({ type: "session.update", session: { voice: "shimmer" } }));
  // append audio chunks as base64 PCM24k strings
  ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: "<b64>" }));
  ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  ws.send(JSON.stringify({ type: "response.create" }));
});
ws.on("message", (d) => console.log(JSON.parse(d.toString())));
```

### Python client snippet

```python
import asyncio
import base64
import websockets

async def main():
    uri = "ws://localhost:8080/v1/realtime"
    async with websockets.connect(uri, extra_headers={"Authorization": "Bearer TOKEN"}) as ws:
        await ws.send('{"type": "session.update", "session": {"voice": "shimmer"}}')
        # send PCM16 (24 kHz) audio
        await ws.send('{"type": "input_audio_buffer.append", "audio": "' + base64.b64encode(b"\x00\x00" * 960).decode() + '"}')
        await ws.send('{"type": "input_audio_buffer.commit"}')
        await ws.send('{"type": "response.create"}')
        async for message in ws:
            print(message)

asyncio.run(main())
```

---

## Deployment

* **Production server**: run `uvicorn` (or `hypercorn`) with WebSocket support. Example:

  ```bash
  uvicorn app:app --host 0.0.0.0 --port 8080 --ws websockets
  ```
* **Containers**: ensure `ffmpeg` binary is present in the image.
* **Scaling**: conversation state lives in-memory per process; use sticky sessions or external state if horizontal scaling is required.
* **CPU/GPU**: ASR/TTS are upstream services; this app mainly does muxing/resampling (CPU-bound) and I/O.

---

## Security notes

* Gate realtime endpoints with `API_TOKENS`.
* Use HTTPS/secure proxies in production.
* Never log API keys.

---

## Troubleshooting

* **No audio output**: verify TTS credentials and upstream service health.
* **Whisper errors**: check `AUDIO_API_KEY`, `WHISPER_URL`, and upstream service health.
* **`ffmpeg` not found**: install it and ensure it’s on `$PATH` inside your runtime.
* **High latency**: tune VAD (`start_ms`, `end_ms`, `pad_ms`, `max_ms`) and reduce TTS chunk size.
* **WebSocket closes immediately**: missing/invalid token with `API_TOKENS` set.

---

## License

MIT (or your preferred license).
