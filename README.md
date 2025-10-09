# Realtime Speech Bridge (Flask + Socket.IO)

A lightweight Flask service that bridges **browser audio → ASR (Whisper)** → optional **diarization** → **LLM translation** → **TTS** with both **HTTP** and **Realtime** interfaces.

* **/upload**: file upload → Whisper transcription → optional diarization → on-the-fly translations.
* **/tts-proxy**: secure proxy to an external TTS API.
* **/realtime** (Socket.IO): low-latency streaming with VAD (server-side) → Whisper chunks → TTS stream back to client.
* **/v1/realtime** (WebSocket): OpenAI-style Realtime API bridge compatible with simple event schema.

> Uses `ffmpeg` for audio muxing/resampling, `webrtcvad` for server VAD, and a remote Whisper+TTS backend via HTTP.

---

## Table of contents

* [Architecture](#architecture)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
* [HTTP API](#http-api)
* [Realtime — Socket.IO](#realtime--socketio)
* [Realtime — WebSocket Bridge (/v1/realtime)](#realtime--websocket-bridge-v1realtime)
* [Examples](#examples)
* [Deployment](#deployment)
* [Security notes](#security-notes)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Architecture

```
Browser mic (24k PCM)  ──► Socket.IO / WebSocket
         │                    │
         │                    ▼
         │             Server VAD (webrtcvad)
         │                    │ (chunking)
         │                    ▼
         │               Whisper API (HTTP)
         │                    │
         │         + optional diarization API
         │                    │
         │                    ▼
         │              Transcription text
         │               + translations
         │                    │
         │                    ▼
         │                 TTS API
         │                    │
         ▼                    ▼
     Transcripts        Streamed PCM/Opus back to client
```

---

## Features

* **File uploads** (`/upload`) with automatic **Whisper** transcription and language detection, optional **diarization**.
* **Translation cache** with `lru_cache` to reduce latency/cost.
* **TTS proxy** (`/tts-proxy`) returning audio (WebM/Opus by default) or PCM16 helper internally.
* **Realtime streaming** (`/realtime` namespace):

  * Server-side **VAD** parameters configurable via session update.
  * Emits `transcript_temp` and `transcript_final` as soon as chunks resolve.
  * Streams TTS audio back in small hops for low latency.
* **OpenAI-like Realtime bridge** (`/v1/realtime`):

  * Accepts `input_audio_buffer.append` events with base64 PCM24k.
  * Responds with `response.output_text.delta` and audio frames.
* **Token gate** for realtime endpoints via `API_TOKENS`.

---

## Prerequisites

* Python 3.10+
* **ffmpeg** installed and available on `$PATH` (used for (de)muxing & resampling)
* A reachable **Whisper** API endpoint and **TTS** endpoint

Suggested packages (see your `requirements.txt`):

```
flask
flask-cors
flask-socketio
flask-sock
requests
numpy
webrtcvad
```

---

## Quickstart

### 1) Environment

Create a `.env` or export environment variables (see [Configuration](#configuration)):

```bash
export AUDIO_API_KEY=...          # Whisper backend
export OPENAI_API_KEY=...         # LLM for translations and simple replies
export OPENAI_API_BASE=https://your-openai-compatible-host
export OPENAI_API_MODEL=gpt-oss
export TTS_API_KEY=...
export TTS_API_URL=https://api-txt2audio.cloud-pi-native.com/v1/audio/speech
export DIARIZATION_TOKEN=...      # optional
export API_TOKENS="token1,token2" # optional auth for realtime
```

### 2) Install & run

```bash
pip install -r requirements.txt
python app.py  # or: gunicorn -k eventlet -w 1 app:sio --bind 0.0.0.0:5001
```

> Note: `SocketIO` server is exposed via `sio.run(...)`. For production, prefer Gunicorn + Eventlet/Gevent (see [Deployment](#deployment)).

---

## Configuration

| Variable            | Required                 | Default                                                             | Description                                                                    |
| ------------------- | ------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `AUDIO_API_KEY`     | ✅                        | –                                                                   | API key for Whisper transcription backend.                                     |
| `OPENAI_API_KEY`    | ✅                        | –                                                                   | API key for OpenAI-compatible chat endpoint (translations, simple replies).    |
| `OPENAI_API_BASE`   | ✅ (if not default)       | ``                                                                  | Base URL for OpenAI-compatible API (must expose `/chat/completions`).          |
| `OPENAI_API_MODEL`  | ❌                        | `gpt-oss`                                                           | Model name for translations / simple replies.                                  |
| `WHISPER_URL`       | ❌                        | `https://api-audio2txt.cloud-pi-native.com/v1/audio/transcriptions` | Whisper transcription endpoint.                                                |
| `DIAR_URL`          | ❌                        | `https://api-diarization.cloud-pi-native.com/upload-audio/`         | Diarization endpoint.                                                          |
| `DIARIZATION_TOKEN` | ❌                        | –                                                                   | Bearer token for diarization API.                                              |
| `TTS_API_KEY`       | ❌ (required if TTS used) | –                                                                   | API key for TTS endpoint.                                                      |
| `TTS_API_URL`       | ❌                        | `https://api-txt2audio.cloud-pi-native.com/v1/audio/speech`         | TTS endpoint.                                                                  |
| `REQUEST_TIMEOUT`   | ❌                        | `30`                                                                | HTTP timeout (seconds) for upstream calls.                                     |
| `MAX_CACHE_SIZE`    | ❌                        | `256`                                                               | LRU cache entries for translations.                                            |
| `VAD_AGGR`          | ❌                        | `2`                                                                 | Default VAD aggressiveness (0..3).                                             |
| `API_TOKENS`        | ❌                        | –                                                                   | Comma-separated tokens to allow access to realtime endpoints. Empty = no auth. |
| `SERVER_NAME`       | ❌                        | `0.0.0.0`                                                           | Bind address.                                                                  |
| `SERVER_PORT`       | ❌                        | `5001`                                                              | Bind port.                                                                     |

**Audio formats**

* **Input** realtime: base64 **PCM16** at **24 kHz** (frame size = 20 ms, 960 bytes).
* Whisper upload: `audio/webm` (Opus) is produced internally from PCM using `ffmpeg`.
* TTS: external API returns WebM/Opus; helper converts to **PCM16 16 kHz** where needed.

---

## HTTP API

### `POST /upload`

Upload a single audio file for transcription (+ optional diarization + translations).

**Form fields**

* `file` (required): audio file (e.g. WebM/Opus). Tiny chunks (<16 KB) are ignored.
* `target_lang` (optional, default `fr`)
* `primary_lang` (optional, default `fr`)

**Response**

```json
{
  "detected_lang": "en",
  "transcription": "Hello world",
  "diarization": { /* optional backend JSON */ },
  "translation_fr": "Bonjour le monde",
  "translation_bg": "Здравей свят"
}
```

**cURL**

```bash
curl -F file=@sample.webm \
     -F target_lang=bg -F primary_lang=fr \
     -H "Authorization: Bearer $AUDIO_API_KEY" \
     http://localhost:5001/upload
```

### `POST /tts-proxy`

Proxies a JSON payload directly to the configured TTS provider.

**Request**

```json
{
  "model": "gpt-4o-mini-tts",
  "input": "Bonjour !",
  "voice": "coral",
  "response_format": "opus"
}
```

**Response**

* Content-Type: audio (e.g. `audio/webm`)
* Body: binary audio

---

## Realtime — Socket.IO

* Namespace: **`/realtime`**
* Auth: optional Bearer token in `Authorization` header, `auth` payload, or querystring (`token=` / `access_token=`). Controlled by `API_TOKENS`.

**Server events → client**

* `message` with shapes:

  * `{ "type":"connected", "content": {"engine":"internal", "sr":24000} }`
  * `{ "type":"event", "content": {"type":"turn.start"} }`
  * `{ "type":"transcript_temp", "content":"partial text" }`
  * `{ "type":"transcript_final", "content":"final text" }`
  * `{ "type":"audio", "content":"<base64 pcm24k chunk>" }`
  * `{ "type":"turn_done" }`
  * `{ "type":"error", "content":"..." }`

**Client → server**

* `update_session`: `{ session: { voice?: string, turn_detection?: { type: "server_vad", aggressiveness?: 0..3, start_ms?, end_ms?, pad_ms?, max_ms? } } }`
* `audio`: `{ audio: "<base64 pcm24k>" , commit?: true }`
* `commit`: no payload; forces current segment commit.
* `system_prompt`: (no-op placeholder)

---

## Realtime — WebSocket Bridge (`/v1/realtime`)

* Plain WebSocket endpoint mimicking an OpenAI Realtime flow.
* Auth: optional Bearer token or `?token=` in query (depends on `API_TOKENS`).

**Client → server events**

* `session.update`: `{ "session": { "voice": "shimmer", "modalities":["audio","text"], ... } }`
* `conversation.item.create`: `{ "item": { "id?": "...", "type":"message", "role":"user", "content":[ {"type":"input_text", "text":"..."} ] } }`
* `input_audio_buffer.append`: `{ "audio": "<base64 pcm24k>" }`
* `input_audio_buffer.commit`: no payload → triggers transcription
* `conversation.item.retrieve` / `conversation.item.delete`
* `response.create`: no payload → generates reply + audio

**Server → client events**

* `session.created`, `session.updated`
* `conversation.item.created` / `retrieved` / `deleted`
* `response.output_text.delta`: text token(s)
* `response.audio.delta`: base64 PCM24k chunks
* `response.done`: end of response
* `error`: `{ error: { message: "..." } }`

---

## Examples

### Minimal Socket.IO client (browser)

```html
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script>
const sr = 24000;
const socket = io("http://localhost:5001/realtime", {
  transports: ["websocket"],
  auth: { token: "YOUR_TOKEN_IF_ANY" }
});

socket.on("message", (m) => {
  if (m.type === "audio") {
    // decode base64 PCM24 and play with WebAudio (left as an exercise)
  } else {
    console.log("server:", m);
  }
});

// send 24kHz PCM16 frames (base64) periodically
function sendChunk(b64, commit=false){
  socket.emit("audio", { audio: b64, commit });
}
</script>
```

### WebSocket bridge (Node.js)

```js
import WebSocket from "ws";
const ws = new WebSocket("ws://localhost:5001/v1/realtime", {
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

---

## Deployment

* **Production server**: prefer `gunicorn` with **eventlet** or **gevent** workers for WebSockets.

  ```bash
  pip install gunicorn eventlet
  gunicorn -k eventlet -w 1 app:sio --bind 0.0.0.0:5001
  ```
* **Containers**: ensure `ffmpeg` is present in the image.
* **Scaling**: Socket sessions are in-memory per process; for multi-instance scaling, configure a message queue (e.g., Redis) via Flask-SocketIO if you need cross-instance events.
* **CPU/GPU**: ASR/TTS are upstream services; this app mainly does muxing/resampling (CPU-bound) and I/O.

---

## Security notes

* Gate realtime endpoints with `API_TOKENS`.
* Use HTTPS/secure proxies in production.
* Validate and size-limit uploads (tiny chunks are ignored; add reverse-proxy limits as needed).
* Never log API keys.

---

## Troubleshooting

* **No audio output**: verify TTS credentials and that `/tts-proxy` returns audio Content-Type.
* **Whisper errors**: check `AUDIO_API_KEY`, `WHISPER_URL`, and upstream service health.
* **`ffmpeg` not found**: install it and ensure it’s on `$PATH` inside your runtime.
* **High latency**: tune VAD (`start_ms`, `end_ms`, `pad_ms`, `max_ms`) and reduce TTS chunk size.
* **WebSocket closes immediately**: missing/invalid token with `API_TOKENS` set.

---

## License

MIT (or your preferred license).
