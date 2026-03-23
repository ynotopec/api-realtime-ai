# External Test Quick Start

This folder contains lightweight clients for quickly validating realtime API behavior from a browser and from Python.

## 1) Prerequisites

- Python 3.10+
- An API key exported as `OPENAI_API_KEY`
- Network access to your realtime endpoint (default uses OpenAI-compatible endpoints)

## 2) Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn openai websockets
```

## 3) Run the local websocket bridge

The browser pages connect to `ws://localhost:8000/ws`, so start the bridge first:

```bash
export OPENAI_API_KEY="<your_key>"
# optional: export OPENAI_API_BASE="https://api.openai.com/v1"
python external-test/proxy.py
```

The bridge will:
- accept browser audio,
- open a realtime session,
- stream translated text/audio back to the browser.

## 4) Open a browser test page

In a separate terminal, serve this folder statically:

```bash
cd external-test
python -m http.server 5173
```

Then open one of these pages in your browser:
- `http://localhost:5173/index.html` (main speech translator demo)
- `http://localhost:5173/echo-cancel.html` (echo-cancel tuning test)
- `http://localhost:5173/half-duplex.html` (half-duplex behavior test)

Click **Start**, allow microphone access, and speak.

## 5) Run the text-only realtime translator script

If you want a fast CLI check without browser audio:

```bash
export OPENAI_API_KEY="<your_key>"
python external-test/translate_text.py --text "Bonjour, comment ça va ?"
```

You should see event logs and a final translated output with latency.

## 6) Common issues

- **`OPENAI_API_KEY not set`**: export the key in the same shell where you run the command.
- **Browser can’t connect to WS**: make sure `proxy.py` is running on port `8000`.
- **No microphone input**: re-check browser permissions for `localhost`.
- **Realtime handshake/auth errors**: verify endpoint URL and key validity.
