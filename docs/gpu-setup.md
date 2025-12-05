# GPU-ready FastAPI setup (Python 3.11)

Follow these steps to spin up the service inside an isolated virtualenv with GPU-capable dependencies.

1. **Create & activate a Python 3.11 virtualenv**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. **Install core requirements plus GPU runtime**
   ```bash
   pip install -r requirements.txt \
       --extra-index-url https://download.pytorch.org/whl/cu121
   ```
   *`torch` in `requirements.txt` pulls a CUDA-enabled wheel when the extra index is provided; CPU wheels install otherwise.*

3. **Export runtime configuration**
   ```bash
   export OPENAI_API_KEY=sk-...              # required
   export OPENAI_API_MODEL=gpt-4o-mini       # optional, defaults to gpt-4o-mini
   export CHAT_FALLBACK_MODEL=gpt-4o         # optional, used when feedback is negative
   export FEEDBACK_DB=data/feedback.db       # optional custom path
   ```

4. **Run the FastAPI service**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8080 --ws websockets
   ```
   The server now exposes:
   * `/v1/realtime` (existing WebSocket bridge)
   * `/v1/chat/completions` (OpenAI-compatible, streaming supported)
   * `/v1/feedback`, `/v1/feedback/summary`, `/v1/feedback/recent`

5. **Exercise the chat endpoint with streaming**
   ```bash
   curl -N -X POST http://localhost:8080/v1/chat/completions \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4o-mini",
       "channel": "team-alpha",
       "stream": true,
       "messages": [
         {"role": "user", "content": "Give me three steps to deploy a FastAPI service"}
       ]
     }'
   ```
   Streaming responses mirror OpenAI's `data: ...` server-sent events and end with `data: [DONE]`.

6. **Log feedback to drive self-improvement**
   ```bash
   curl -X POST http://localhost:8080/v1/feedback \
     -H "Content-Type: application/json" \
     -d '{
       "channel": "team-alpha",
       "message": "Give me three steps to deploy a FastAPI service",
       "response": "1) Create venv...",
       "positive": true,
       "tags": ["clarity", "deployment"]
     }'
   ```
   Subsequent requests for the same `channel` auto-adjust prompts/model routing based on aggregated feedback.
