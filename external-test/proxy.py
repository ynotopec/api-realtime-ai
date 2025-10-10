import os
import json
import asyncio
import base64
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from openai import AsyncOpenAI  # same client you were using

SAMPLE_RATE = 24000

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Use environment variable for API key
OPENAI_API_KEY = 'sk-OX_zYocmpagMGfLyJhig6g'
#os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set. Please export OPENAI_API_KEY environment variable.")

OPENAI_API_BASE = 'https://api-translate-rt.cloud-pi-native.com/v1'
#os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logging.info("Browser websocket accepted")

    # wait for initial config message from client
    try:
        init_msg = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        config = json.loads(init_msg)
    except Exception:
        # default config if none provided
        config = {"voice": "ballad", "source_lang": "English", "target_lang": "French"}

    voice = config.get("voice", "ballad")
    source_lang = config.get("source_lang", "English")
    target_lang = config.get("target_lang", "French")

    # create OpenAI realtime client and connection
    #client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    logging.info("Connecting to OpenAI Realtime...")
    try:
        async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
            logging.info("OpenAI realtime connected for websocket client")

            # configure session
            await conn.session.update(session={"turn_detection": {"type": "server_vad"}, "voice": voice})

            # system prompt config for translations
            await conn.conversation.item.create(
                item={
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"You translate the user's messages from {source_lang} "
                                f"to {target_lang} and repeat them."
                            ),
                        }
                    ],
                }
            )

            # task: forward audio from browser -> openai
            async def recv_from_browser_and_forward():
                try:
                    while True:
                        msg = await ws.receive_text()
                        data = json.loads(msg)
                        # expect: {type: "input_audio", audio: "<base64-encoded Int16 pcm>"}
                        if data.get("type") == "input_audio":
                            encoded_audio = data.get("audio")
                            if encoded_audio:
                                # Append audio to OpenAI realtime input buffer
                                await conn.input_audio_buffer.append(audio=encoded_audio)
                        elif data.get("type") == "config":
                            # allow runtime re-config if needed (not necessary)
                            pass
                except WebSocketDisconnect:
                    logging.info("Browser disconnected (recv task)")
                    raise
                except Exception as e:
                    logging.exception("recv_from_browser_and_forward error: %s", e)
                    raise

            # task: listen to openai events and send to browser
            async def listen_openai_and_forward():
                try:
                    async for event in conn:               
                        ev_type = getattr(event, "type", None)
                        

                        # Send audio deltas back to browser
                        if ev_type == "response.audio.delta":
                            # event.delta is base64 audio (Int16) from OpenAI
                            # forward to client as {type:'audio', audio: '<base64>'}
                            try:
                                await ws.send_text(json.dumps({"type": "audio", "audio": event.delta}))
                            except Exception:
                                logging.exception("Failed to forward audio to browser")
                        else:
                            # Try to extract any text-like pieces (transcripts or text deltas)
                            delta = getattr(event, "delta", None)
                            
                            try:
                                text_piece = ""
                                if isinstance(delta, str):
                                    text_piece = delta
                                elif isinstance(delta, dict):
                                    if "text" in delta and isinstance(delta["text"], str):
                                        text_piece = delta["text"]
                                    elif "content" in delta:
                                        content = delta["content"]
                                        if isinstance(content, list):
                                            for elem in content:
                                                if isinstance(elem, dict) and "text" in elem:
                                                    text_piece += elem["text"]
                                                else:
                                                    text_piece += str(elem)
                                        else:
                                            text_piece = str(content)
                                else:
                                    text_piece = ""
                                if text_piece:
                                    await ws.send_text(json.dumps({"type": "transcription.delta", "text": text_piece}))
                            except Exception:
                                # fallback: send entire event
                                try:
                                    await ws.send_text(json.dumps({"type": "event", "event_type": ev_type, "delta": str(delta)}))
                                except Exception:
                                    pass

                        # response.done -> notify browser turn complete
                        if ev_type == "response.done":
                            await ws.send_text(json.dumps({"type": "response.done"}))
                except Exception as e:
                    logging.exception("listen_openai_and_forward error: %s", e)
                    raise

            # spawn tasks
            forward_task = asyncio.create_task(recv_from_browser_and_forward())
            listen_task = asyncio.create_task(listen_openai_and_forward())

            # wait until any of the tasks finishes (client disconnect or exception)
            done, pending = await asyncio.wait(
                [forward_task, listen_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for t in pending:
                t.cancel()

    except Exception as e:
        logging.exception("Realtime session error: %s", e)
        try:
            await ws.close()
        except Exception:
            pass

if __name__ == "__main__":
#    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
