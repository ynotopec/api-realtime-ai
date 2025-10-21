from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

import websockets

REALTIME_WS_URL = os.getenv(
    "OPENAI_REALTIME_URL",
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
)

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user input to English. "
    "Preserve meaning, tone, and formatting. Respond with only the translation."
)


def _headers() -> dict[str, str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {key}", "OpenAI-Beta": "realtime=v1"}
    project = os.getenv("OPENAI_PROJECT")
    if project:
        headers["OpenAI-Project"] = project
    return headers


async def _ws_connect(url: str, headers: dict[str, str], timeout: int):
    try:
        return await websockets.connect(url, additional_headers=headers, open_timeout=timeout)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, open_timeout=timeout)


def _coerce_text(value: Any) -> str:
    """Extract textual content from realtime payload fragments."""

    collected: list[str] = []

    def visit(node: Any) -> None:
        if not node:
            return
        if isinstance(node, str):
            collected.append(node)
        elif isinstance(node, dict):
            # Official protocol nests text in several possible keys.
            for key in (
                "text",
                "delta",
                "value",
                "output_text",
                "content",
            ):
                if key in node:
                    visit(node[key])
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(value)
    return "".join(collected)


async def translate_to_english(text: str, timeout: int = 60) -> tuple[str, float]:
    headers = _headers()
    print("[DEBUG] URL:", REALTIME_WS_URL)
    print(
        "[DEBUG] Headers:",
        {k: ("***" if k == "Authorization" else v) for k, v in headers.items()},
    )
    started = time.time()
    ws = await _ws_connect(REALTIME_WS_URL, headers, timeout)
    try:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"instructions": SYSTEM_PROMPT, "modalities": ["text"]},
                }
            )
        )

        await ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                        "instructions": SYSTEM_PROMPT,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": text},
                                ],
                            }
                        ],
                    },
                }
            )
        )

        chunks: list[str] = []
        final_text: str | None = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            data = json.loads(msg)
            etype = data.get("type")
            print("[EVT]", etype)

            if etype in {
                "response.text.delta",
                "response.output_text.delta",
                "response.delta",
                "response.content_part.delta",
            }:
                piece = _coerce_text(
                    data.get("delta") or data.get("output_text") or data.get("text")
                )
                if piece:
                    chunks.append(piece)
            elif etype in {
                "response.text.done",
                "response.output_text.done",
                "response.content_part.done",
            }:
                text_value = _coerce_text(
                    data.get("text") or data.get("output_text") or data.get("delta")
                )
                if text_value:
                    final_text = text_value
            elif etype == "response.completed":
                if final_text is None and chunks:
                    final_text = "".join(chunks)
                break
            elif etype == "error":
                raise RuntimeError(f"Realtime error: {data}")

        output = (final_text or "").strip()
        latency = time.time() - started
        return output, latency
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="Bonjour, comment ça va aujourd’hui toto ?",
    )
    args = parser.parse_args()
    output, latency = asyncio.run(translate_to_english(args.text))
    print("SOURCE :", args.text)
    print("OUTPUT :", output)
    print(f"LATENCY: {latency:.3f}s")


if __name__ == "__main__":
    main()
