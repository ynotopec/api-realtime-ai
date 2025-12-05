"""Chat completion + feedback endpoints with self-improvement loop."""
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from opentelemetry import trace
from observability import (
    inject_tracing_headers,
    record_feedback_submission,
    record_model_inference,
    record_token_usage,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.getenv("FEEDBACK_DB", "data/feedback.db"))
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    channel: Optional[str] = None

    @validator("content")
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("content cannot be empty")
        return value


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Preferred model name")
    messages: List[ChatMessage]
    stream: bool = False
    channel: str = Field(default="default", description="Group channel identifier")
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = None


class FeedbackIn(BaseModel):
    channel: str = Field(default="default", description="Group channel identifier")
    message: str
    response: Optional[str] = None
    positive: bool
    label: Optional[Literal["positive", "negative", "neutral"]] = Field(
        default=None, description="Normalized user sentiment label"
    )
    score: Optional[float] = Field(
        default=None, description="Optional numeric score or rating"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="ID that ties feedback to the originating request",
        min_length=1,
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Trace identifier to join with distributed traces",
        min_length=1,
    )
    prompt: Optional[str] = None
    model_version: Optional[str] = Field(
        default=None, description="Model or routing decision used for the response"
    )
    routing_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary routing/decision metadata recorded for observability",
    )
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedbackSummary(BaseModel):
    channel: str
    positives: int
    negatives: int
    net: int


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT,
                    positive INTEGER NOT NULL,
                    label TEXT,
                    score REAL,
                    request_id TEXT,
                    trace_id TEXT,
                    prompt TEXT,
                    model_version TEXT,
                    routing_metadata TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            self._ensure_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure newly added columns exist for older databases."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
        migrations = {
            "label": "ALTER TABLE feedback ADD COLUMN label TEXT",
            "score": "ALTER TABLE feedback ADD COLUMN score REAL",
            "request_id": "ALTER TABLE feedback ADD COLUMN request_id TEXT",
            "trace_id": "ALTER TABLE feedback ADD COLUMN trace_id TEXT",
            "prompt": "ALTER TABLE feedback ADD COLUMN prompt TEXT",
            "model_version": "ALTER TABLE feedback ADD COLUMN model_version TEXT",
            "routing_metadata": "ALTER TABLE feedback ADD COLUMN routing_metadata TEXT",
        }
        for column, ddl in migrations.items():
            if column not in existing:
                conn.execute(ddl)

    def insert(self, fb: FeedbackIn) -> int:
        payload = fb.model_dump()
        label = payload.get("label") or ("positive" if payload["positive"] else "negative")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback (
                    channel,
                    message,
                    response,
                    positive,
                    label,
                    score,
                    request_id,
                    trace_id,
                    prompt,
                    model_version,
                    routing_metadata,
                    tags,
                    metadata,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["channel"],
                    payload["message"],
                    payload.get("response"),
                    1 if payload["positive"] else 0,
                    label,
                    payload.get("score"),
                    payload.get("request_id"),
                    payload.get("trace_id"),
                    payload.get("prompt"),
                    payload.get("model_version"),
                    json.dumps(payload.get("routing_metadata")),
                    json.dumps(payload.get("tags")),
                    json.dumps(payload.get("metadata")),
                    time.time(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def summary(self, channel: str) -> FeedbackSummary:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT SUM(positive) AS pos, COUNT(*)-SUM(positive) AS neg FROM feedback WHERE channel = ?",
                (channel,),
            ).fetchone()
            pos = int(row["pos"] or 0)
            neg = int(row["neg"] or 0)
        return FeedbackSummary(channel=channel, positives=pos, negatives=neg, net=pos - neg)

    def recent(self, channel: Optional[str], limit: int = 25) -> List[Dict[str, Any]]:
        query = "SELECT * FROM feedback"
        params: List[Any] = []
        if channel:
            query += " WHERE channel = ?"
            params.append(channel)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]


class GroupConversationStore:
    def __init__(self) -> None:
        self._messages: Dict[str, List[Dict[str, str]]] = {}
        self._lock = threading.Lock()

    def history(self, channel: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._messages.get(channel, []))

    def append(self, channel: str, messages: Iterable[Dict[str, str]]) -> None:
        with self._lock:
            existing = self._messages.setdefault(channel, [])
            existing.extend(messages)


class SelfImprovementLoop:
    def __init__(self, store: FeedbackStore) -> None:
        self.store = store
        self.base_prompt = os.getenv(
            "CHAT_SYSTEM_PROMPT",
            "You are a concise assistant that answers with actionable bullet points and upbeat tone.",
        )
        self.default_model = os.getenv("CHAT_DEFAULT_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"))
        self.fallback_model = os.getenv("CHAT_FALLBACK_MODEL")

    def plan(self, channel: str) -> Dict[str, Any]:
        summary = self.store.summary(channel)
        adjustments: List[str] = []
        if summary.negatives > summary.positives:
            adjustments.append(
                "Users reported confusion. Prefer numbered steps, avoid speculation, and keep answers under 120 words."
            )
        else:
            adjustments.append("Users responded positively. Preserve the supportive, to-the-point tone.")

        if summary.negatives - summary.positives >= 3 and self.fallback_model:
            model = self.fallback_model
            adjustments.append("Routing to fallback model because recent feedback trends are negative.")
        else:
            model = self.default_model

        prompt = f"{self.base_prompt} {' '.join(adjustments)}"
        return {"prompt": prompt, "model": model, "summary": summary, "adjustments": adjustments}


router = APIRouter(prefix="/v1")
feedback_store = FeedbackStore(DEFAULT_DB_PATH)
conversations = GroupConversationStore()
improver = SelfImprovementLoop(feedback_store)

API_TIMEOUT = float(os.getenv("CHAT_REQUEST_TIMEOUT", 30))
OPENAI_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _headers() -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required for chat completions")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


def _render_messages(channel: str, prompt: str, incoming: List[ChatMessage]) -> List[Dict[str, str]]:
    history = conversations.history(channel)
    prepared: List[Dict[str, str]] = []
    if prompt:
        prepared.append({"role": "system", "content": prompt})
    prepared.extend(history)
    for msg in incoming:
        prepared.append({"role": msg.role, "content": msg.content})
    return prepared


def _extract_delta_text(chunk: Dict[str, Any]) -> str:
    try:
        delta = chunk["choices"][0]["delta"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([part.get("text", "") for part in content if isinstance(part, dict)])
    return ""


async def _stream_response(payload: Dict[str, Any], channel: str, prompt: str) -> StreamingResponse:
    url = f"{OPENAI_BASE}/chat/completions"
    final_text: List[str] = []
    headers = _headers()
    inject_tracing_headers(headers)

    async def event_stream():
        nonlocal final_text
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise HTTPException(status_code=resp.status_code, detail=body.decode())
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                    else:
                        data = line
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        parsed = json.loads(data)
                        final_text.append(_extract_delta_text(parsed))
                        yield f"data: {json.dumps(parsed)}\n\n"
                    except json.JSONDecodeError:
                        continue
        # persist conversation after stream completes
        full_text = "".join(final_text).strip()
        if full_text:
            conversations.append(channel, [{"role": "assistant", "content": full_text}])
        duration = time.perf_counter() - start
        record_model_inference(duration, payload.get("model"))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _invoke_non_stream(payload: Dict[str, Any], channel: str) -> Dict[str, Any]:
    url = f"{OPENAI_BASE}/chat/completions"
    headers = _headers()
    inject_tracing_headers(headers)
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        duration = time.perf_counter() - start
        record_model_inference(duration, payload.get("model"))
        data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""
    if content:
        conversations.append(channel, [{"role": "assistant", "content": content}])
    usage = data.get("usage") if isinstance(data, dict) else None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            record_token_usage(prompt_tokens, "prompt", payload.get("model"))
        if completion_tokens is not None:
            record_token_usage(completion_tokens, "completion", payload.get("model"))
    return data


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    plan = improver.plan(request.channel)
    messages = _render_messages(request.channel, plan["prompt"], request.messages)
    payload: Dict[str, Any] = {
        "model": request.model or plan["model"],
        "messages": messages,
        "stream": request.stream,
        "temperature": request.temperature,
    }
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens
    if request.metadata:
        payload["metadata"] = request.metadata

    # record the user turn for the channel history before calling the model
    conversations.append(
        request.channel,
        [{"role": msg.role, "content": msg.content} for msg in request.messages if msg.role != "system"],
    )

    if request.stream:
        return await _stream_response(payload, request.channel, plan["prompt"])
    return await _invoke_non_stream(payload, request.channel)


@router.post("/feedback", response_model=Dict[str, Any])
async def submit_feedback(fb: FeedbackIn, request: Request):
    start = time.perf_counter()
    derived_label = fb.label or ("positive" if fb.positive else "negative")
    span = trace.get_current_span()
    span_ctx = span.get_span_context()
    trace_id = fb.trace_id or (
        format(span_ctx.trace_id, "032x") if span_ctx and span_ctx.is_valid else None
    )
    request_id = fb.request_id or request.headers.get("x-request-id")

    if not request_id and not trace_id:
        raise HTTPException(
            status_code=400,
            detail="feedback must include a request_id or be submitted within a traced request",
        )

    enriched_fb = fb.model_copy(
        update={"label": derived_label, "trace_id": trace_id, "request_id": request_id}
    )
    feedback_id = await run_in_threadpool(feedback_store.insert, enriched_fb)

    log_fields = {
        "event": "feedback.submitted",
        "feedback_id": feedback_id,
        "label": derived_label,
        "channel": fb.channel,
        "trace_id": trace_id,
        "request_id": request_id,
        "model_version": fb.model_version,
    }
    logger.info("feedback.submitted", extra=log_fields)

    if span.is_recording():
        span.add_event(
            "feedback.submitted",
            attributes={
                "feedback.id": feedback_id,
                "feedback.label": derived_label,
                "feedback.score": fb.score if fb.score is not None else 0,
                "feedback.channel": fb.channel,
                "feedback.request_id": request_id or "",
                "feedback.trace_id": trace_id or "",
                "feedback.model_version": fb.model_version or "",
            },
        )

    duration = time.perf_counter() - start
    record_feedback_submission(derived_label, duration)
    return {"id": feedback_id, "channel": fb.channel, "label": derived_label}


@router.get("/feedback/summary", response_model=FeedbackSummary)
async def feedback_summary(channel: str = "default"):
    return await run_in_threadpool(feedback_store.summary, channel)


@router.get("/feedback/recent", response_model=List[Dict[str, Any]])
async def feedback_recent(channel: Optional[str] = None, limit: int = 25):
    return await run_in_threadpool(feedback_store.recent, channel, limit)
