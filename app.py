# /home/ailab/api-realtime-ai/app.py
import asyncio
import base64
import json
import logging
import os
import re
import string
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import requests
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

# ────────────────────────────── Config & logging ──────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s – %(message)s')
logger = logging.getLogger(__name__)


def log(message: str, *args: Any, **kwargs: Any) -> None:
    """Adapter retained for backward compatibility with existing log() calls."""
    logger.info(message, *args, **kwargs)


def _summarize_payload(payload: Any, *, limit: int = 200) -> str:
    try:
        if payload is None:
            return 'None'
        if isinstance(payload, (str, bytes)):
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8', errors='replace')
            return payload if len(payload) <= limit else payload[:limit] + '…'
        if isinstance(payload, (int, float, bool)):
            return repr(payload)
        if isinstance(payload, dict):
            j = json.dumps(payload, default=str)
            return j if len(j) <= limit else j[:limit] + '…'
        if isinstance(payload, (list, tuple, set)):
            j = json.dumps(list(payload), default=str)
            return j if len(j) <= limit else j[:limit] + '…'
        r = repr(payload)
        return r[:limit] + ('…' if len(r) > limit else '')
    except Exception as exc:
        return f'<unserializable payload: {exc}>'


class Cfg:
    AUDIO_API_KEY = os.getenv('AUDIO_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', '')
    OPENAI_MODEL = os.getenv('OPENAI_API_MODEL', 'gpt-oss')
    WHISPER_URL = os.getenv('WHISPER_URL', 'https://api-audio2txt.cloud-pi-native.com/v1/audio/transcriptions')
    DIAR_URL = os.getenv('DIAR_URL', 'https://api-diarization.cloud-pi-native.com/upload-audio/')
    DIAR_TOKEN = os.getenv('DIARIZATION_TOKEN')
    TTS_API_KEY = os.getenv('TTS_API_KEY')
    TTS_URL = os.getenv('TTS_API_URL', 'https://api-txt2audio.cloud-pi-native.com/v1/audio/speech')
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '256'))


for var_name in ("AUDIO_API_KEY", "OPENAI_API_KEY"):
    if not getattr(Cfg, var_name):
        raise RuntimeError(f"Missing mandatory env variable : {var_name}")

# Default OpenAI API base if not provided
if not Cfg.OPENAI_API_BASE:
    Cfg.OPENAI_API_BASE = 'https://api.openai.com/v1'

session = requests.Session()
session.mount(
    "http://",
    requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=2),
)
session.mount(
    "https://",
    requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=2),
)


def _preview_payload(options: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Build a lightweight preview of an HTTP payload for safe logging."""
    preview: Dict[str, Any] = {}
    for key in ("json", "data"):
        value = options.get(key)
        if value is not None:
            preview[key] = _summarize_payload(value)
    files = options.get("files")
    if isinstance(files, MutableMapping):
        preview["files"] = list(files.keys())
    return preview


def post(url: str, **kwargs: Any) -> requests.Response:
    """POST wrapper that centralises logging, timeouts and error handling."""
    kwargs.setdefault("timeout", Cfg.REQUEST_TIMEOUT)
    payload_preview = _preview_payload(kwargs)
    log(
        "[IO][HTTP][OUTBOUND] POST %s opts={'timeout': %s} payload=%s",
        url,
        kwargs.get("timeout"),
        payload_preview,
    )
    response = session.post(url, **kwargs)
    log(
        "[IO][HTTP][OUTBOUND][RESPONSE] url=%s status=%s length=%s",
        url,
        response.status_code,
        len(response.content),
    )
    response.raise_for_status()
    return response


# ────────────────────────────── Helpers ──────────────────────────────
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    return s.translate(str.maketrans('', '', string.punctuation))


FILTER = {_norm(s) for s in ("thank you", "thanks", "merci", "ok merci")}
LANG_NAME = {
    "fr": "French",
    "en": "English",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Brazilian Portuguese",
    "ru": "Russian",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
}

REALTIME_SR = 24000
FRAME_MS = 20
BYTES_PER_SAMPLE = 2
SMP_24K = int(REALTIME_SR * FRAME_MS / 1000)
BPC_24K = SMP_24K * BYTES_PER_SAMPLE
SMP_16K = int(16000 * FRAME_MS / 1000)

_RESAMPLE_RATIO_24_TO_16 = np.float32(SMP_24K / SMP_16K)
_RESAMPLE_POS_24_TO_16 = np.arange(SMP_16K, dtype=np.float32) * _RESAMPLE_RATIO_24_TO_16
_RESAMPLE_I0_24_TO_16 = _RESAMPLE_POS_24_TO_16.astype(np.int32)
_RESAMPLE_I1_24_TO_16 = np.minimum(_RESAMPLE_I0_24_TO_16 + 1, SMP_24K - 1)
_RESAMPLE_FRAC_24_TO_16 = _RESAMPLE_POS_24_TO_16 - _RESAMPLE_I0_24_TO_16


def _stream_length(file_obj: Any) -> Optional[int]:
    """Best effort attempt at retrieving the length of an uploaded stream."""
    length = getattr(file_obj, "content_length", None)
    if length is not None:
        return int(length)
    stream = getattr(file_obj, "stream", None)
    if stream is None or not hasattr(stream, "tell") or not hasattr(stream, "seek"):
        return None
    position = stream.tell()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(position)
    return int(size)


def tiny_chunk(file_obj: Any) -> bool:
    """Return True when the upload is smaller than the Whisper minimum."""
    size = _stream_length(file_obj)
    return size is not None and size < 16_000


def call_whisper(file) -> Dict[str, Any]:
    """Send the uploaded audio blob to the Whisper transcription backend."""
    # Some endpoints expect the "model" field to be part of the multipart form-data
    # rather than the request body. Align with the historically working payload
    # structure to maximise compatibility with both the legacy and current
    # backends.
    files = {
        'file': (file.filename, file.stream, 'audio/webm'),
        'model': (None, 'whisper-1'),
    }
    return post(
        Cfg.WHISPER_URL,
        headers={'Authorization': f'Bearer {Cfg.AUDIO_API_KEY}'},
        files=files
    ).json()


def call_diarization(file, target_lang: str) -> Dict[str, Any]:
    if not Cfg.DIAR_TOKEN:
        return {}
    file.stream.seek(0)
    files = {'file': (file.filename, file.stream, 'audio/webm'), 'target_lang': (None, target_lang)}
    try:
        return post(Cfg.DIAR_URL, headers={'Authorization': f'Bearer {Cfg.DIAR_TOKEN}'}, files=files).json()
    except Exception as e:
        log(f'[DIARIZATION ERROR] {e}')
        return {}


@lru_cache(maxsize=Cfg.MAX_CACHE_SIZE)
def translate_text(text: str, lang: str) -> str:
    lang_prompt = LANG_NAME.get(lang, lang)
    data = {
        'model': Cfg.OPENAI_MODEL,
        'messages': [
            {'role': 'system', 'content': 'You are a professional translator. Translate accurately and naturally.'},
            {'role': 'user', 'content': f'Translate the text into {lang_prompt}. Return ONLY the translation.\n\n{text}'}
        ],
        'temperature': 0
    }
    out = post(
        f"{Cfg.OPENAI_API_BASE}/chat/completions",
        headers={'Authorization': f'Bearer {Cfg.OPENAI_API_KEY}'},
        json=data
    ).json()['choices'][0]['message']['content']
    return re.sub(r'\s+', ' ', out).strip()


def build_translations(txt: str, detected: str, primary: str, target: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for lg in {primary, target} - {detected}:
        try:
            out[f'translation_{lg}'] = translate_text(txt, lg)
        except Exception as e:
            log(f'[TRANSLATION ERROR target={lg}] {e}')
            out[f'translation_{lg}'] = txt
    return out


def call_tts_webm(text: str, voice: str, instructions: str) -> bytes:
    payload = {
        'model': 'gpt-4o-mini-tts',
        'input': text,
        'voice': voice,
        'instructions': instructions,
        'response_format': 'opus'
    }
    return post(
        Cfg.TTS_URL,
        headers={'Authorization': f'Bearer {Cfg.TTS_API_KEY}', 'Content-Type': 'application/json'},
        json=payload
    ).content


def call_tts_pcm16le(text: str, voice: str, instructions: str) -> bytes:
    webm = call_tts_webm(text, voice, instructions)
    with tempfile.NamedTemporaryFile(suffix='.webm') as fi, tempfile.NamedTemporaryFile(suffix='.pcm') as fo:
        fi.write(webm)
        fi.flush()
        subprocess.check_output(
            ['ffmpeg', '-y', '-i', fi.name, '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', fo.name],
            stderr=subprocess.DEVNULL
        )
        fo.seek(0)
        return fo.read()


def _resample_24k_to_16k_linear(b24: bytes) -> bytes:
    if not b24:
        return b""
    src = np.frombuffer(b24, dtype=np.int16)
    if not src.size:
        return b""
    src_f = src.astype(np.float32)
    if src.size == SMP_24K:
        base = src_f[_RESAMPLE_I0_24_TO_16]
        diff = src_f[_RESAMPLE_I1_24_TO_16] - base
        out = base + diff * _RESAMPLE_FRAC_24_TO_16
    else:
        # Generic path kept for unexpected frame sizes (e.g. partial buffers).
        limit = max(src.size - 1, 0)
        pos = np.linspace(0.0, limit, SMP_16K, dtype=np.float32)
        i0 = np.floor(pos).astype(np.int32)
        i1 = np.minimum(i0 + 1, limit)
        frac = pos - i0
        base = src_f[i0]
        diff = src_f[i1] - base
        out = base + diff * frac
    return np.clip(out, -32768.0, 32767.0).astype(np.int16, copy=False).tobytes()


def _resample_pcm_ffmpeg(pcm: bytes, sr_in: int, sr_out: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix='.pcm') as fi, tempfile.NamedTemporaryFile(suffix='.pcm') as fo:
        fi.write(pcm)
        fi.flush()
        subprocess.check_output(
            ['ffmpeg', '-y', '-f', 's16le', '-ar', str(sr_in), '-ac', '1', '-i', fi.name,
             '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', str(sr_out), '-ac', '1', fo.name],
            stderr=subprocess.DEVNULL
        )
        fo.seek(0)
        return fo.read()


def _pcm24k_to_webm_for_whisper(pcm: bytes) -> str:
    f_pcm = tempfile.NamedTemporaryFile(suffix='.pcm', delete=False)
    f_webm = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
    try:
        f_pcm.write(pcm)
        f_pcm.flush()
        subprocess.check_output(
            [
                'ffmpeg', '-y',
                '-f', 's16le', '-ar', str(REALTIME_SR), '-ac', '1',
                '-i', f_pcm.name,
                '-c:a', 'libopus', '-b:a', '32k',
                f_webm.name
            ],
            stderr=subprocess.DEVNULL
        )
        return f_webm.name
    finally:
        try:
            f_pcm.close()
        except Exception:
            pass
        try:
            f_webm.close()
        except Exception:
            pass


# ────────────────────────────── FastAPI App ──────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────── WS bridge (/v1/realtime) ──────────────────────
def openai_error(
    message: str,
    error_type: str = 'internal_error',
    param: Optional[str] = None,
    code: Optional[int] = None,
) -> Dict[str, Any]:
    """Return an error payload that mimics OpenAI's Realtime schema."""
    return {
        'type': 'response.error',
        'error': {
            'message': message,
            'type': error_type,
            'param': param,
            'code': code,
        },
    }


def _nid(p='item') -> str:
    return f"{p}_{uuid.uuid4().hex[:12]}"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _default_session_config() -> Dict[str, Any]:
    return {
        "voice": "shimmer",
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {"type": "none"},
        "input_audio_transcription": {"model": "gpt-4o-transcribe"},
    }


@dataclass
class WSConv:
    items: List[Dict[str, Any]] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)
    session: Dict[str, Any] = field(default_factory=_default_session_config)

    def add(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self.items.append(item)
        return item

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        return next((x for x in self.items if x.get("id") == item_id), None)

    def delete(self, item_id: str) -> bool:
        initial_len = len(self.items)
        self.items = [x for x in self.items if x.get("id") != item_id]
        return len(self.items) != initial_len


def _extract_text_from_parts(parts: Iterable[Dict[str, Any]]) -> str:
    buffer: List[str] = []
    for content in parts:
        content_type = (content.get("type") or "").lower()
        if content_type in {"input_text", "text", "output_text"}:
            text = content.get("text")
            if text:
                buffer.append(str(text))
        elif content_type == "input_audio":
            transcript = content.get("transcript")
            if transcript:
                buffer.append(str(transcript))
    return " ".join(segment.strip() for segment in buffer if segment and segment.strip())


def _extract_auth_token_fastapi(websocket: WebSocket) -> str:
    tok = (websocket.headers.get('authorization') or '').strip()
    if tok.lower().startswith('bearer '):
        tok = tok.split(' ', 1)[1].strip()
    if tok:
        return tok
    q = websocket.query_params
    tok = (q.get('token') or q.get('access_token') or '').strip()
    if tok:
        return tok
    proto = websocket.headers.get('sec-websocket-protocol', '')
    for part in proto.split(','):
        part = part.strip()
        prefix = 'openai-insecure-api-key.'
        if part.startswith(prefix):
            return part[len(prefix):].strip()
    return ''


API_TOKENS = {t.strip() for t in (os.getenv('API_TOKENS') or '').split(',') if t.strip()}
_is_tok_ok = lambda t: (not API_TOKENS) or (t and t in API_TOKENS)


def _ws_auth_ok_fastapi(websocket: WebSocket) -> bool:
    return _is_tok_ok(_extract_auth_token_fastapi(websocket))


def _session_payload(session_id: str, session_config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the session identifier into the payload expected by clients."""
    return {'id': session_id, **session_config}


async def _send_ws_json_async(
    ws: WebSocket,
    payload: Dict[str, Any],
    *,
    lock: Optional[asyncio.Lock] = None,
) -> None:
    """Serialize and send JSON data over the websocket with consistent logging."""
    try:
        serialized = json.dumps(payload)
    except Exception as exc:
        log("[IO][WS][SEND][SERDE_ERROR] %s", exc)
        return

    try:
        if lock:
            async with lock:
                log("[IO][WS][SEND] %s", _summarize_payload(payload))
                await ws.send_text(serialized)
        else:
            log("[IO][WS][SEND] %s", _summarize_payload(payload))
            await ws.send_text(serialized)
    except Exception as exc:
        log("[IO][WS][SEND][ERROR] %s", exc)


def _transcribe_24k_pcm(pcm24: bytes) -> str:
    path = _pcm24k_to_webm_for_whisper(pcm24)
    class DF:
        filename = 'audio.webm'
    df = DF()
    df.stream = open(path, 'rb')
    df.content_length = os.path.getsize(path)
    try:
        return (call_whisper(df) or {}).get('text', '').strip()
    finally:
        df.stream.close()
        os.unlink(path)


def _messages_from_items(st: WSConv) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    instr = st.session.get('instructions')
    if instr:
        msgs.append({'role': 'system', 'content': instr})

    for it in st.items:
        if (it.get('type') != 'message') or (it.get('role') not in ('system', 'user', 'assistant')):
            continue
        text = _extract_text_from_parts(it.get('content') or [])
        role = it.get('role')
        if role == 'system' and not text and it.get('content'):
            continue
        if text:
            msgs.append({'role': role, 'content': text})
    return msgs


def _llm_reply_from_history(st: WSConv) -> str:
    try:
        data = {'model': Cfg.OPENAI_MODEL, 'messages': _messages_from_items(st), 'temperature': 0.3}
        r = post(f"{Cfg.OPENAI_API_BASE}/chat/completions",
                 headers={'Authorization': f'Bearer {Cfg.OPENAI_API_KEY}'}, json=data).json()
        return (r['choices'][0]['message']['content'] or '').strip()
    except Exception:
        return ''


def _prune_history(st: WSConv, max_items: int = int(os.getenv('MAX_HISTORY_ITEMS', '50'))) -> None:
    if len(st.items) > max_items:
        st.items = st.items[-max_items:]


@app.websocket('/v1/realtime')
async def realtime_ws_endpoint(websocket: WebSocket):
    if not _ws_auth_ok_fastapi(websocket):
        log('[REALTIME][WS] unauthorized websocket attempt rejected')
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    log('[REALTIME][WS] connection accepted')

    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()
    st = WSConv()

    async def send_async(obj):
        """Async sender for the main event loop."""
        await _send_ws_json_async(websocket, obj, lock=send_lock)

    def send_from_thread(obj):
        """Thread-safe sender for background tasks."""
        future = asyncio.run_coroutine_threadsafe(send_async(obj), loop)

        def log_exception(fut):
            try:
                fut.result(timeout=0.1)  # Non-blocking check for immediate errors
            except Exception as e:
                log(f"[IO][WS][SEND][THREAD_ERROR] {e}")
        future.add_done_callback(log_exception)

    st.session.setdefault('sr', REALTIME_SR)
    st.session.setdefault('turn_detection', {'type': 'none'})

    sess_id = _nid('sess')
    session_payload = _session_payload(sess_id, st.session)
    await send_async({'type': 'session.created', 'session': session_payload})

    current_resp = {'id': None, 'cancelled': False}

    vad_state = {
        'vad': webrtcvad.Vad(int(os.getenv('DEFAULT_VAD_AGGR', '2'))),
        'aggr': int(os.getenv('DEFAULT_VAD_AGGR', '2')),
        'start_ms': 80, 'end_ms': 300, 'pad_ms': 120, 'max_ms': 10_000,
        '_frames': [], '_trig': False, '_start_idx': None, '_v': 0, '_nv': 0,
        'auto_create_response': os.getenv('DEFAULT_VAD_AUTORESP', '1') == '1',
        'interrupt_response': False
    }

    # Core response generator (runs in a thread)
    def _handle_response_create():
        def resp_created(rid): send_from_thread({'type': 'response.created', 'response': {'id': rid}})
        def response_text_delta(rid, item_id, token):
            send_from_thread({"type": "response.output_text.delta", "response_id": rid, "item_id": item_id, "output_index": 0, "content_index": 0, "delta": token})
        def response_text_done(rid, item_id, full_text):
            send_from_thread({"type": "response.output_text.done", "response_id": rid, "item_id": item_id, "output_index": 0, "content_index": 0, "text": full_text})
        def response_audio_delta(rid, item_id, b64chunk):
            send_from_thread({"type": "response.audio.delta", "response_id": rid, "item_id": item_id, "output_index": 0, "content_index": 0, "delta": b64chunk})
        def response_audio_done(rid, item_id):
            send_from_thread({"type": "response.audio.done", "response_id": rid, "item_id": item_id, "output_index": 0, "content_index": 0})
        def response_done(rid, asst_item_id):
            send_from_thread({"type": "response.done", "response": {"id": rid, "output": [{"id": asst_item_id, "type": "message", "role": "assistant"}]}})

        rid = _nid('resp')
        asst_item_id = _nid('item')
        resp_created(rid)
        send_from_thread({"type": "response.output_item.added", "response_id": rid, "output_index": 0, "item": {"id": asst_item_id, "type": "message", "role": "assistant"}})
        current_resp['id'] = rid
        current_resp['cancelled'] = False

        try: ans = _llm_reply_from_history(st)
        except Exception as e:
            send_from_thread(openai_error(f'llm failed: {e}', 'internal_error', None, 500))
            response_done(rid, asst_item_id)
            return

        accum: List[str] = []
        for i, tok in enumerate(ans.split()):
            if current_resp['cancelled']:
                send_from_thread({'type': 'response.cancelled', 'response': {'id': rid}}); break
            piece = tok if i == 0 else " " + tok
            accum.append(piece)
            response_text_delta(rid, asst_item_id, piece)

        if not current_resp['cancelled']: response_text_done(rid, asst_item_id, "".join(accum))
        else: response_done(rid, asst_item_id); return

        voice = st.session.get('voice', 'shimmer')
        tts_instructions = st.session.get('instructions', 'Speak clearly and positively.')
        try:
            if not current_resp['cancelled']:
                pcm16 = call_tts_pcm16le(ans, voice, tts_instructions)
                pcm24 = _resample_pcm_ffmpeg(pcm16, 16000, REALTIME_SR)
                hop = int(REALTIME_SR * 0.06) * 2
                for i in range(0, len(pcm24), hop):
                    if current_resp['cancelled']:
                        send_from_thread({'type': 'response.cancelled', 'response': {'id': rid}}); break
                    response_audio_delta(rid, asst_item_id, _b64(pcm24[i:i + hop]))
                if not current_resp['cancelled']: response_audio_done(rid, asst_item_id)
        except Exception as e: send_from_thread(openai_error(f'tts failed: {e}', 'internal_error', None, 500))

        response_done(rid, asst_item_id)
        st.add({'id': asst_item_id, 'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': ans}]})
        _prune_history(st)

    def _start_response_thread():
        t = threading.Thread(target=_handle_response_create, daemon=True)
        t.start()

    # VAD processing (async, runs in main loop)
    async def _process_vad_buffer():
        frames = vad_state['_frames']
        if not frames: return
        start_fr, end_fr, pad_fr, max_fr = [max(1, int(round(vad_state[k] / FRAME_MS))) for k in ('start_ms', 'end_ms', 'pad_ms', 'max_ms')]
        f = frames[-1]

        if not vad_state['_trig']:
            vad_state['_v'] = vad_state['_v'] + 1 if f['v'] else 0
            if vad_state['_v'] >= start_fr:
                vad_state['_trig'] = True
                vad_state['_start_idx'] = max(0, len(frames) - vad_state['_v'] - pad_fr)
                vad_state['_nv'] = 0
                await send_async({"type": "input_audio_buffer.speech_started"})
        else:
            vad_state['_nv'] = 0 if f['v'] else vad_state['_nv'] + 1
            dur = len(frames) - (vad_state['_start_idx'] or 0)
            if vad_state['_nv'] >= end_fr or dur >= max_fr:
                end = min(len(frames), max((len(frames) - vad_state['_nv']) + pad_fr, (vad_state['_start_idx'] or 0) + 1))
                i0 = vad_state['_start_idx'] or 0; i1 = end
                if i1 <= i0:
                    vad_state['_trig'] = False; vad_state['_start_idx'] = None; vad_state['_v'] = vad_state['_nv'] = 0
                    return
                b = b''.join(fr['b24'] for fr in frames[i0:i1])
                vad_state['_frames'] = frames[i1:]
                vad_state['_trig'] = False; vad_state['_start_idx'] = None; vad_state['_v'] = vad_state['_nv'] = 0
                await send_async({"type": "input_audio_buffer.speech_stopped"})
                try:
                    uid = _nid('item')
                    user_item = {'id': uid, 'type': 'message', 'role': 'user', 'content': [{'type': 'input_audio', 'mime_type': f'audio/pcm;rate={REALTIME_SR}'}]}
                    st.add(user_item); _prune_history(st)
                    await send_async({'type': 'conversation.item.created', 'item': user_item})
                    try: tr = _transcribe_24k_pcm(b) if b else ''
                    except Exception as e: log(f"[REALTIME][WS][VAD][WHISPER] transcription error: {e}"); tr = ''
                    await send_async({"type": "conversation.item.input_audio_transcription.completed", "item_id": uid, "transcript": tr})
                    st.items = [({'id': uid, 'type': 'message', 'role': 'user', 'content': [{'type': 'input_audio', 'transcript': tr}]} if x['id'] == uid else x) for x in st.items]
                    if vad_state.get('auto_create_response'): _start_response_thread()
                except Exception as exc: log(f"[REALTIME][WS][VAD][COMMIT] error: {exc}")

    async def _on_append_feed_vad(chunk_bytes):
        td_local = st.session.get('turn_detection', {}) or {}
        if td_local.get('type') == 'server_vad' and td_local.get('interrupt_response'):
            if current_resp.get('id') and not current_resp.get('cancelled'):
                current_resp['cancelled'] = True
                await send_async({'type': 'response.cancelled', 'response': {'id': current_resp['id']}})
        st.audio.extend(chunk_bytes)
        while len(st.audio) >= BPC_24K:
            f24 = bytes(st.audio[:BPC_24K]); del st.audio[:BPC_24K]
            try: v = vad_state['vad'].is_speech(_resample_24k_to_16k_linear(f24), 16000)
            except Exception as e: log(f"[REALTIME][WS][VAD] vad error: {e}"); v = False
            vad_state['_frames'].append({'b24': f24, 'v': v})
            await _process_vad_buffer()

    # Main message loop
    try:
        while True:
            msg = await websocket.receive_text()
            try: d = json.loads(msg)
            except Exception: await send_async(openai_error('invalid JSON', 'client_error', None, 400)); continue
            t = d.get('type')
            log(f"[REALTIME][WS] received type={t}")

            if t == 'session.update':
                sess = d.get('session') or {}
                for k in ('input_audio_format', 'output_audio_format', 'voice', 'instructions'):
                    if k in sess: st.session[k] = sess[k]
                if 'turn_detection' in sess:
                    td = sess['turn_detection'] or {}
                    st.session['turn_detection'] = td
                    if td.get('type') == 'server_vad':
                        aggr = int(td.get('aggressiveness', vad_state['aggr']))
                        vad_state['aggr'] = aggr
                        try: vad_state['vad'] = webrtcvad.Vad(aggr)
                        except Exception: vad_state['vad'] = webrtcvad.Vad(vad_state['aggr'])
                        for k in ('silence_duration_ms', 'prefix_padding_ms', 'start_ms', 'end_ms', 'pad_ms', 'max_ms'):
                            if k in td: vad_state[k.replace('silence_duration', 'end').replace('prefix_padding', 'pad')] = int(td[k])
                        for k in ('create_response', 'interrupt_response'):
                            if k in td: vad_state[f'auto_{k}' if k == 'create_response' else k] = bool(td[k])
                session_payload = _session_payload(sess_id, st.session)
                await send_async({'type': 'session.updated', 'session': session_payload})
            elif t == 'input_audio_buffer.append':
                try: chunk = base64.b64decode(d.get('audio') or '')
                except Exception: await send_async(openai_error('bad audio base64', 'client_error', None, 400)); continue
                if (st.session.get('turn_detection', {}) or {}).get('type') == 'server_vad':
                    await _on_append_feed_vad(chunk)
                else: st.audio.extend(chunk)
            elif t == 'input_audio_buffer.clear':
                st.audio.clear(); vad_state['_frames'].clear()
                vad_state['_trig'] = False; vad_state['_start_idx'] = None; vad_state['_v'] = vad_state['_nv'] = 0
                await send_async({'type': 'input_audio_buffer.cleared'})
            elif t == 'input_audio_buffer.commit':
                uid = _nid('item')
                user_item = {'id': uid, 'type': 'message', 'role': 'user', 'content': [{'type': 'input_audio', 'mime_type': f'audio/pcm;rate={REALTIME_SR}'}]}
                st.add(user_item); _prune_history(st)
                await send_async({'type': 'conversation.item.created', 'item': user_item})
                pcm = bytes(st.audio); st.audio.clear()
                vad_state['_frames'].clear(); vad_state['_trig'] = False; vad_state['_start_idx'] = None; vad_state['_v'] = vad_state['_nv'] = 0
                await send_async({"type": "input_audio_buffer.committed"})
                try: tr = _transcribe_24k_pcm(pcm) if pcm else ''
                except Exception as e: log(f"[REALTIME][WS][WHISPER] transcription error: {e}"); tr = ''
                await send_async({"type": "conversation.item.input_audio_transcription.completed", "item_id": uid, "transcript": tr})
                st.items = [({'id': uid, 'type': 'message', 'role': 'user', 'content': [{'type': 'input_audio', 'transcript': tr}]} if x['id'] == uid else x) for x in st.items]
                td = st.session.get('turn_detection', {}) or {}
                if (td.get('type') == 'server_vad' and vad_state.get('auto_create_response')) or os.getenv('COMMIT_AUTORESP', '1') == '1':
                    log("[REALTIME][WS][COMMIT] auto_create_response → response.create()")
                    _start_response_thread()
            elif t == 'conversation.item.create':
                it = d.get('item') or {}
                if it.get('type') == 'message' and it.get('role') in ('user', 'system'):
                    norm = [{"type": "input_text", "text": p.get('text', '')} if p.get('type') in ('input_text', 'text') else p for p in it.get('content', [])]
                    it['content'] = norm
                it.setdefault('id', _nid('item')); st.add(it); _prune_history(st)
                await send_async({'type': 'conversation.item.created', 'item': it})
            elif t == 'conversation.item.retrieve':
                item_id = d.get('item_id')
                await send_async({'type': 'conversation.item.retrieved', 'item': st.get(item_id) or {'id': item_id}})
            elif t == 'conversation.item.delete':
                item_id = d.get('item_id')
                ok = st.delete(item_id)
                await send_async({'type': 'conversation.item.deleted', 'item_id': item_id, 'deleted': ok})
            elif t == 'response.create':
                _start_response_thread()
            elif t == 'response.cancel':
                rid = d.get('response', {}).get('id') or d.get('response_id')
                if rid and current_resp.get('id') == rid:
                    current_resp['cancelled'] = True
                    await send_async({'type': 'response.cancelled', 'response': {'id': rid}})
                else:
                    await send_async(openai_error('response id not found', 'client_error', None, 404))
            else:
                await send_async(openai_error(f'unhandled type: {t}', 'client_error', None, 400))
    except WebSocketDisconnect:
        log('[REALTIME][WS] client disconnected')
    except Exception as e:
        log(f'[REALTIME][WS] an error occurred in main loop: {e}', exc_info=True)
    finally:
        log(f'[REALTIME][WS] connection closed for {websocket.client}')


# ────────────────────────────── Entrypoint ──────────────────────────────
# To run this application:
# uvicorn app:app --host 0.0.0.0 --port 8080 --ws websockets
#
# Example using environment variables:
# SERVER_PORT=8080 SERVER_NAME=0.0.0.0 uvicorn app:app --host "$SERVER_NAME" --port "$SERVER_PORT" --ws websockets
if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('SERVER_PORT', 8080))
    host = os.getenv('SERVER_NAME', '0.0.0.0')
    log(f"[BOOT] FastAPI WS Realtime server starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port, ws="websockets")
