# /home/ailab/api-translate-rt/app.py
import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, MutableMapping, Optional
from urllib.parse import parse_qs

import numpy as np
import requests
import webrtcvad
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO  # utilisé pour l’entrypoint HTTP

# ────────────────────────────── Config & logging ──────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s – %(message)s')
logger = logging.getLogger(__name__)


def log(message: str, *args: Any, **kwargs: Any) -> None:
    """Adapter retained for backward compatibility with existing log() calls."""

    logger.info(message, *args, **kwargs)

def _summarize_payload(payload: Any, *, limit: int = 200) -> str:
    try:
        if payload is None: return 'None'
        if isinstance(payload, (str, bytes)):
            if isinstance(payload, bytes): payload = payload.decode('utf-8', errors='replace')
            return payload if len(payload) <= limit else payload[:limit] + '…'
        if isinstance(payload, (int, float, bool)): return repr(payload)
        if isinstance(payload, dict):
            j = json.dumps(payload, default=str)
            return j if len(j) <= limit else j[:limit] + '…'
        if isinstance(payload, (list, tuple, set)):
            j = json.dumps(list(payload), default=str)
            return j if len(j) <= limit else j[:limit] + '…'
        return repr(payload)[:limit] + ('…' if len(repr(payload)) > limit else '')
    except Exception as exc:
        return f'<unserializable payload: {exc}>'

class Cfg:
    AUDIO_API_KEY   = os.getenv('AUDIO_API_KEY')
    OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', '')
    OPENAI_MODEL    = os.getenv('OPENAI_API_MODEL', 'gpt-oss')
    WHISPER_URL     = os.getenv('WHISPER_URL', 'https://api-audio2txt.cloud-pi-native.com/v1/audio/transcriptions')
    DIAR_URL        = os.getenv('DIAR_URL', 'https://api-diarization.cloud-pi-native.com/upload-audio/')
    DIAR_TOKEN      = os.getenv('DIARIZATION_TOKEN')
    TTS_API_KEY     = os.getenv('TTS_API_KEY')
    TTS_URL         = os.getenv('TTS_API_URL', 'https://api-txt2audio.cloud-pi-native.com/v1/audio/speech')
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_CACHE_SIZE  = int(os.getenv('MAX_CACHE_SIZE', '256'))

for var_name in ("AUDIO_API_KEY", "OPENAI_API_KEY"):
    if not getattr(Cfg, var_name):
        raise RuntimeError(f"Missing mandatory env variable : {var_name}")

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
FILTER = {s.lower() for s in ("thank you.",)}
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
    files = {'file': (file.filename, file.stream, 'audio/webm'), 'model': (None, 'whisper-1')}
    return post(Cfg.WHISPER_URL, headers={'Authorization': f'Bearer {Cfg.AUDIO_API_KEY}'}, files=files).json()

def call_diarization(file, target_lang: str) -> Dict[str, Any]:
    if not Cfg.DIAR_TOKEN: return {}
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
            {'role':'system','content':'You are a professional translator. Translate accurately and naturally.'},
            {'role':'user','content':f'Translate the text into {lang_prompt}. Return ONLY the translation.\n\n{text}'}
        ],
        'temperature': 0
    }
    out = post(f"{Cfg.OPENAI_API_BASE}/chat/completions", headers={'Authorization': f'Bearer {Cfg.OPENAI_API_KEY}'}, json=data).json()['choices'][0]['message']['content']
    return re.sub(r'\s+', ' ', out).strip()

def build_translations(txt: str, detected: str, primary: str, target: str) -> Dict[str, str]:
    out = {}
    for lg in {primary, target} - {detected}:
        try:
            out[f'translation_{lg}'] = translate_text(txt, lg)
        except Exception as e:
            log(f'[TRANSLATION ERROR target={lg}] {e}')
            out[f'translation_{lg}'] = txt
    return out

def call_tts_webm(text: str, voice: str, instructions: str) -> bytes:
    payload = {
        'model':'gpt-4o-mini-tts',
        'input': text,
        'voice': voice,
        'instructions': instructions,
        'response_format':'opus'
    }
    return post(Cfg.TTS_URL, headers={'Authorization': f'Bearer {Cfg.TTS_API_KEY}','Content-Type':'application/json'}, json=payload).content

def call_tts_pcm16le(text: str, voice: str, instructions: str) -> bytes:
    webm = call_tts_webm(text, voice, instructions)
    with tempfile.NamedTemporaryFile(suffix='.webm') as fi, tempfile.NamedTemporaryFile(suffix='.pcm') as fo:
        fi.write(webm); fi.flush()
        subprocess.check_output(['ffmpeg','-y','-i',fi.name,'-f','s16le','-acodec','pcm_s16le','-ar','16000','-ac','1',fo.name], stderr=subprocess.DEVNULL)
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
        fi.write(pcm); fi.flush()
        subprocess.check_output(['ffmpeg','-y','-f','s16le','-ar',str(sr_in),'-ac','1','-i',fi.name,
                                 '-f','s16le','-acodec','pcm_s16le','-ar',str(sr_out),'-ac','1',fo.name],
                                stderr=subprocess.DEVNULL)
        fo.seek(0)
        return fo.read()

def _pcm24k_to_webm_for_whisper(pcm: bytes) -> str:
    f_pcm = tempfile.NamedTemporaryFile(suffix='.pcm', delete=False)
    f_webm = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
    try:
        f_pcm.write(pcm); f_pcm.flush()
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
        try: f_pcm.close()
        except Exception: pass
        try: f_webm.close()
        except Exception: pass

# ────────────────────────────── Flask + Socket.IO ──────────────────────────────
app = Flask(__name__); CORS(app)
sio = SocketIO(app, cors_allowed_origins='*')

@app.route('/tts-proxy', methods=['POST'])
def tts_proxy():
    payload = request.get_json(force=True)
    r = post(Cfg.TTS_URL, headers={'Authorization': f'Bearer {Cfg.TTS_API_KEY}','Content-Type':'application/json'}, json=payload)
    return r.content, r.status_code, {'Content-Type': r.headers.get('Content-Type','audio/webm')}

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f: return jsonify({'error':'No file provided'}), 400
    if tiny_chunk(f): return jsonify({'text':''})
    target = request.form.get('target_lang','fr'); primary = request.form.get('primary_lang','fr')
    diar = call_diarization(f, target)
    f.stream.seek(0)
    try: w = call_whisper(f); text = (w or {}).get('text','').strip()
    except Exception as e: log(f'[WHISPER ERROR] {e}'); text=''
    if not text or text.lower() in FILTER: return jsonify({'text':'','diarization':diar})
    try: from langdetect import detect; detected = detect(text)
    except Exception: detected = ''
    res = {'detected_lang': detected, 'transcription': text, 'diarization': diar}
    res.update(build_translations(text, detected, primary, target))
    return jsonify(res)

# ────────────────────────────── WS bridge (/v1/realtime) ──────────────────────
try:
    from flask_sock import Sock
    sock = Sock(app); HAS_WS=True
except Exception:
    sock=None; HAS_WS=False

def _nid(p='item'): return f"{p}_{uuid.uuid4().hex[:12]}"
def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _default_session_config() -> Dict[str, Any]:
    return {
        "voice": "shimmer",
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",  # encodage int16 ; le rate est précisé via mime_type
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

def _extract_auth_token(env:Dict[str,Any])->str:
    tok=(env.get('HTTP_AUTHORIZATION') or '').strip()
    if tok.lower().startswith('bearer '): tok=tok.split(' ',1)[1].strip()
    if tok: return tok
    q=parse_qs(env.get('QUERY_STRING') or ''); tok=(q.get('token') or q.get('access_token') or [''])[0]
    if tok: return tok
    proto=env.get('HTTP_SEC_WEBSOCKET_PROTOCOL','')
    for part in proto.split(','):
        part=part.strip()
        prefix='openai-insecure-api-key.'
        if part.startswith(prefix): return part[len(prefix):].strip()
    return ''

API_TOKENS={t.strip() for t in (os.getenv('API_TOKENS') or '').split(',') if t.strip()}
_is_tok_ok=lambda t:(not API_TOKENS) or (t and t in API_TOKENS)
def _ws_auth_ok(env): return _is_tok_ok(_extract_auth_token(env))

def _transcribe_24k_pcm(pcm24: bytes) -> str:
    path=_pcm24k_to_webm_for_whisper(pcm24)
    class DF: filename='audio.webm'
    df=DF(); df.stream=open(path,'rb'); import os as _os; df.content_length=_os.path.getsize(path)
    try:
        return (call_whisper(df) or {}).get('text','').strip()
    finally:
        df.stream.close(); os.unlink(path)


def _messages_from_items(st: WSConv) -> list:
    """
    Construit l'historique complet (system, user, assistant) pour le chat.completions.
    Agrège input_text / output_text / transcript dans l'ordre d'apparition.
    Ignore les messages vides pour éviter les répétitions inutiles.
    """
    msgs = []
    instr = st.session.get('instructions')
    if instr:
        msgs.append({'role': 'system', 'content': instr})

    for it in st.items:
        if (it.get('type') != 'message') or (it.get('role') not in ('system', 'user', 'assistant')):
            continue

        text = _extract_text_from_parts(it.get('content') or [])

        # Pour les rares messages 'system' déjà présents dans st.items
        role = it.get('role')
        if role == 'system' and not text and it.get('content'):
            # garde au moins un system vide si besoin, sinon saute
            continue
        if text:
            msgs.append({'role': role, 'content': text})
    return msgs


def _llm_reply_from_history(st: WSConv) -> str:
    try:
        data={'model': Cfg.OPENAI_MODEL, 'messages': _messages_from_items(st), 'temperature': 0.3}
        r=post(f"{Cfg.OPENAI_API_BASE}/chat/completions",
               headers={'Authorization': f'Bearer {Cfg.OPENAI_API_KEY}'}, json=data).json()
        return (r['choices'][0]['message']['content'] or '').strip()
    except Exception:
        return ''

# --- WS handler with VAD + spec-like events ---
if HAS_WS:
    @sock.route('/v1/realtime')
    def realtime_ws(ws):
        def send(obj):
            try:
                log(f"[IO][WS][SEND] {_summarize_payload(obj)}"); ws.send(json.dumps(obj))
            except Exception as exc:
                log(f"[IO][WS][SEND][ERROR] {exc}")

        def openai_error(message, etype='internal_error', param=None, code=None):
            return {'type': 'response.error', 'error': {'message': message, 'type': etype, 'param': param, 'code': code}}

        if not _ws_auth_ok(ws.environ):
            log('[REALTIME][WS] unauthorized websocket attempt rejected')
            send(openai_error('unauthorized', 'auth_error', None, 401))
            return

        log('[REALTIME][WS] connection accepted')
        st = WSConv()
        st.session.setdefault('sr', REALTIME_SR)
        st.session.setdefault('turn_detection', {'type': 'none'})

        # Session ID aplati dans session.created / updated
        sess_id = _nid('sess')
        session_payload = {"id": sess_id, **st.session}
        send({'type': 'session.created', 'session': session_payload})

        current_resp = {'id': None, 'cancelled': False}

        vad_state = {
            'vad': webrtcvad.Vad(int(os.getenv('DEFAULT_VAD_AGGR', '2'))),
            'aggr': int(os.getenv('DEFAULT_VAD_AGGR', '2')),
            'start_ms': 80, 'end_ms': 300, 'pad_ms': 120, 'max_ms': 10_000,
            '_frames': [], '_trig': False, '_start_idx': None, '_v': 0, '_nv': 0,
            'auto_create_response': os.getenv('DEFAULT_VAD_AUTORESP', '1') == '1',
            'interrupt_response': False
        }

        # Response lifecycle helpers (spec-like)
        def resp_created(rid): send({'type': 'response.created', 'response': {'id': rid}})

        def response_text_delta(rid, item_id, token):
            send({"type":"response.output_text.delta","response_id":rid,"item_id":item_id,
                  "output_index":0,"content_index":0,"delta":token})

        def response_text_done(rid, item_id, full_text):
            send({"type":"response.output_text.done","response_id":rid,"item_id":item_id,
                  "output_index":0,"content_index":0,"text":full_text})

        def response_audio_delta(rid, item_id, b64chunk):
            send({"type":"response.audio.delta","response_id":rid,"item_id":item_id,
                  "output_index":0,"content_index":0,"delta":b64chunk})

        def response_audio_done(rid, item_id):
            send({"type":"response.audio.done","response_id":rid,"item_id":item_id,
                  "output_index":0,"content_index":0})

        def response_done(rid, asst_item_id):
            send({"type":"response.done","response":{"id":rid,"output":[{"id":asst_item_id,"type":"message","role":"assistant"}]}})

        # VAD processing
        def _process_vad_buffer():
            frames = vad_state['_frames']
            if not frames: return
            start_fr = max(1, int(round(vad_state['start_ms']/FRAME_MS)))
            end_fr   = max(1, int(round(vad_state['end_ms']/FRAME_MS)))
            pad_fr   = max(1, int(round(vad_state['pad_ms']/FRAME_MS)))
            max_fr   = max(1, int(round(vad_state['max_ms']/FRAME_MS)))
            f = frames[-1]

            if not vad_state['_trig']:
                vad_state['_v'] = vad_state['_v'] + 1 if f['v'] else 0
                if vad_state['_v'] >= start_fr:
                    vad_state['_trig'] = True
                    vad_state['_start_idx'] = max(0, len(frames) - vad_state['_v'] - pad_fr)
                    vad_state['_nv'] = 0
                    send({"type":"input_audio_buffer.speech_started"})
            else:
                vad_state['_nv'] = 0 if f['v'] else vad_state['_nv'] + 1
                dur = len(frames) - (vad_state['_start_idx'] or 0)
                if vad_state['_nv'] >= end_fr or dur >= max_fr:
                    end = min(len(frames), len(frames) - vad_state['_nv'] + pad_fr)
                    i0 = vad_state['_start_idx'] or 0
                    i1 = end
                    if i1 <= i0:
                        vad_state['_trig']=False; vad_state['_start_idx']=None; vad_state['_v']=vad_state['_nv']=0
                        return
                    b = b''.join(fr['b24'] for fr in frames[i0:i1])
                    vad_state['_frames'] = frames[i1:]
                    vad_state['_trig']=False; vad_state['_start_idx']=None; vad_state['_v']=vad_state['_nv']=0
                    send({"type":"input_audio_buffer.speech_stopped"})

                    try:
                        uid = _nid('item')
                        user_item = {'id': uid, 'type': 'message', 'role': 'user',
                                     'content': [{'type': 'input_audio', 'mime_type': f'audio/pcm;rate={REALTIME_SR}'}]}
                        st.add(user_item)
                        send({'type': 'conversation.item.created', 'item': user_item})

                        try:
                            tr = _transcribe_24k_pcm(b) if b else ''
                        except Exception as e:
                            log(f"[REALTIME][WS][VAD][WHISPER] transcription error: {e}"); tr = ''

                        send({"type":"conversation.item.input_audio_transcription.completed","item_id":uid,"transcript":tr})

                        # stocker transcript côté serveur
                        st.items = [({'id': uid, 'type':'message','role':'user',
                                      'content':[{'type':'input_audio','transcript':tr}]}
                                     if x['id']==uid else x) for x in st.items]

                        if vad_state.get('auto_create_response'):
                            _handle_response_create()
                    except Exception as exc:
                        log(f"[REALTIME][WS][VAD][COMMIT] error: {exc}")
                    return

        def _on_append_feed_vad(chunk_bytes):
            st.audio.extend(chunk_bytes)
            while len(st.audio) >= BPC_24K:
                f24 = bytes(st.audio[:BPC_24K]); del st.audio[:BPC_24K]
                try:
                    # Pour plus de robustesse VAD, remplacer par _resample_pcm_ffmpeg(f24, 24000, 16000)
                    v = vad_state['vad'].is_speech(_resample_24k_to_16k_linear(f24), 16000)
                except Exception as e:
                    log(f"[REALTIME][WS][VAD] vad error: {e}"); v = False
                vad_state['_frames'].append({'b24': f24, 'v': v})
                _process_vad_buffer()

        # Core response generator
        def _handle_response_create():
            rid = _nid('resp'); asst_item_id = _nid('item')
            resp_created(rid)
            send({"type":"response.output_item.added","response_id":rid,"output_index":0,
                  "item":{"id":asst_item_id,"type":"message","role":"assistant"}})
            current_resp['id']=rid; current_resp['cancelled']=False

            # Génération texte depuis l'historique (system+user + transcript)
            try:
                ans = _llm_reply_from_history(st)
            except Exception as e:
                send(openai_error(f'llm failed: {e}','internal_error',None,500))
                response_done(rid, asst_item_id); return

            # Stream des deltas texte
            accum=[]
            for i, tok in enumerate(ans.split()):
                if current_resp['cancelled']:
                    send({'type':'response.cancelled','response':{'id':rid}}); break
                piece = tok if i==0 else " "+tok
                accum.append(piece); response_text_delta(rid, asst_item_id, piece)

            if not current_resp['cancelled']:
                response_text_done(rid, asst_item_id, "".join(accum))
            else:
                response_done(rid, asst_item_id); return

            # TTS: respecter voice + instructions de session
            voice = st.session.get('voice', 'shimmer')
            tts_instructions = st.session.get('instructions', 'Speak clearly and positively.')

            try:
                if not current_resp['cancelled']:
                    pcm16 = call_tts_pcm16le(ans, voice, tts_instructions)  # 16k mono
                    pcm24 = _resample_pcm_ffmpeg(pcm16, 16000, REALTIME_SR)  # upsample 24k
                    hop = int(REALTIME_SR * 0.06) * 2  # ~60 ms
                    for i in range(0, len(pcm24), hop):
                        if current_resp['cancelled']:
                            send({'type':'response.cancelled','response':{'id':rid}}); break
                        response_audio_delta(rid, asst_item_id, _b64(pcm24[i:i+hop]))
                    if not current_resp['cancelled']:
                        response_audio_done(rid, asst_item_id)
            except Exception as e:
                send(openai_error(f'tts failed: {e}','internal_error',None,500))

            response_done(rid, asst_item_id)
            st.add({'id': asst_item_id, 'type': 'message', 'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': ans}]})

        # Main loop
        while True:
            msg = ws.receive()
            if msg is None:
                log('[REALTIME][WS] client closed websocket'); break
            try:
                d = json.loads(msg)
            except Exception:
                send(openai_error('invalid JSON','client_error',None,400)); continue

            t = d.get('type'); log(f"[REALTIME][WS] received type={t}")

            if t == 'session.update':
                sess = d.get('session') or {}
                for k in ('input_audio_format','output_audio_format','voice'):
                    if k in sess: st.session[k] = sess[k]
                if 'instructions' in sess: st.session['instructions'] = sess['instructions']
                if 'turn_detection' in sess:
                    td = sess['turn_detection'] or {}
                    st.session['turn_detection'] = td
                    if td.get('type') == 'server_vad':
                        aggr = int(td.get('aggressiveness', vad_state['aggr']))
                        vad_state['aggr'] = aggr
                        try: vad_state['vad'] = webrtcvad.Vad(aggr)
                        except Exception: vad_state['vad'] = webrtcvad.Vad(vad_state['aggr'])
                        # mappings
                        if 'silence_duration_ms' in td: vad_state['end_ms'] = int(td['silence_duration_ms'])
                        if 'prefix_padding_ms' in td: vad_state['pad_ms'] = int(td['prefix_padding_ms'])
                        for k in ('start_ms','end_ms','pad_ms','max_ms'):
                            if k in td: vad_state[k] = int(td[k])
                        if 'create_response' in td:
                            vad_state['auto_create_response'] = bool(td['create_response'])
                        if 'interrupt_response' in td:
                            vad_state['interrupt_response'] = bool(td['interrupt_response'])
                # session.updated aplati (inclut l'id)
                session_payload = {"id": sess_id, **st.session}
                send({'type':'session.updated','session': session_payload})
                continue

            if t == 'input_audio_buffer.append':
                try:
                    chunk_b64 = d.get('audio') or ''
                    chunk = base64.b64decode(chunk_b64) if chunk_b64 else b''
                except Exception:
                    send(openai_error('bad audio base64','client_error',None,400)); continue
                td = st.session.get('turn_detection', {}) or {}
                if td.get('type') == 'server_vad': _on_append_feed_vad(chunk)
                else: st.audio.extend(chunk)
                # pas d’ACK selon la spec
                continue

            if t == 'input_audio_buffer.clear':
                st.audio.clear()
                vad_state['_frames'].clear(); vad_state['_trig']=False
                vad_state['_start_idx']=None; vad_state['_v']=vad_state['_nv']=0
                send({'type':'input_audio_buffer.cleared'})
                continue

            if t == 'input_audio_buffer.commit':
                uid = _nid('item')
                user_item = {'id': uid, 'type': 'message', 'role': 'user',
                             'content':[{'type':'input_audio','mime_type':f'audio/pcm;rate={REALTIME_SR}'}]}
                st.add(user_item)
                send({'type':'conversation.item.created','item':user_item})

                pcm = bytes(st.audio); st.audio.clear()
                vad_state['_frames'].clear(); vad_state['_trig']=False
                vad_state['_start_idx']=None; vad_state['_v']=vad_state['_nv']=0

                # ACK pratique (optionnel)
                send({"type":"input_audio_buffer.committed"})

                try: tr = _transcribe_24k_pcm(pcm) if pcm else ''
                except Exception as e: log(f"[REALTIME][WS][WHISPER] transcription error: {e}"); tr=''

                send({"type":"conversation.item.input_audio_transcription.completed","item_id":uid,"transcript":tr})
                # stocke transcript côté serveur
                st.items = [({'id': uid, 'type':'message','role':'user',
                              'content':[{'type':'input_audio','transcript':tr}]}
                             if x['id']==uid else x) for x in st.items]

                # Alignement OpenAI: auto réponse aussi après commit
                td = st.session.get('turn_detection', {}) or {}
                if (td.get('type') == 'server_vad' and vad_state.get('auto_create_response')) \
                   or os.getenv('COMMIT_AUTORESP', '1') == '1':
                    log("[REALTIME][WS][COMMIT] auto_create_response → response.create()")
                    _handle_response_create()
                continue

            if t == 'conversation.item.create':
                it = d.get('item') or {}
                if it.get('type') == 'message' and it.get('role') in ('user','system'):
                    content = it.get('content') or []
                    norm=[]
                    for part in content:
                        typ = part.get('type')
                        if typ in ('input_text', 'text'):
                            norm.append({"type":"input_text","text": part.get('text','')})
                        elif typ == 'input_audio':
                            norm.append(part)  # laisser tel quel (mime_type / transcript éventuel)
                        else:
                            pass
                    it['content']=norm
                    it.setdefault('id', _nid('item'))
                else:
                    it.setdefault('id', _nid('item'))
                st.add(it)
                send({'type':'conversation.item.created','item':it})
                continue

            if t == 'conversation.item.retrieve':
                item_id = d.get('item_id')
                send({'type':'conversation.item.retrieved','item': st.get(item_id) or {'id': item_id}})
                continue

            if t == 'conversation.item.delete':
                ok = st.delete(d.get('item_id'))
                send({'type':'conversation.item.deleted','item_id': d.get('item_id'),'deleted': ok})
                continue

            if t == 'response.create':
                _handle_response_create(); continue

            if t == 'response.cancel':
                rid = d.get('response', {}).get('id') or d.get('response_id') or None
                if rid and current_resp.get('id') == rid:
                    current_resp['cancelled']=True
                    send({'type':'response.cancelled','response':{'id':rid}})
                else:
                    send(openai_error('response id not found','client_error',None,404))
                continue

            send(openai_error(f'unhandled type: {t}','client_error',None,400))

# ────────────────────────────── Entrypoint ──────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('SERVER_PORT', 8080))
    host = os.getenv('SERVER_NAME', '0.0.0.0')
    log(f"[BOOT] Flask+SocketIO WS Realtime server listening on {host}:{port}")
    sio.run(app, host=host, port=port, debug=False)
