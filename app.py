import os, json, base64, tempfile, subprocess, uuid, re, logging
from functools import lru_cache
from urllib.parse import parse_qs
from typing import Dict, Any

import requests, numpy as np, webrtcvad
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ────────────────────────────── Config & logging ──────────────────────────────
log = logging.getLogger(__name__).info
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s – %(message)s')


def _summarize_payload(payload: Any, *, limit: int = 200) -> str:
    """Return a compact string representation of payload objects for logging."""
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
        return repr(payload)[:limit] + ('…' if len(repr(payload)) > limit else '')
    except Exception as exc:  # pragma: no cover - defensive logging helper
        return f'<unserializable payload: {exc}>'

class Cfg:
    AUDIO_API_KEY   = os.getenv('AUDIO_API_KEY')
    OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', '')
    OPENAI_MODEL    = os.getenv('OPENAI_API_MODEL', 'gpt-oss')
#qwen3-coder')
#ai-chat')

#    WHISPER_URL     = 'https://api-audio2txt.c0.cloud-pi-native.com/v1/audio/transcriptions'
    WHISPER_URL     = 'https://api-audio2txt.cloud-pi-native.com/v1/audio/transcriptions'
    DIAR_URL        = 'https://api-diarization.cloud-pi-native.com/upload-audio/'
    DIAR_TOKEN      = os.getenv('DIARIZATION_TOKEN')

    TTS_API_KEY     = os.getenv('TTS_API_KEY')
    TTS_URL         = os.getenv('TTS_API_URL', 'https://api-txt2audio.cloud-pi-native.com/v1/audio/speech')

    REQUEST_TIMEOUT = 30
    MAX_CACHE_SIZE  = 256

for v in ('AUDIO_API_KEY', 'OPENAI_API_KEY'):
    if not getattr(Cfg, v):
        raise RuntimeError(f'Missing mandatory env variable : {v}')

session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=2))
session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=2))

def post(url: str, **kw) -> requests.Response:
    kw.setdefault('timeout', Cfg.REQUEST_TIMEOUT)
    payload_preview = {}
    for key in ('json', 'data'):
        if key in kw and kw[key] is not None:
            payload_preview[key] = _summarize_payload(kw[key])
    if 'files' in kw:
        payload_preview['files'] = list(kw['files'].keys())
    log(f"[IO][HTTP][OUTBOUND] POST {url} opts={{'timeout': {kw.get('timeout')}}} payload={payload_preview}")
    r = session.post(url, **kw)
    log(f"[IO][HTTP][OUTBOUND][RESPONSE] url={url} status={r.status_code} length={len(r.content)}")
    r.raise_for_status(); return r

# ────────────────────────────── Helpers ──────────────────────────────
FILTER = {s.lower() for s in ('thank you.',)}
LANG_NAME = {'fr':'French','en':'English','ro':'Romanian','bg':'Bulgarian','es':'Spanish','de':'German','it':'Italian','pt':'Brazilian Portuguese','ru':'Russian','zh-cn':'Simplified Chinese','zh-tw':'Traditional Chinese'}

REALTIME_SR = 24000
FRAME_MS = 20
BYTES_PER_SAMPLE = 2
SMP_24K = int(REALTIME_SR * FRAME_MS / 1000)        # 480
BPC_24K = SMP_24K * BYTES_PER_SAMPLE                # 960
SMP_16K = int(16000 * FRAME_MS / 1000)              # 320
BPC_16K = SMP_16K * BYTES_PER_SAMPLE                # 640

_b64 = lambda b: base64.b64encode(b).decode()

def tiny_chunk(file) -> bool:
    if getattr(file, 'content_length', None):
        return file.content_length < 16_000
    p = file.stream.tell(); file.stream.seek(0, os.SEEK_END)
    s = file.stream.tell(); file.stream.seek(p); return s < 16_000

def call_whisper(file) -> Dict[str, Any]:
    files = {'file': (file.filename, file.stream, 'audio/webm'), 'model': (None, 'whisper-1')}
    return post(Cfg.WHISPER_URL, headers={'Authorization': f'Bearer {Cfg.AUDIO_API_KEY}'}, files=files).json()

def call_diarization(file, target_lang: str) -> Dict[str, Any]:
    if not Cfg.DIAR_TOKEN: return {}
    file.stream.seek(0)
    files = {'file': (file.filename, file.stream, 'audio/webm'), 'target_lang': (None, target_lang)}
    try:    return post(Cfg.DIAR_URL, headers={'Authorization': f'Bearer {Cfg.DIAR_TOKEN}'}, files=files).json()
    except Exception as e:
        log(f'[DIARIZATION ERROR] {e}'); return {}

@lru_cache(maxsize=Cfg.MAX_CACHE_SIZE)
def translate_text(text: str, lang: str) -> str:
    lang_prompt = LANG_NAME.get(lang, lang)
    data = {
        'model': Cfg.OPENAI_MODEL,
        'messages': [
            {'role':'system','content':'You are a professional translator. Translate accurately and naturally.'},
            {'role':'user','content':f'Translate the text into {lang_prompt}. Return ONLY the translation on a single line (no line breaks).\n\n{text}'}
        ],
        'temperature': 0
    }
    out = post(f"{Cfg.OPENAI_API_BASE}/chat/completions", headers={'Authorization': f'Bearer {Cfg.OPENAI_API_KEY}'}, json=data).json()['choices'][0]['message']['content']
    return re.sub(r'\s+', ' ', out).strip()

def build_translations(txt: str, detected: str, primary: str, target: str) -> Dict[str, str]:
    out = {}
    for lg in {primary, target} - {detected}:
        try:    out[f'translation_{lg}'] = translate_text(txt, lg)
        except Exception as e:
            log(f'[TRANSLATION ERROR target={lg}] {e}'); out[f'translation_{lg}'] = txt
    return out

# TTS (returns WebM/Opus) + PCM16 16k helper

def call_tts(text: str) -> bytes:
    if not Cfg.TTS_API_KEY: raise RuntimeError('TTS_API_KEY not configured')
    payload = {'model':'gpt-4o-mini-tts','input':text,'voice':'coral','instructions':'Speak in a cheerful and positive tone.','response_format':'opus'}
    return post(Cfg.TTS_URL, headers={'Authorization': f'Bearer {Cfg.TTS_API_KEY}','Content-Type':'application/json'}, json=payload).content

def call_tts_pcm16le(text: str) -> bytes:
    webm = call_tts(text)
    with tempfile.NamedTemporaryFile(suffix='.webm') as fi, tempfile.NamedTemporaryFile(suffix='.pcm') as fo:
        fi.write(webm); fi.flush()
        subprocess.check_output(['ffmpeg','-y','-i',fi.name,'-f','s16le','-acodec','pcm_s16le','-ar','16000','-ac','1',fo.name], stderr=subprocess.DEVNULL)
        fo.seek(0); return fo.read()

# Resampling

def _resample_pcm_ffmpeg(pcm: bytes, sr_in: int, sr_out: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix='.pcm') as fi, tempfile.NamedTemporaryFile(suffix='.pcm') as fo:
        fi.write(pcm); fi.flush()
        subprocess.check_output(['ffmpeg','-y','-f','s16le','-ar',str(sr_in),'-ac','1','-i',fi.name,'-f','s16le','-acodec','pcm_s16le','-ar',str(sr_out),'-ac','1',fo.name], stderr=subprocess.DEVNULL)
        fo.seek(0); return fo.read()

def _pcm24k_to_webm_for_whisper(pcm: bytes) -> str:
    f_pcm = tempfile.NamedTemporaryFile(suffix='.pcm', delete=False)
    f_webm= tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
    try:
        f_pcm.write(pcm); f_pcm.flush()
        subprocess.check_output(['ffmpeg','-y','-f','s16le','-ar',str(REALTIME_SR),'-ac','1','-i',f_pcm.name,'-c:a','libopus','-b:a','32k',f_webm.name], stderr=subprocess.DEVNULL)
        return f_webm.name
    finally:
        try: f_pcm.close()
        except: pass

# ────────────────────────────── Flask + Socket.IO ──────────────────────────────
app = Flask(__name__); CORS(app)
sio = SocketIO(app, cors_allowed_origins='*')


@app.before_request
def _log_request_in():
    info = {
        'method': request.method,
        'path': request.path,
        'remote_addr': request.remote_addr,
        'args': request.args.to_dict(flat=False),
        'content_type': request.content_type,
    }
    if request.files:
        info['files'] = {
            name: {'filename': f.filename, 'size': getattr(f, 'content_length', None)}
            for name, f in request.files.items()
        }
    if request.is_json:
        info['json'] = _summarize_payload(request.get_json(silent=True))
    elif request.form:
        info['form'] = {k: request.form.getlist(k) for k in request.form.keys()}
    else:
        data = request.get_data(cache=True)
        if data:
            info['data'] = _summarize_payload(data)
    log(f"[IO][HTTP][REQUEST] {info}")


@app.after_request
def _log_response_out(response):
    try:
        body = response.get_data()
        preview = '<binary>'
        if body:
            if response.mimetype and 'json' in (response.mimetype or ''):
                try:
                    preview = _summarize_payload(json.loads(body.decode('utf-8', errors='replace')))
                except Exception:
                    preview = _summarize_payload(body)
            else:
                preview = _summarize_payload(body)
        log(
            f"[IO][HTTP][RESPONSE] status={response.status_code} path={request.path} "
            f"content_type={response.mimetype} length={len(body)} body={preview}"
        )
    except Exception as exc:  # pragma: no cover - defensive
        log(f"[IO][HTTP][RESPONSE][ERROR] {exc}")
    return response


_socketio_emit_original = sio.emit


def _log_socketio_emit(event, data=None, *args, **kwargs):
    namespace = kwargs.get('namespace')
    if namespace is None and len(args) >= 1:
        namespace = args[0]
    target = kwargs.get('to') or kwargs.get('room') or kwargs.get('sid')
    log(
        f"[IO][SOCKETIO][EMIT] event={event} namespace={namespace or '/'} "
        f"target={target} payload={_summarize_payload(data)}"
    )
    return _socketio_emit_original(event, data, *args, **kwargs)


sio.emit = _log_socketio_emit


_flask_emit_original = emit


def _log_flask_emit(event, data=None, *args, **kwargs):
    namespace = args[0] if args else kwargs.get('namespace')
    target = kwargs.get('to') or kwargs.get('room') or kwargs.get('sid')
    log(
        f"[IO][SOCKETIO][FLASK_EMIT] event={event} namespace={namespace or '/'} "
        f"target={target} payload={_summarize_payload(data)}"
    )
    return _flask_emit_original(event, data, *args, **kwargs)


emit = _log_flask_emit

try:
    from flask_sock import Sock
    sock = Sock(app); HAS_WS = True
except Exception:
    sock = None; HAS_WS = False

@app.route('/tts-proxy', methods=['POST'])
def tts_proxy():
    if not Cfg.TTS_API_KEY: return jsonify({'error':'TTS_API_KEY not configured'}), 500
    try:
        payload = request.get_json(force=True)
        r = post(Cfg.TTS_URL, headers={'Authorization': f'Bearer {Cfg.TTS_API_KEY}','Content-Type':'application/json'}, json=payload)
        return r.content, r.status_code, {'Content-Type': r.headers.get('Content-Type','audio/webm')}
    except Exception as e:
        log(f'[TTS PROXY ERROR] {e}'); return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or f.filename == '': return jsonify({'error':'No file provided'}), 400
    if tiny_chunk(f): return jsonify({'text':''})

    target = request.form.get('target_lang','fr'); primary = request.form.get('primary_lang','fr')
    diar = call_diarization(f, target)

    f.stream.seek(0)
    try:    w = call_whisper(f); text = (w or {}).get('text','').strip()
    except Exception as e:
        log(f'[WHISPER ERROR] {e}'); text = ''

    if not text or text.lower() in FILTER: return jsonify({'text':'','diarization':diar})

    try:    from langdetect import detect; detected = detect(text)
    except Exception: detected = ''

    res = {'detected_lang': detected, 'transcription': text, 'diarization': diar}
    res.update(build_translations(text, detected, primary, target))
    log(f'DEBUG detected={detected} target={target} primary={primary} keys={list(res.keys())}')
    return jsonify(res)

# ────────────────────────────── Realtime internals (VAD + Whisper + TTS) ──────────────────────────────
VAD_AGGR = int(os.getenv('VAD_AGGR','2'))

def _resample_24k_to_16k_linear(b24: bytes) -> bytes:
    x = np.frombuffer(b24, dtype=np.int16).astype(np.float32)
    out = np.empty(SMP_16K, dtype=np.float32); step = 1.5; pos = 0.0
    for i in range(SMP_16K):
        i0 = int(pos); frac = pos - i0
        s0 = x[i0] if i0 < x.size else 0.0; s1 = x[i0+1] if (i0+1) < x.size else s0
        out[i] = s0 + (s1 - s0) * frac; pos += step
    return np.clip(out, -32768.0, 32767.0).astype(np.int16).tobytes()

_ms2fr = lambda ms: max(1, int(round(ms / FRAME_MS)))

class RTSess:
    def __init__(self, sid: str):
        self.sid = sid; self.voice = 'echo'; self.server_vad = True
        self._vad = webrtcvad.Vad(VAD_AGGR)
        self._aggr, self._start, self._end, self._pad, self._max = VAD_AGGR, 80, 300, 120, 10_000
        self._left = bytearray(); self._frames = []; self._trig = False
        self._start_idx = None; self._v = 0; self._nv = 0

    def session_update(self, session: dict):
        td = (session or {}).get('turn_detection', {})
        self.server_vad = (td.get('type') == 'server_vad'); self.voice = (session or {}).get('voice', self.voice)
        if 'aggressiveness' in td: self._aggr = int(td['aggressiveness']); self._vad = webrtcvad.Vad(self._aggr)
        for k,attr in [('start_ms','_start'),('end_ms','_end'),('pad_ms','_pad'),('max_ms','_max')]:
            if k in td: setattr(self, attr, int(td[k]))
        log(f"[REALTIME] session update server_vad={self.server_vad} voice={self.voice} vad_aggr={self._aggr}")

    def append_audio(self, b64: str):
        if not b64: return
        self._left.extend(base64.b64decode(b64))
        while len(self._left) >= BPC_24K:
            f24 = bytes(self._left[:BPC_24K]); del self._left[:BPC_24K]
            v = self._vad.is_speech(_resample_24k_to_16k_linear(f24), 16000)
            self._frames.append({'b24': f24, 'v': v})
            if self.server_vad: self._step()

    def _step(self):
        if not self._frames: return
        f = self._frames[-1]; st, ed, pad, mx = map(_ms2fr, (self._start, self._end, self._pad, self._max))
        if not self._trig:
            self._v = self._v + 1 if f['v'] else 0
            if self._v >= st: self._trig = True; self._start_idx = max(0, len(self._frames) - self._v - pad); self._nv = 0
        else:
            self._nv = 0 if f['v'] else self._nv + 1
            dur = len(self._frames) - (self._start_idx or 0)
            if self._nv >= ed or dur >= mx:
                end = min(len(self._frames), len(self._frames) - self._nv + pad)
                self._commit(self._start_idx, end, reason='vad'); self._trig = False; self._start_idx = None; self._v = self._nv = 0

    def commit(self):
        if self._trig and self._start_idx is not None:
            end = max(self._start_idx + 1, len(self._frames) - _ms2fr(100))
            self._commit(self._start_idx, end, reason='manual-flush'); self._trig = False; self._start_idx = None
        elif self._frames:
            self._commit(0, min(len(self._frames), _ms2fr(2000)), reason='manual')

    def _commit(self, i0: int, i1: int, *, reason: str='vad'):
        if i1 <= i0: return
        b = b''.join(f['b24'] for f in self._frames[i0:i1]); self._frames = self._frames[i1:]
        log(f"[REALTIME] commit reason={reason} frames={i1-i0} bytes={len(b)}")
        sio.emit('message', {'type':'event','content':{'type':'turn.start'}}, namespace='/realtime', to=self.sid)
        # Whisper
        p = _pcm24k_to_webm_for_whisper(b)
        try:
            class F: filename='audio.webm';
            def __init__(self, p): self.stream=open(p,'rb'); self.content_length=os.path.getsize(p)
            import os
            self = self
        except: pass
        class DF: filename='audio.webm'
        df = DF(); df.stream=open(p,'rb'); import os; df.content_length=os.path.getsize(p)
        try:
            try:    text = (call_whisper(df) or {}).get('text','').strip()
            except Exception as e:
                log(f'[REALTIME][WHISPER ERROR] {e}'); text = ''
        finally:
            df.stream.close();
            try: os.unlink(p)
            except Exception: pass
        # transcripts
        if text:
            log(f"[REALTIME] transcript len={len(text)} text={text!r}")
            sio.emit('message', {'type':'transcript_temp','content':text}, namespace='/realtime', to=self.sid)
        else:
            log('[REALTIME] transcription empty or unavailable')
        sio.emit('message', {'type':'transcript_final','content':text}, namespace='/realtime', to=self.sid)
        # TTS
        if text:
            try:
                pcm16 = call_tts_pcm16le(text); pcm24 = _resample_pcm_ffmpeg(pcm16,16000,REALTIME_SR)
                hop = int(REALTIME_SR*0.1)*2
                for i in range(0, len(pcm24), hop):
                    sio.emit('message', {'type':'audio','content': _b64(pcm24[i:i+hop])}, namespace='/realtime', to=self.sid)
            except Exception as e:
                log(f'[REALTIME][TTS ERROR] {e}')
        sio.emit('message', {'type':'turn_done'}, namespace='/realtime', to=self.sid)

_RT = {}

# ────────────────────────────── Socket.IO namespace /realtime ──────────────────────────────
API_TOKENS = {t.strip() for t in (os.getenv('API_TOKENS') or '').split(',') if t.strip()}

_is_tok_ok = lambda t: (not API_TOKENS) or (t and t in API_TOKENS)

@sio.on('connect', namespace='/realtime')
def rt_connect(auth=None):
    log(f"[REALTIME] connect auth={auth} headers_auth={'yes' if request.headers.get('Authorization') else 'no'}")
    tok = None
    if isinstance(auth, dict):
        tok = auth.get('token') or auth.get('access_token') or auth.get('Authorization') or auth.get('authorization')
    if not tok:
        h = request.headers.get('Authorization','')
        if h.lower().startswith('bearer '): tok = h.split(' ',1)[1].strip()
    if not tok:
        tok = request.args.get('token') or request.args.get('access_token')
    if not tok:
        tok = _extract_auth_token(request.environ)
    if API_TOKENS and not _is_tok_ok((tok or '').strip()):
        raise ConnectionRefusedError('unauthorized')

    _RT[request.sid] = RTSess(request.sid)
    emit('message', {'type':'connected','content':{'engine':'internal','sr':REALTIME_SR}}, namespace='/realtime')

@sio.on('disconnect', namespace='/realtime')
def rt_disconnect():
    _RT.pop(request.sid, None); log(f"[REALTIME] Client disconnected (sid={request.sid})")

@sio.on('update_session', namespace='/realtime')
def rt_update_session(data):
    s = _RT.get(request.sid);
    if not s: emit('message', {'type':'error','content':'No realtime session'}, namespace='/realtime'); return
    log(f"[REALTIME] update_session payload={data}")
    s.session_update((data or {}).get('session', {}))

@sio.on('system_prompt', namespace='/realtime')
def rt_system_prompt(data):
    s = _RT.get(request.sid)
    if not s: emit('message', {'type':'error','content':'No realtime session'}, namespace='/realtime'); return
    # kept for compatibility (no-op storage)

@sio.on('audio', namespace='/realtime')
def rt_audio(data):
    s = _RT.get(request.sid)
    if not s: emit('message', {'type':'error','content':'No realtime session'}, namespace='/realtime'); return
    b64 = (data or {}).get('audio'); size = len(base64.b64decode(b64)) if b64 else 0
    log(f'[REALTIME] audio chunk received: {size} bytes')
    if not b64: emit('message', {'type':'error','content':'Missing audio'}, namespace='/realtime'); return
    try:
        s.append_audio(b64)
        if (data or {}).get('commit'):
            log('[REALTIME] client requested commit flag on audio payload')
            s.commit()
    except Exception as e:
        log(f'[REALTIME][AUDIO ERROR] {e}'); emit('message', {'type':'error','content':str(e)}, namespace='/realtime')

@sio.on('commit', namespace='/realtime')
def rt_commit():
    s = _RT.get(request.sid)
    if not s: emit('message', {'type':'error','content':'No realtime session'}, namespace='/realtime'); return
    log('[REALTIME] manual commit requested via event')
    s.commit()

# ────────────────────────────── WS bridge (/v1/realtime) ──────────────────────────────

def _nid(p='item'): return f"{p}_{uuid.uuid4().hex[:12]}"

class WSConv:
    def __init__(self):
        self.items = []; self.audio = bytearray()
        self.session = {"voice":"shimmer","modalities":["audio","text"],"input_audio_format":"pcm16","output_audio_format":"pcm16","input_audio_transcription":{"model":"gpt-4o-transcribe"}}
    def add(self, it): self.items.append(it); return it
    def get(self, i): return next((x for x in self.items if x.get('id')==i), None)
    def delete(self, i):
        n=len(self.items); self.items=[x for x in self.items if x.get('id')!=i]; return len(self.items)!=n

def _ws_send(ws, obj):
    try:
        log(f"[IO][WS][SEND] {_summarize_payload(obj)}")
        ws.send(json.dumps(obj))
    except Exception as exc:
        log(f"[IO][WS][SEND][ERROR] {exc}")

def _extract_auth_token(env: Dict[str, Any]) -> str:
    tok = (env.get('HTTP_AUTHORIZATION') or '').strip()
    if tok.lower().startswith('bearer '):
        tok = tok.split(' ', 1)[1].strip()
    if tok:
        return tok

    q = parse_qs(env.get('QUERY_STRING') or '')
    tok = (q.get('token') or q.get('access_token') or [''])[0]
    if tok:
        return tok

    proto = env.get('HTTP_SEC_WEBSOCKET_PROTOCOL', '')
    for part in proto.split(','):
        part = part.strip()
        if not part:
            continue
        prefix = 'openai-insecure-api-key.'
        if part.startswith(prefix):
            return part[len(prefix):].strip()
    return ''


def _ws_auth_ok(env):
    if not API_TOKENS:
        return True
    return _is_tok_ok(_extract_auth_token(env))

def _transcribe_24k_pcm(pcm24: bytes) -> str:
    path = _pcm24k_to_webm_for_whisper(pcm24)
    class DF: filename='audio.webm'
    df = DF(); df.stream = open(path,'rb'); import os; df.content_length=os.path.getsize(path)
    try:
        try: return (call_whisper(df) or {}).get('text','').strip()
        finally: df.stream.close()
    finally:
        try: os.unlink(path)
        except Exception: pass

def _simple_llm_reply(t: str) -> str:
    try:
        data = {'model': Cfg.OPENAI_MODEL,'messages':[{'role':'system','content':'You are a concise, helpful assistant.'},{'role':'user','content': t or 'Reply with a short acknowledgement.'}], 'temperature':0.3}
        r = post(f"{Cfg.OPENAI_API_BASE}/chat/completions", headers={'Authorization': f'Bearer {Cfg.OPENAI_API_KEY}'}, json=data).json()
        return r['choices'][0]['message']['content'].strip()
    except Exception:
        return t or ''

def _stream_pcm24(ws, pcm24: bytes, hop_ms: int=60):
    hop = int(REALTIME_SR*(hop_ms/1000.0))*2
    for i in range(0,len(pcm24),hop):
        _ws_send(ws, {'type':'response.audio.delta','audio': _b64(pcm24[i:i+hop])})

if HAS_WS:
    @sock.route('/v1/realtime')
    def realtime_ws(ws):
        if not _ws_auth_ok(ws.environ):
            log('[REALTIME][WS] unauthorized websocket attempt rejected')
            _ws_send(ws, {'type':'error','error':{'message':'unauthorized'}})
            return

        log('[REALTIME][WS] connection accepted')
        st = WSConv(); _ws_send(ws, {'type':'session.created','session':{'id':_nid('sess'),'sr':REALTIME_SR}})
        while True:
            msg = ws.receive()
            if msg is None:
                log('[REALTIME][WS] client closed websocket')
                break
            try:
                d = json.loads(msg)
            except Exception:
                log(f"[REALTIME][WS] invalid JSON payload: {msg!r}")
                _ws_send(ws, {'type':'error','error':{'message':'invalid JSON'}}); continue
            t = d.get('type')
            log(f"[REALTIME][WS] received type={t}")
            if t == 'session.update': st.session.update(d.get('session') or {}); _ws_send(ws, {'type':'session.updated','session': st.session}); continue
            if t == 'input_audio_buffer.append':
                try:
                    chunk = d.get('audio') or ''
                    st.audio.extend(base64.b64decode(chunk))
                    log(f"[REALTIME][WS] buffered audio chunk bytes={len(chunk) * 3 // 4 if chunk else 0}")
                except Exception: _ws_send(ws, {'type':'error','error':{'message':'bad audio base64'}})
                continue
            if t == 'input_audio_buffer.commit':
                uid = _nid('item'); ph = {'id':uid,'type':'message','role':'user','content':[{'type':'input_audio','mime_type':'audio/pcm;rate=24000'}]}
                st.add(ph); _ws_send(ws, {'type':'conversation.item.created','item': ph})
                pcm = bytes(st.audio); st.audio.clear(); tr = _transcribe_24k_pcm(pcm) if pcm else ''
                log(f"[REALTIME][WS] committed buffered audio bytes={len(pcm)} transcript_len={len(tr)}")
                it = {'id':uid,'type':'message','role':'user','content':[{'type':'input_audio','transcript': tr}]}
                st.items = [it if x['id']==uid else x for x in st.items]
                _ws_send(ws, {'type':'conversation.item.retrieved','item': it}); continue
            if t == 'conversation.item.create': it=d.get('item') or {}; it.setdefault('id', _nid('item')); st.add(it); _ws_send(ws, {'type':'conversation.item.created','item': it}); continue
            if t == 'conversation.item.retrieve': _ws_send(ws, {'type':'conversation.item.retrieved','item': st.get(d.get('item_id')) or {'id': d.get('item_id')}}); continue
            if t == 'conversation.item.delete': ok = st.delete(d.get('item_id')); _ws_send(ws, {'type':'conversation.item.deleted','item_id': d.get('item_id'), 'deleted': ok}); continue
            if t == 'response.create':
                last = next((x for x in reversed(st.items) if x.get('role')=='user'), {})
                c = (last.get('content') or [{}])[0]; last_txt = c.get('transcript') or c.get('text') or ''
                ans = _simple_llm_reply(last_txt)
                log(f"[REALTIME][WS] generating response for transcript_len={len(last_txt)} reply_len={len(ans)}")
                _ws_send(ws, {'type':'response.output_text.delta','delta': ans})
                try:
                    pcm16 = call_tts_pcm16le(ans); pcm24 = _resample_pcm_ffmpeg(pcm16,16000,REALTIME_SR); _stream_pcm24(ws, pcm24)
                    log(f"[REALTIME][WS] streamed tts bytes={len(pcm24)}")
                except Exception as e:
                    log(f"[REALTIME][WS][TTS ERROR] {e}")
                    _ws_send(ws, {'type':'error','error':{'message': f'tts failed: {e}'}})
                st.add({'id': _nid('item'),'type':'message','role':'assistant','content':[{'type':'output_text','text': ans}]})
                _ws_send(ws, {'type':'response.done','response': {'id': _nid('resp')}}); continue
            log(f"[REALTIME][WS] unhandled type received: {t}")
            _ws_send(ws, {'type':'error','error':{'message': f'unhandled type: {t}'}})

# ────────────────────────────── Entrypoint ──────────────────────────────
if __name__ == '__main__':
    sio.run(app, host=os.getenv('SERVER_NAME','0.0.0.0'), port=int(os.getenv('SERVER_PORT',5001)), debug=False)
