"""
Síntesis de voz (TTS): texto o SSML -> WAV mono 16 bits a 24 kHz.

Backends (variable de entorno `TTS_BACKEND`, por defecto `edge`):

- **edge**: voces neuronales de Microsoft vía `edge-tts` (gratis, sin clave). La voz del bot es
  `es-ES-ElviraNeural` (`EDGE_VOZ`). edge-tts no acepta SSML propio, así que el SSML se
  `segmentar`-a: cada trozo se sintetiza con su rate/pitch/volume **en paralelo** (asyncio), las
  pausas se insertan como silencio y todo se concatena con numpy.
- **elevenlabs**: API REST de ElevenLabs (`ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`,
  `ELEVENLABS_MODEL`). Admite `<break>` de forma nativa; el resto de etiquetas se aplana.
  Con `ELEVENLABS_VOICE_ID` = voz clonada (ver `voz/clonar_voz.py`) el bot habla con tu voz.
- **xtts**: Coqui XTTS-v2 en local, clonando la voz de `voz/muestras/referencia.wav`. Se usa el
  paquete `TTS` si está instalado en este Python o, si no, el servidor `voz/xtts_servidor.py`
  lanzado con el entorno aparte `.venv-xtts` (`XTTS_URL`, por defecto http://127.0.0.1:8020).

Una caché en disco (`.cache/tts/`, por hash de backend+voz+texto) hace que las frases repetidas
(prompts de slot filling, saludos...) salgan en milisegundos.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np

from voz.ssml import a_ssml, segmentar, ssml_para_elevenlabs

FRECUENCIA = 24000  # Hz de todo el audio que devuelve este módulo
RAIZ = Path(__file__).resolve().parent.parent
DIR_CACHE = Path(os.environ.get("TTS_CACHE_DIR", RAIZ / ".cache" / "tts"))
DIR_MUESTRAS = Path(__file__).resolve().parent / "muestras"

VOZ_EDGE_POR_DEFECTO = "es-ES-ElviraNeural"
VERSION_CACHE = "v2"  # cambiarla invalida la caché si cambia el procesado


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def sintetizar(texto_o_ssml: str, backend: str | None = None, voz: str | None = None) -> bytes:
    """
    Sintetiza texto plano o SSML y devuelve un WAV (mono, 16 bits, 24 kHz).

    `backend`: edge | elevenlabs | xtts (por defecto `TTS_BACKEND` o edge).
    `voz`: nombre de voz edge, voice_id de ElevenLabs o ruta del WAV de referencia para XTTS.
    Es seguro llamarla desde código síncrono aunque haya un bucle asyncio en marcha.
    """
    backend = (backend or os.environ.get("TTS_BACKEND") or "edge").lower()
    ssml = a_ssml(texto_o_ssml or "")
    clave = _clave_cache(backend, voz or "", ssml)

    en_cache = _leer_cache(clave)
    if en_cache is not None:
        return en_cache

    if backend == "edge":
        muestras = _ejecutar_async(_sintetizar_edge(ssml, voz or os.environ.get("EDGE_VOZ")
                                                    or VOZ_EDGE_POR_DEFECTO))
    elif backend == "elevenlabs":
        muestras = _sintetizar_elevenlabs(ssml, voz)
    elif backend == "xtts":
        muestras = _sintetizar_xtts(ssml, voz)
    else:
        raise ValueError(f"Backend TTS desconocido: {backend!r} (usa edge, elevenlabs o xtts)")

    wav = a_wav(muestras)
    _guardar_cache(clave, wav)
    return wav


async def sintetizar_async(texto_o_ssml: str, backend: str | None = None,
                           voz: str | None = None) -> bytes:
    """Versión para código asíncrono (bot de Telegram, FastAPI): no bloquea el bucle."""
    return await asyncio.to_thread(sintetizar, texto_o_ssml, backend, voz)


def a_ogg_opus(wav: bytes, kbps: int = 32) -> bytes:
    """Convierte WAV a OGG/Opus mono (formato de las notas de voz de Telegram) con ffmpeg."""
    return _ffmpeg(["-i", "pipe:0", "-ac", "1", "-c:a", "libopus", "-b:a", f"{kbps}k",
                    "-application", "voip", "-f", "ogg", "pipe:1"], wav)


def duracion(wav: bytes) -> float:
    """Duración en segundos de un WAV."""
    with wave.open(io.BytesIO(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def a_wav(muestras: np.ndarray, frecuencia: int = FRECUENCIA) -> bytes:
    """float32 [-1, 1] o int16 -> bytes WAV mono 16 bits."""
    if muestras.dtype != np.int16:
        muestras = (np.clip(muestras, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(frecuencia)
        w.writeframes(muestras.tobytes())
    return buf.getvalue()


def leer_wav(wav: bytes) -> tuple[np.ndarray, int]:
    """WAV (cualquier formato que entienda ffmpeg/soundfile) -> (float32 mono, frecuencia)."""
    import soundfile as sf
    datos, fs = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    return datos.mean(axis=1), fs


# ---------------------------------------------------------------------------
# Utilidades de audio
# ---------------------------------------------------------------------------

def _silencio(ms: int) -> np.ndarray:
    return np.zeros(int(FRECUENCIA * ms / 1000), dtype=np.float32)


def _recortar_silencio(x: np.ndarray, umbral: float = 0.01, margen_ms: int = 40) -> np.ndarray:
    """Quita el silencio de principio y final de un trozo (los motores lo añaden siempre)
    para que las pausas del SSML sean las que mandan."""
    if x.size == 0:
        return x
    activos = np.flatnonzero(np.abs(x) > umbral * max(float(np.abs(x).max()), 1e-6))
    if activos.size == 0:
        return x[:0]
    margen = int(FRECUENCIA * margen_ms / 1000)
    return x[max(0, activos[0] - margen): activos[-1] + margen]


def _decodificar(audio: bytes) -> np.ndarray:
    """mp3/ogg/wav -> float32 mono a 24 kHz. soundfile (rápido, sin proceso) y si no, ffmpeg."""
    try:
        import soundfile as sf
        datos, fs = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        mono = datos.mean(axis=1)
        if fs == FRECUENCIA:
            return mono
    except Exception:
        pass
    pcm = _ffmpeg(["-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(FRECUENCIA), "pipe:1"], audio)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _ffmpeg(args: list[str], entrada: bytes) -> bytes:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                          input=entrada, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló: {proc.stderr.decode(errors='replace')[:300]}")
    return proc.stdout


def _unir(trozos: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(trozos) if trozos else np.zeros(0, dtype=np.float32)


def _ejecutar_async(corrutina):
    """Ejecuta una corrutina desde código síncrono, haya o no un bucle ya en marcha
    (en ese caso se usa un hilo con su propio bucle para no bloquearlo ni fallar)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(corrutina)
    resultado: dict = {}

    def hilo():
        try:
            resultado["ok"] = asyncio.run(corrutina)
        except BaseException as e:  # se relanza en el hilo llamante
            resultado["error"] = e

    t = threading.Thread(target=hilo)
    t.start()
    t.join()
    if "error" in resultado:
        raise resultado["error"]
    return resultado["ok"]


# ---------------------------------------------------------------------------
# Caché en disco
# ---------------------------------------------------------------------------

def _clave_cache(*partes: str) -> str:
    return hashlib.sha256("|".join((VERSION_CACHE, *partes)).encode("utf-8")).hexdigest()[:32]


def _cache_activa() -> bool:
    return os.environ.get("TTS_CACHE", "1") != "0"


def _leer_cache(clave: str) -> bytes | None:
    ruta = DIR_CACHE / f"{clave}.wav"
    if _cache_activa() and ruta.exists():
        return ruta.read_bytes()
    return None


def _guardar_cache(clave: str, wav: bytes) -> None:
    if not _cache_activa():
        return
    try:
        DIR_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = DIR_CACHE / f"{clave}.{os.getpid()}.tmp"
        tmp.write_bytes(wav)
        os.replace(tmp, DIR_CACHE / f"{clave}.wav")  # escritura atómica
    except OSError:
        pass  # la caché es una optimización: si falla, no rompe la síntesis


# ---------------------------------------------------------------------------
# Backend edge-tts
# ---------------------------------------------------------------------------

async def _trozo_edge(texto: str, voz: str, rate: str, pitch: str, volume: str) -> np.ndarray:
    """Sintetiza un trozo con edge-tts (mp3) y lo devuelve como float32 a 24 kHz.
    Cada trozo se cachea aparte: los fragmentos fijos se reutilizan entre respuestas distintas."""
    import edge_tts

    clave = _clave_cache("edge-trozo", voz, rate, pitch, volume, texto)
    en_cache = _leer_cache(clave)
    if en_cache is not None:
        return leer_wav(en_cache)[0]

    ultimo_error: Exception | None = None
    for _ in range(3):  # el servicio a veces corta la conexión: reintentamos
        try:
            com = edge_tts.Communicate(texto, voz, rate=rate, pitch=pitch, volume=volume)
            mp3 = bytearray()
            async for parte in com.stream():
                if parte["type"] == "audio":
                    mp3.extend(parte["data"])
            if mp3:
                muestras = await asyncio.to_thread(_decodificar, bytes(mp3))
                muestras = _recortar_silencio(muestras)
                _guardar_cache(clave, a_wav(muestras))
                return muestras
        except Exception as e:  # NoAudioReceived, WebSocketError...
            ultimo_error = e
        await asyncio.sleep(0.2)
    raise RuntimeError(f"edge-tts no devolvió audio para {texto[:40]!r}: {ultimo_error}")


async def _sintetizar_edge(ssml: str, voz: str) -> np.ndarray:
    segmentos = segmentar(ssml)
    tareas = {
        i: asyncio.create_task(_trozo_edge(s["texto"], voz, s["rate"], s["pitch"], s["volume"]))
        for i, s in enumerate(segmentos) if s["tipo"] == "texto"
    }
    trozos = []
    for i, s in enumerate(segmentos):
        if s["tipo"] == "pausa":
            trozos.append(_silencio(s["ms"]))
        else:
            trozos.append(await tareas[i])
            # Entre trozos de texto sin pausa explícita, un respiro mínimo para que no se pisen.
            siguiente = segmentos[i + 1] if i + 1 < len(segmentos) else None
            if siguiente and siguiente["tipo"] == "texto":
                trozos.append(_silencio(60))
    return _unir(trozos)


# ---------------------------------------------------------------------------
# Backend ElevenLabs (API REST)
# ---------------------------------------------------------------------------

URL_ELEVENLABS = "https://api.elevenlabs.io/v1"


def _sintetizar_elevenlabs(ssml: str, voz: str | None) -> np.ndarray:
    """
    POST /v1/text-to-speech/{voice_id}?output_format=pcm_24000 -> PCM 16 bits crudo.
    `eleven_multilingual_v2` (calidad) o `eleven_flash_v2_5` (≈75 ms, menor latencia).
    """
    import requests

    clave = os.environ.get("ELEVENLABS_API_KEY")
    voice_id = voz or os.environ.get("ELEVENLABS_VOICE_ID")
    if not clave or not voice_id:
        raise RuntimeError("Backend elevenlabs: define ELEVENLABS_API_KEY y ELEVENLABS_VOICE_ID "
                           "(la voz clonada se crea con `python -m voz.clonar_voz elevenlabs`).")
    modelo = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    cuerpo = {
        "text": ssml_para_elevenlabs(ssml),
        "model_id": modelo,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.0,
                           "use_speaker_boost": True},
    }
    if "flash" in modelo or "turbo" in modelo:
        cuerpo["language_code"] = "es"  # solo los modelos v2.5 aceptan forzar idioma
    resp = requests.post(
        f"{URL_ELEVENLABS}/text-to-speech/{voice_id}",
        params={"output_format": f"pcm_{FRECUENCIA}"},
        headers={"xi-api-key": clave, "Content-Type": "application/json"},
        json=cuerpo, timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs {resp.status_code}: {resp.text[:300]}")
    return np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Backend XTTS-v2 (clonación de voz local)
# ---------------------------------------------------------------------------

INSTRUCCIONES_XTTS = """\
XTTS-v2 no está disponible. Para usar la voz clonada en local (GPU recomendada):

  uv venv --python 3.11 .venv-xtts
  uv pip install --python .venv-xtts/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
  uv pip install --python .venv-xtts/Scripts/python.exe coqui-tts six
  .venv-xtts\\Scripts\\python.exe -m voz.xtts_servidor        (deja el servidor en marcha)

La primera vez descarga el modelo (~1,8 GB) y pide aceptar la licencia CPML de Coqui (uso no comercial).
Muestra de referencia: voz/muestras/referencia.wav (grábala con `python -m voz.clonar_voz grabar`).
"""


def referencia_por_defecto() -> Path:
    """Muestra de voz a clonar: la tuya si la has grabado; si no, la demo sintética."""
    propia = DIR_MUESTRAS / "referencia.wav"
    return propia if propia.exists() else DIR_MUESTRAS / "referencia_demo.wav"


def _sintetizar_xtts(ssml: str, voz: str | None) -> np.ndarray:
    """
    XTTS no entiende SSML: se segmenta, cada trozo de texto se sintetiza con la voz clonada
    (la velocidad del <prosody> se traslada al parámetro `speed`) y se insertan las pausas.
    """
    referencia = Path(voz) if voz else referencia_por_defecto()
    if not referencia.exists():
        raise RuntimeError(f"No existe la muestra de voz {referencia}.\n{INSTRUCCIONES_XTTS}")
    sintetizar_trozo = _motor_xtts()

    trozos = []
    for s in segmentar(ssml):
        if s["tipo"] == "pausa":
            trozos.append(_silencio(s["ms"]))
        else:
            velocidad = 1.0 + int(s["rate"].rstrip("%")) / 100
            trozos.append(_recortar_silencio(sintetizar_trozo(s["texto"], str(referencia), velocidad)))
            trozos.append(_silencio(80))
    return _unir(trozos)


_xtts_local = None
_xtts_lock = threading.Lock()


def _motor_xtts():
    """Devuelve una función (texto, referencia, velocidad) -> float32 24 kHz.
    1º intenta el paquete `TTS` en este proceso; 2º el servidor local de `.venv-xtts`."""
    global _xtts_local
    try:
        from voz.xtts_servidor import MotorXTTS  # importa torch/TTS: falla si no están
        with _xtts_lock:
            if _xtts_local is None:
                _xtts_local = MotorXTTS()
        return _xtts_local.sintetizar
    except ImportError:
        pass

    import requests
    url = os.environ.get("XTTS_URL", "http://127.0.0.1:8020")
    try:
        requests.get(f"{url}/salud", timeout=1).raise_for_status()
    except Exception:
        raise RuntimeError(f"No hay XTTS en este Python ni servidor en {url}.\n{INSTRUCCIONES_XTTS}")

    def remoto(texto: str, referencia: str, velocidad: float) -> np.ndarray:
        r = requests.post(f"{url}/tts", json={"texto": texto, "referencia": referencia,
                                              "velocidad": velocidad, "idioma": "es"}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Servidor XTTS {r.status_code}: {r.text[:300]}")
        return _decodificar(r.content)

    return remoto
