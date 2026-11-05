"""
Reconocimiento de voz (STT) con faster-whisper, en local.

- Modelo `WHISPER_MODEL` (por defecto "small"), cargado una sola vez. Se intenta GPU (CUDA,
  float16) y, si no hay o faltan las DLL de CUDA 12, se cae a CPU int8 automáticamente.
  `WHISPER_DEVICE=cpu|cuda` fuerza el dispositivo.
- Acepta wav/ogg/webm/mp3... en bytes: ffmpeg lo convierte a 16 kHz mono (un WAV ya en
  16 kHz mono 16 bits se lee directamente, sin lanzar ffmpeg).
- `initial_prompt` con vocabulario del dominio (Seguros Alonso, peritaje, franquicia...) y una
  matrícula de ejemplo "1234 BCD" para que Whisper escriba las matrículas en ese formato.
- Post-proceso `normalizar_matricula_hablada`: "uno dos tres cuatro be ce de" -> "1234 BCD",
  que es lo que espera la entidad regex `@matricula` de Dialogflow.
"""

from __future__ import annotations

import glob
import io
import os
import re
import subprocess
import sys
import threading
import wave

import numpy as np

FRECUENCIA = 16000
PROMPT_DOMINIO = (
    "Seguros Alonso. Quiero pedir cita de peritaje para la matrícula 1234 BCD. "
    "Póliza, siniestro, franquicia, perito, taller concertado, parabrisas, granizo. "
    "Madrid, Barcelona, Valencia, Sevilla, Bilbao, Zaragoza, Málaga, Gijón, Oviedo, A Coruña."
)

# Frases que Whisper "alucina" con silencio o ruido (vienen de subtítulos de YouTube).
ALUCINACIONES = re.compile(
    r"^(¡?suscr[ií]bete!?|gracias por ver(lo| el v[ií]deo)?\.?|subt[ií]tulos.*|"
    r"amara\.org.*|\.+|gracias\.?)$", re.IGNORECASE)

_modelo = None
_dispositivo = ""
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def _registrar_dll_cuda() -> None:
    """En Windows, ctranslate2 necesita cuBLAS/cuDNN de CUDA 12. Si están instalados como
    paquetes pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) se añaden sus carpetas `bin` al PATH."""
    if os.name != "nt":
        return
    patron = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")
    for carpeta in glob.glob(patron):
        if carpeta not in os.environ["PATH"]:
            os.add_dll_directory(carpeta)
            os.environ["PATH"] = carpeta + os.pathsep + os.environ["PATH"]


def _cargar():
    global _modelo, _dispositivo
    with _lock:
        if _modelo is not None:
            return _modelo
        from faster_whisper import WhisperModel

        nombre = os.environ.get("WHISPER_MODEL", "small")
        preferido = os.environ.get("WHISPER_DEVICE", "auto").lower()
        intentos = [("cuda", "float16"), ("cpu", "int8")]
        if preferido == "cpu":
            intentos = [("cpu", "int8")]
        elif preferido == "cuda":
            intentos = [("cuda", "float16")]

        _registrar_dll_cuda()
        ultimo_error = None
        for dispositivo, tipo in intentos:
            try:
                modelo = WhisperModel(nombre, device=dispositivo, compute_type=tipo)
                # La primera inferencia es la que carga cuBLAS/cuDNN: si faltan, falla aquí y no
                # en mitad de una conversación. Además deja la GPU "caliente" (~5 s la 1ª vez).
                list(modelo.transcribe(np.zeros(FRECUENCIA, dtype=np.float32), language="es")[0])
                _modelo, _dispositivo = modelo, f"{dispositivo}/{tipo}"
                return _modelo
            except Exception as e:  # sin GPU, sin DLL de CUDA 12...
                ultimo_error = e
        raise RuntimeError(f"No se pudo cargar faster-whisper '{nombre}': {ultimo_error}")


def precargar() -> str:
    """Carga y calienta el modelo (llamar al arrancar). Devuelve el dispositivo usado."""
    _cargar()
    return _dispositivo


def dispositivo() -> str:
    """'cuda/float16', 'cpu/int8' o '' si aún no se ha cargado."""
    return _dispositivo


# ---------------------------------------------------------------------------
# Audio de entrada
# ---------------------------------------------------------------------------

def decodificar(audio: bytes) -> np.ndarray:
    """Bytes de audio (cualquier formato) -> float32 mono a 16 kHz."""
    if audio[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(audio), "rb") as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (FRECUENCIA, 1, 2):
                    datos = w.readframes(w.getnframes())
                    return np.frombuffer(datos, dtype=np.int16).astype(np.float32) / 32768.0
        except wave.Error:
            pass  # WAV en float o comprimido: lo resuelve ffmpeg
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(FRECUENCIA), "pipe:1"],
        input=audio, capture_output=True, check=False)
    if proc.returncode != 0:
        raise ValueError(f"Audio no reconocido por ffmpeg: {proc.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Transcripción
# ---------------------------------------------------------------------------

def transcribir(audio: bytes | np.ndarray, idioma: str = "es") -> str:
    """
    Transcribe audio (bytes en cualquier formato, o float32 16 kHz) y normaliza matrículas.
    Devuelve "" si no se ha dicho nada reconocible.
    """
    muestras = audio if isinstance(audio, np.ndarray) else decodificar(audio)
    if muestras.size < FRECUENCIA // 4:  # menos de 0,25 s
        return ""
    modelo = _cargar()
    segmentos, _info = modelo.transcribe(
        muestras.astype(np.float32),
        language=idioma,
        beam_size=2,                     # 1-2 basta con frases cortas; 5 duplica la latencia
        vad_filter=True,                 # descarta silencios: menos alucinaciones y más rápido
        vad_parameters={"min_silence_duration_ms": 400},
        initial_prompt=PROMPT_DOMINIO,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    textos = [s.text.strip() for s in segmentos
              if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
    texto = " ".join(t for t in textos if t and not ALUCINACIONES.match(t))
    return normalizar_matricula_hablada(texto.strip())


# ---------------------------------------------------------------------------
# Normalización de matrículas dictadas
# ---------------------------------------------------------------------------

CIFRAS_HABLADAS = {"cero": "0", "uno": "1", "un": "1", "una": "1", "dos": "2", "tres": "3",
                   "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
                   "nueve": "9"}

# Nombres de letra en español (incluye variantes que Whisper suele escribir).
LETRAS_HABLADAS = {
    "a": "A", "be": "B", "ve": "B", "ce": "C", "de": "D", "e": "E", "efe": "F", "ge": "G",
    "je": "G", "hache": "H", "i": "I", "jota": "J", "ka": "K", "ele": "L",
    "eme": "M", "ene": "N", "eñe": "Ñ", "o": "O", "pe": "P", "cu": "Q", "erre": "R",
    "ere": "R", "ese": "S", "te": "T", "u": "U", "uve": "V", "equis": "X", "zeta": "Z",
    "ceta": "Z", "seta": "Z", "ye": "Y",
}
LETRAS_DOBLES = {("uve", "doble"): "W", ("doble", "uve"): "W", ("i", "griega"): "Y"}

# Letras válidas en una matrícula española actual (sin vocales, Ñ ni Q).
LETRAS_MATRICULA = set("BCDFGHJKLMNPRSTVWXYZ")

# "1234BCD" se parte en "1234" + "BCD" (el separador vacío vale), así que no hace falta otro caso.
_TOKEN = re.compile(r"\d+(?:\.\d+)*|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_SEPARADOR = re.compile(r"^[\s,.\-–:;]*$")


def _letras_de(tokens: list[re.Match], i: int) -> tuple[str, int]:
    """Letras que representa el token i (y quizá el siguiente). Devuelve (letras, tokens usados)."""
    palabra = tokens[i].group().lower()
    if i + 1 < len(tokens):
        par = (palabra, tokens[i + 1].group().lower())
        if par in LETRAS_DOBLES:
            return LETRAS_DOBLES[par], 2
    if palabra in LETRAS_HABLADAS:
        return LETRAS_HABLADAS[palabra], 1
    original = tokens[i].group()
    # "BCD", "bcd" o "B": letras sueltas escritas tal cual (máx. 3, todas válidas en matrícula)
    if len(original) <= 3 and all(c.upper() in LETRAS_MATRICULA for c in original):
        return original.upper(), 1
    return "", 0


def normalizar_matricula_hablada(texto: str) -> str:
    """
    Reescribe matrículas dictadas en el formato "1234 BCD":

        "1234-BCD", "1234bcd", "1.234 BCD"         -> "1234 BCD"
        "1 2 3 4 B C D", "12 34, b, c, d"          -> "1234 BCD"
        "uno dos tres cuatro be ce de"             -> "1234 BCD"
        "cinco seis siete ocho efe ge hache"       -> "5678 FGH"

    Solo se reescribe si salen exactamente 4 cifras seguidas de 3 letras válidas, separadas
    únicamente por espacios/comas/guiones/puntos; el resto del texto no se toca.
    """
    if not texto:
        return texto
    tokens = list(_TOKEN.finditer(texto))
    reemplazos: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        resultado = _matricula_desde(texto, tokens, i)
        if resultado:
            fin_token, matricula = resultado
            reemplazos.append((tokens[i].start(), tokens[fin_token].end(), matricula))
            i = fin_token + 1
        else:
            i += 1

    for inicio, fin, matricula in reversed(reemplazos):
        texto = texto[:inicio] + matricula + texto[fin:]
    return texto


def _matricula_desde(texto: str, tokens: list[re.Match], i: int) -> tuple[int, str] | None:
    """Intenta leer una matrícula empezando en el token i. Devuelve (último token, "1234 BCD")."""
    cifras, letras = "", ""
    j = i
    while j < len(tokens):
        if j > i and not _SEPARADOR.match(texto[tokens[j - 1].end():tokens[j].start()]):
            return None  # entre medias hay algo raro (otra palabra, un símbolo...)
        palabra = tokens[j].group()
        if len(cifras) < 4:
            if palabra.replace(".", "").isdigit():
                cifras += palabra.replace(".", "")
            elif palabra.lower() in CIFRAS_HABLADAS:
                cifras += CIFRAS_HABLADAS[palabra.lower()]
            else:
                return None
            if len(cifras) > 4:
                return None
            j += 1
            continue
        nuevas, usados = _letras_de(tokens, j)
        if not nuevas:
            return None
        letras += nuevas
        if len(letras) > 3:
            return None
        j += usados
        if len(letras) == 3:
            if not set(letras) <= LETRAS_MATRICULA:
                return None
            return j - 1, f"{cifras} {letras}"
    return None

