"""
Cliente de voz de Seguros Alonso: micrófono -> Dialogflow (+ webhook) -> altavoz.

Versión ampliada del cliente de voz del curso. Se puede combinar:

  --backend  local | google        emulador local de Dialogflow (lee agente_dialogflow/) o la API real
  --stt      whisper | dialogflow  faster-whisper en local (GPU) o el STT nativo de Dialogflow
  --tts      edge | elevenlabs | xtts | dialogflow
                                   edge-tts (por defecto), ElevenLabs (voz clonada), XTTS-v2 (voz
                                   clonada en local) o el audio que devuelve Dialogflow (TTS nativo)

Ejemplos (desde la raíz del repo, con el webhook arrancado: uvicorn webhook.main:app):

  python cliente_voz.py                                   # local + whisper + edge
  python cliente_voz.py --backend google --stt dialogflow --tts dialogflow   # 100 % Dialogflow (como el curso)
  python cliente_voz.py --tts xtts                        # el bot habla con la voz clonada
  python cliente_voz.py --archivo turno1.wav turno2.ogg --sin-audio          # conversación grabada

Cada turno envía `payload = {"canal": "voz"}` para que el webhook responda en SSML, y muestra
una tabla de latencias por etapa (STT, Dialogflow+webhook, TTS, total) con aviso si pasa de 5 s.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FS_MICRO = 16000          # Dialogflow (LINEAR_16) y Whisper trabajan a 16 kHz
LIMITE_LATENCIA = 5.0     # requisito del enunciado: < 5 s por turno
PAYLOAD_VOZ = {"canal": "voz"}

if hasattr(sys.stdout, "reconfigure"):  # tildes y ñ en la consola de Windows
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Audio: grabar y reproducir
# ---------------------------------------------------------------------------

def escuchar(segundos_max: float, parar_en_silencio: bool = True) -> np.ndarray:
    """
    Graba del micrófono (int16, 16 kHz) hasta `segundos_max` o, si `parar_en_silencio`, hasta
    que tras haber hablado haya ~0,9 s de silencio: así no se espera siempre el máximo.
    """
    import sounddevice as sd

    bloque = FS_MICRO // 10                     # trozos de 100 ms
    trozos: list[np.ndarray] = []
    ruido = None                                # nivel de fondo medido al principio
    habla, silencio = 0, 0
    print(f"  Habla (máx. {segundos_max:.0f} s)...", flush=True)
    with sd.InputStream(samplerate=FS_MICRO, channels=1, dtype="int16", blocksize=bloque) as flujo:
        for n in range(int(segundos_max * 10)):
            datos, _ = flujo.read(bloque)
            trozos.append(datos[:, 0].copy())
            energia = float(np.sqrt(np.mean(datos.astype(np.float32) ** 2)))
            if n < 3:
                ruido = energia if ruido is None else max(ruido, energia)
                continue
            if energia > max(3 * (ruido or 0), 300):
                habla, silencio = habla + 1, 0
            else:
                silencio += 1
            if parar_en_silencio and habla >= 3 and silencio >= 9:
                break
    return np.concatenate(trozos)


def pcm_a_wav(pcm: np.ndarray, fs: int = FS_MICRO) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


def reproducir(wav: bytes) -> None:
    import sounddevice as sd
    import soundfile as sf
    datos, fs = sf.read(io.BytesIO(wav), dtype="float32")
    sd.play(datos, fs)
    sd.wait()


# ---------------------------------------------------------------------------
# Un turno de conversación
# ---------------------------------------------------------------------------

@dataclass
class ResultadoTurno:
    turno: object                       # dialogflow_local.cliente.Turno
    transcripcion: str
    audio_respuesta: bytes | None
    latencias: dict[str, float] = field(default_factory=dict)


class ConversacionVoz:
    """Encadena STT -> Dialogflow (+webhook) -> TTS midiendo cada etapa."""

    def __init__(self, backend: str, stt: str, tts: str, voz: str | None = None):
        from dialogflow_local.cliente import crear_cliente

        if "dialogflow" in (stt, tts) and backend != "google":
            raise SystemExit("--stt/--tts dialogflow usan el audio nativo de Dialogflow: "
                             "requieren --backend google.")
        # Con --tts dialogflow, el cliente de Google pide también el audio de respuesta (TTS nativo)
        # en las consultas de texto/evento, no solo en las de audio.
        opciones = {"sintetizar": True} if backend == "google" and tts == "dialogflow" else {}
        self.cliente = crear_cliente(backend, **opciones)
        self.backend, self.stt, self.tts, self.voz = backend, stt, tts, voz
        self.sesion = f"voz-{uuid.uuid4()}"   # una sesión por conversación (contextos, slots)

    def precargar(self) -> None:
        """Carga Whisper y calienta el TTS antes del primer turno (no cuenta en la latencia)."""
        if self.stt == "whisper":
            from voz.stt import precargar
            inicio = time.perf_counter()
            dispositivo = precargar()
            print(f"  Whisper listo en {dispositivo} ({time.perf_counter() - inicio:.1f} s)")

    # -- etapas -------------------------------------------------------------

    def bienvenida(self) -> ResultadoTurno:
        inicio = time.perf_counter()
        turno = self.cliente.evento(self.sesion, "WELCOME", payload=PAYLOAD_VOZ)
        latencias = {"dialogflow": time.perf_counter() - inicio}
        return self._con_tts(ResultadoTurno(turno, "", None, latencias))

    def desde_texto(self, texto: str) -> ResultadoTurno:
        inicio = time.perf_counter()
        turno = self.cliente.texto(self.sesion, texto, payload=PAYLOAD_VOZ)
        latencias = {"dialogflow": time.perf_counter() - inicio}
        return self._con_tts(ResultadoTurno(turno, texto, None, latencias))

    def desde_audio(self, audio: bytes | np.ndarray) -> ResultadoTurno:
        """`audio`: PCM int16 16 kHz (micrófono) o bytes de un fichero en cualquier formato."""
        latencias: dict[str, float] = {}
        if self.stt == "dialogflow":
            # STT nativo: el audio va dentro del detectIntent y la transcripción vuelve en query_text.
            pcm = audio if isinstance(audio, np.ndarray) else self._a_pcm16k(audio)
            inicio = time.perf_counter()
            turno = self.cliente.audio(self.sesion, pcm_a_wav(pcm), payload=PAYLOAD_VOZ)
            latencias["dialogflow"] = time.perf_counter() - inicio
            transcripcion = turno.query_text
        else:
            from voz.stt import transcribir
            entrada = (audio.astype(np.float32) / 32768.0) if isinstance(audio, np.ndarray) else audio
            inicio = time.perf_counter()
            transcripcion = transcribir(entrada)
            latencias["stt"] = time.perf_counter() - inicio
            if not transcripcion:
                return ResultadoTurno(None, "", None, latencias)
            inicio = time.perf_counter()
            turno = self.cliente.texto(self.sesion, transcripcion, payload=PAYLOAD_VOZ)
            latencias["dialogflow"] = time.perf_counter() - inicio
        return self._con_tts(ResultadoTurno(turno, transcripcion, None, latencias))

    def _con_tts(self, r: ResultadoTurno) -> ResultadoTurno:
        turno = r.turno
        if not (turno.texto or turno.ssml):
            return r
        if self.tts == "dialogflow":
            r.audio_respuesta = turno.audio  # ya viene sintetizado en la respuesta de Dialogflow
            if r.audio_respuesta is None:
                print("  (Dialogflow no devolvió audio; se usa edge-tts)")
        if r.audio_respuesta is None:
            from voz.tts import sintetizar
            backend = "edge" if self.tts == "dialogflow" else self.tts
            inicio = time.perf_counter()
            r.audio_respuesta = sintetizar(turno.ssml or turno.texto, backend=backend, voz=self.voz)
            r.latencias["tts"] = time.perf_counter() - inicio
        return r

    @staticmethod
    def _a_pcm16k(audio: bytes) -> np.ndarray:
        from voz.stt import decodificar
        return (decodificar(audio) * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

def mostrar(r: ResultadoTurno) -> None:
    turno = r.turno
    if turno is None:
        print("  (no se ha entendido nada; prueba otra vez)")
        _tabla_latencias(r, None)
        return
    if r.transcripcion:
        print(f"  Has dicho  : {r.transcripcion}")
    print(f"  Intención  : {turno.intent or '-'} ({turno.confianza:.2f})")
    parametros = {k: v for k, v in (turno.parametros or {}).items() if v not in ("", None, [], {})}
    print(f"  Parámetros : {parametros}")
    print(f"  Responde   : {turno.texto}")
    if turno.ssml:
        print(f"  SSML       : {turno.ssml}")
    _tabla_latencias(r, turno)


def _tabla_latencias(r: ResultadoTurno, turno) -> None:
    lat = r.latencias
    filas = [("STT", lat.get("stt")), ("Dialogflow + webhook", lat.get("dialogflow")),
             ("TTS", lat.get("tts"))]
    webhook = (getattr(turno, "latencias", None) or {}).get("webhook") if turno else None
    total = sum(v for _, v in filas if v)
    print("  ┌──────────────────────┬─────────┐")
    for nombre, valor in filas:
        texto = "    -  " if valor is None else f"{valor:6.2f} s"
        print(f"  │ {nombre:<20} │ {texto} │")
        if nombre.startswith("Dialogflow") and webhook is not None:
            print(f"  │   └ de ello webhook  │ {webhook:6.2f} s │")
    print("  ├──────────────────────┼─────────┤")
    print(f"  │ {'TOTAL turno':<20} │ {total:6.2f} s │")
    print("  └──────────────────────┴─────────┘")
    if total > LIMITE_LATENCIA:
        print(f"  ⚠ AVISO: el turno ha tardado {total:.2f} s (> {LIMITE_LATENCIA:.0f} s)")


def responder(r: ResultadoTurno, args, n: int) -> None:
    mostrar(r)
    if r.audio_respuesta:
        if args.guardar:
            args.guardar.mkdir(parents=True, exist_ok=True)
            (args.guardar / f"respuesta_{n:02d}.wav").write_bytes(r.audio_respuesta)
        if not args.sin_audio:
            reproducir(r.audio_respuesta)


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Cliente de voz del asistente de Seguros Alonso")
    p.add_argument("--backend", choices=["local", "google"], default=None,
                   help="Dialogflow: emulador local o API de Google (por defecto DIALOGFLOW_BACKEND)")
    p.add_argument("--stt", choices=["whisper", "dialogflow"], default="whisper")
    p.add_argument("--tts", choices=["edge", "elevenlabs", "xtts", "dialogflow"], default="edge")
    p.add_argument("--voz", default=None,
                   help="voz edge (es-ES-ElviraNeural), voice_id de ElevenLabs o WAV de referencia XTTS")
    p.add_argument("--segundos", type=float, default=6, help="duración máxima de cada grabación")
    p.add_argument("--sin-corte", action="store_true",
                   help="grabar siempre los N segundos (no cortar al detectar silencio)")
    p.add_argument("--archivo", type=Path, nargs="+",
                   help="usar ficheros de audio (uno por turno) en lugar del micrófono")
    p.add_argument("--sin-audio", action="store_true", help="no reproducir las respuestas")
    p.add_argument("--sin-bienvenida", action="store_true")
    p.add_argument("--guardar", type=Path, help="carpeta donde guardar los WAV de respuesta")
    args = p.parse_args()

    backend = args.backend or os.environ.get("DIALOGFLOW_BACKEND", "local")
    conversacion = ConversacionVoz(backend, args.stt, args.tts, args.voz)

    print("=" * 60)
    print("Asistente de voz de Seguros Alonso")
    print(f"Dialogflow: {backend} | STT: {args.stt} | TTS: {args.tts} | sesión: {conversacion.sesion}")
    print("=" * 60)
    conversacion.precargar()

    n = 0
    if not args.sin_bienvenida:
        n += 1
        responder(conversacion.bienvenida(), args, n)

    if args.archivo:
        for ruta in args.archivo:
            n += 1
            print(f"\n> [{ruta.name}]")
            responder(conversacion.desde_audio(ruta.read_bytes()), args, n)
        return

    print("\nEnter = hablar | escribe una frase = enviarla como texto | q = salir")
    while True:
        try:
            entrada = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if entrada.lower() in ("q", "salir"):
            break
        n += 1
        if entrada:
            r = conversacion.desde_texto(entrada)
        else:
            r = conversacion.desde_audio(escuchar(args.segundos, not args.sin_corte))
        responder(r, args, n)


if __name__ == "__main__":
    main()
