"""
Motor y servidor local de XTTS-v2 (Coqui) para hablar con una voz clonada.

XTTS necesita torch + coqui-tts, que traen dependencias pesadas (transformers, torch CUDA...).
Para no romper el entorno principal se instalan en un entorno aparte `.venv-xtts` y este módulo
se ejecuta como un pequeño servidor HTTP que mantiene el modelo cargado en GPU:

    .venv-xtts\\Scripts\\python.exe -m voz.xtts_servidor            # escucha en 127.0.0.1:8020

`voz.tts` (backend "xtts") le envía cada trozo de texto. Si el paquete `TTS` estuviera instalado
en el mismo Python, `voz.tts` usa `MotorXTTS` directamente, sin servidor.

Endpoints:
    GET  /salud  -> {"ok": true, "dispositivo": "cuda"}
    POST /tts    {"texto", "referencia" (ruta WAV), "idioma"="es", "velocidad"=1.0} -> audio/wav 24 kHz
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch  # noqa: F401  (si no está, ImportError: voz.tts lo usa para elegir modo)
from TTS.api import TTS

MODELO = "tts_models/multilingual/multi-dataset/xtts_v2"
FRECUENCIA = 24000


class MotorXTTS:
    """Carga XTTS-v2 una vez y cachea los *latents* de cada muestra de referencia
    (calcularlos cuesta ~0,5 s; con caché cada frase solo paga la inferencia)."""

    def __init__(self, dispositivo: str | None = None):
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        inicio = time.perf_counter()
        # La primera descarga del modelo pide aceptar la licencia CPML (uso no comercial).
        self.modelo = TTS(MODELO).to(self.dispositivo).synthesizer.tts_model
        self._latents: dict[tuple[str, float], tuple] = {}
        self._lock = threading.Lock()  # una inferencia a la vez en la GPU
        print(f"[xtts] modelo cargado en {self.dispositivo} en {time.perf_counter() - inicio:.1f} s")

    def _condicionamiento(self, referencia: str):
        clave = (os.path.abspath(referencia), os.path.getmtime(referencia))
        if clave not in self._latents:
            self._latents[clave] = self.modelo.get_conditioning_latents(audio_path=[referencia])
        return self._latents[clave]

    def sintetizar(self, texto: str, referencia: str, velocidad: float = 1.0,
                   idioma: str = "es") -> np.ndarray:
        """Devuelve float32 mono a 24 kHz con la voz de `referencia`."""
        with self._lock, torch.inference_mode():
            gpt_latent, speaker_emb = self._condicionamiento(referencia)
            salida = self.modelo.inference(
                texto, idioma, gpt_latent, speaker_emb,
                temperature=0.65, speed=max(0.5, min(velocidad, 2.0)),
                enable_text_splitting=True,
            )
        return np.asarray(salida["wav"], dtype=np.float32)


def _a_wav(muestras: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FRECUENCIA)
        w.writeframes((np.clip(muestras, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def servir(host: str = "127.0.0.1", puerto: int = 8020) -> None:
    motor = MotorXTTS()

    class Manejador(BaseHTTPRequestHandler):
        def _responder(self, codigo: int, cuerpo: bytes, tipo: str) -> None:
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):  # noqa: N802
            if self.path == "/salud":
                datos = {"ok": True, "dispositivo": motor.dispositivo}
                self._responder(200, json.dumps(datos).encode(), "application/json")
            else:
                self._responder(404, b"no encontrado", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path != "/tts":
                return self._responder(404, b"no encontrado", "text/plain")
            try:
                peticion = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                inicio = time.perf_counter()
                audio = motor.sintetizar(peticion["texto"], peticion["referencia"],
                                         float(peticion.get("velocidad", 1.0)),
                                         peticion.get("idioma", "es"))
                print(f"[xtts] {time.perf_counter() - inicio:.2f} s  {peticion['texto'][:60]!r}")
                self._responder(200, _a_wav(audio), "audio/wav")
            except Exception as e:  # el cliente recibe el error en texto
                self._responder(500, str(e).encode(), "text/plain; charset=utf-8")

        def log_message(self, *args):  # silenciar el log por petición de http.server
            pass

    print(f"[xtts] escuchando en http://{host}:{puerto}")
    ThreadingHTTPServer((host, puerto), Manejador).serve_forever()


if __name__ == "__main__":
    servir(os.environ.get("XTTS_HOST", "127.0.0.1"), int(os.environ.get("XTTS_PUERTO", "8020")))
