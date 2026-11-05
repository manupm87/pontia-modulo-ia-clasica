"""
Demo guiada de extremo a extremo por voz, sin micrófono, que genera audio + vídeo + informe.

Para cada paso del guion (demo/guion.py):
    1. la frase del "cliente" se sintetiza con una voz distinta (edge-tts, es-ES-AlvaroNeural)
    2. ese audio pasa por Whisper (STT)                      -> transcripción real
    3. la transcripción va a Dialogflow (emulador o Google)   -> webhook -> respuesta SSML
    4. la respuesta se sintetiza con la voz del bot (TTS)     -> audio
Todo se encadena en una sola pista y se monta un vídeo con subtítulos (máximo 3 minutos).

Requisito: el webhook arrancado (uvicorn webhook.main:app --port 8000).

    python -m demo.demo_guiada [--salida demo/salida] [--sin-video]
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import time
import uuid
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from demo.guion import GUION
from dialogflow_local.cliente import crear_cliente

load_dotenv()

TASA = 24000
VOZ_USUARIO = "es-ES-AlvaroNeural"


def leer_wav(datos: bytes) -> np.ndarray:
    """WAV -> float32 mono a 24 kHz (remuestreo lineal si hace falta)."""
    with wave.open(io.BytesIO(datos)) as w:
        tasa, canales, ancho = w.getframerate(), w.getnchannels(), w.getsampwidth()
        bruto = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16 if ancho == 2 else np.int32)
    x = bruto.astype(np.float32) / (32768.0 if ancho == 2 else 2**31)
    if canales > 1:
        x = x.reshape(-1, canales).mean(axis=1)
    if tasa != TASA:
        n = int(len(x) * TASA / tasa)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def a_16k(wav: bytes) -> bytes:
    """WAV de cualquier frecuencia -> WAV 16 kHz mono 16 bits (lo que espera el STT de Dialogflow)."""
    x = leer_wav(wav)
    n = int(len(x) * 16000 / TASA)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    salida = io.BytesIO()
    with wave.open(salida, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return salida.getvalue()


def silencio(segundos: float) -> np.ndarray:
    return np.zeros(int(TASA * segundos), dtype=np.float32)


def a_srt(t: float) -> str:
    h, resto = divmod(t, 3600)
    m, s = divmod(resto, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def ejecutar(salida: Path, con_video: bool = True) -> None:
    from voz.stt import transcribir
    from voz.tts import sintetizar

    salida.mkdir(parents=True, exist_ok=True)
    # Con DIALOGFLOW_BACKEND=google se usa el Dialogflow real con su STT y TTS nativos (modo audio):
    # el audio del cliente va directo a detect_intent y la respuesta llega ya sintetizada.
    nativo = os.environ.get("DIALOGFLOW_BACKEND", "local").lower() == "google"
    cliente = crear_cliente(sintetizar=True) if nativo else crear_cliente()
    sesion = f"demo-{uuid.uuid4().hex[:8]}"
    pista: list[np.ndarray] = []
    subtitulos: list[tuple[float, float, str]] = []
    filas: list[dict] = []
    t = 0.0

    escenas: list[dict] = []  # una "diapositiva" del vídeo por cada intervención

    def anadir(audio: np.ndarray, quien: str, texto: str, fila: dict) -> None:
        nonlocal t
        pista.append(audio)
        subtitulos.append((t, t + len(audio) / TASA, f"{quien}: {texto}"))
        escenas.append({"inicio": t, "quien": quien, "texto": texto, "fila": fila})
        t += len(audio) / TASA

    def pausa(seg: float) -> None:
        nonlocal t
        pista.append(silencio(seg))
        t += seg

    for i, paso in enumerate(GUION, 1):
        print(f"\n[{i}/{len(GUION)}] {paso['nota']}")
        fila = {"paso": i, "nota": paso["nota"], "stt": 0.0, "nativo": nativo}
        if "evento" in paso:
            turno = cliente.evento(sesion, paso["evento"], payload={"canal": "voz"})
            fila["dice"] = f"(evento {paso['evento']})"
            fila["entendido"] = ""
        else:
            wav_usuario = sintetizar(paso["dice"], backend="edge", voz=VOZ_USUARIO)
            print(f"  Cliente dice : {paso['dice']}")
            if nativo:
                # STT nativo de Dialogflow: su tiempo va incluido en la llamada (columna DF)
                turno = cliente.audio(sesion, a_16k(wav_usuario), payload={"canal": "voz"})
                entendido = turno.query_text
            else:
                t0 = time.perf_counter()
                entendido = transcribir(wav_usuario)
                fila["stt"] = time.perf_counter() - t0
                turno = cliente.texto(sesion, entendido, payload={"canal": "voz"})
            print(f"  STT oye      : {entendido}")
            anadir(leer_wav(wav_usuario), "Cliente", entendido, fila)
            pausa(0.4)
            fila["dice"], fila["entendido"] = paso["dice"], entendido

        t0 = time.perf_counter()
        wav_bot = turno.audio or sintetizar(turno.ssml or turno.texto)
        fila["tts"] = time.perf_counter() - t0  # ~0 con TTS nativo: el audio llegó con la respuesta
        fila["dialogflow"] = turno.latencias.get("total", turno.latencias.get("dialogflow", 0.0))
        fila["webhook"] = turno.latencias.get("webhook", 0.0)
        fila["total"] = fila["stt"] + fila["dialogflow"] + fila["tts"]
        fila["intent"] = turno.intent
        fila["respuesta"] = turno.texto
        fila["parametros"] = {k: v for k, v in (turno.parametros or {}).items() if v and not k.endswith(".original")}
        fila["contextos"] = [f"{c['name']}({c.get('lifespan', c.get('lifespanCount'))})" for c in turno.contextos]
        print(f"  Intención    : {turno.intent}  {fila['parametros']}")
        print(f"  Clara        : {turno.texto}")
        print(f"  Latencia     : STT {fila['stt']:.2f}s + DF/webhook {fila['dialogflow']:.2f}s "
              f"+ TTS {fila['tts']:.2f}s = {fila['total']:.2f}s")
        anadir(leer_wav(wav_bot), "Clara", turno.texto, fila)
        pausa(0.7)
        filas.append(fila)

    # --- audio completo ---
    audio = np.concatenate(pista)
    ruta_wav = salida / "conversacion.wav"
    with wave.open(str(ruta_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TASA)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"\nAudio: {ruta_wav} ({len(audio) / TASA:.1f} s)")

    # --- subtítulos ---
    srt = "\n".join(f"{n}\n{a_srt(a)} --> {a_srt(b)}\n{texto}\n" for n, (a, b, texto) in enumerate(subtitulos, 1))
    (salida / "conversacion.srt").write_text(srt, encoding="utf-8")

    # --- informe de latencias ---
    lineas = ["# Demo guiada: informe de turnos", "",
              "| # | Paso | STT entendió | Intención | STT | DF+webhook | TTS | Total |",
              "|---|---|---|---|---|---|---|---|"]
    for f in filas:
        lineas.append(f"| {f['paso']} | {f['nota']} | {f['entendido'] or f['dice']} | `{f['intent']}` | "
                      f"{f['stt']:.2f} s | {f['dialogflow']:.2f} s | {f['tts']:.2f} s | **{f['total']:.2f} s** |")
    maximo = max(f["total"] for f in filas)
    medio = sum(f["total"] for f in filas) / len(filas)
    lineas += ["", f"Latencia media por turno: **{medio:.2f} s** · máxima: **{maximo:.2f} s** (objetivo < 5 s)", "",
               "## Respuestas", ""]
    for f in filas:
        lineas.append(f"{f['paso']}. **{f['intent']}** {f['parametros']} → {f['respuesta']}  \n"
                      f"   contextos: {', '.join(f['contextos']) or '—'}")
    (salida / "informe.md").write_text("\n".join(lineas), encoding="utf-8")
    print(f"Informe: {salida / 'informe.md'}  (media {medio:.2f} s, máx {maximo:.2f} s)")

    # --- vídeo: una diapositiva por intervención, sincronizada con el audio ---
    if con_video:
        montar_video(salida, escenas, len(audio) / TASA)
        print(f"Vídeo: {salida / 'demo.mp4'}")

# ---------------------------------------------------------------------------
# Vídeo
# ---------------------------------------------------------------------------

ANCHO, ALTO = 1280, 720
FUENTES = "C:/Windows/Fonts/"
COLORES = {"fondo": (244, 242, 238), "marca": (15, 92, 99), "claro": (217, 236, 236),
           "tinta": (29, 36, 48), "suave": (93, 102, 117), "cliente": (200, 85, 61)}


def _fuente(nombre: str, tam: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FUENTES + nombre, tam)
    except OSError:
        return ImageFont.load_default(tam)


def _envolver(draw, texto: str, fuente, ancho: int) -> list[str]:
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = f"{actual} {palabra}".strip()
        if draw.textlength(prueba, font=fuente) <= ancho:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    return lineas + ([actual] if actual else [])


def diapositiva(escena: dict, n_pasos: int):
    """Imagen 1280x720 de una intervención: quién habla, qué dice y qué pasa por dentro."""
    from PIL import Image, ImageDraw

    f = escena["fila"]
    img = Image.new("RGB", (ANCHO, ALTO), COLORES["fondo"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, ANCHO, 92], fill=COLORES["marca"])
    d.text((48, 18), "Seguros Alonso · Clara, asistente de voz", font=_fuente("segoeuib.ttf", 30), fill="white")
    subtitulo = ("Dialogflow ES (Google) · STT y TTS nativos con SSML · Webhook FastAPI · OCR + RAG"
                 if f.get("nativo") else
                 "Emulador de Dialogflow ES · Whisper + TTS neuronal con SSML · Webhook FastAPI · OCR + RAG")
    d.text((48, 56), subtitulo, font=_fuente("segoeui.ttf", 18), fill=COLORES["claro"])
    d.text((ANCHO - 48, 34), f"paso {f['paso']}/{n_pasos}", font=_fuente("segoeui.ttf", 20),
           fill="white", anchor="ra")

    # Qué se está demostrando
    d.rounded_rectangle([48, 118, ANCHO - 48, 168], radius=24, fill=COLORES["claro"])
    d.text((72, 131), f["nota"], font=_fuente("segoeuib.ttf", 20), fill=COLORES["marca"])

    # Burbuja de quien habla
    es_cliente = escena["quien"] == "Cliente"
    color = COLORES["cliente"] if es_cliente else COLORES["marca"]
    cuerpo = _fuente("segoeui.ttf", 30)
    lineas = _envolver(d, escena["texto"], cuerpo, ANCHO - 260)[:7]
    alto = 70 + 44 * len(lineas)
    x0, x1 = (200, ANCHO - 48) if es_cliente else (48, ANCHO - 200)
    y0 = 200
    d.rounded_rectangle([x0, y0, x1, y0 + alto], radius=22, fill="white", outline=color, width=3)
    if f.get("nativo"):
        etiqueta = ("Cliente (voz sintética → STT de Dialogflow)" if es_cliente
                    else "Clara (TTS nativo de Dialogflow con SSML)")
    else:
        etiqueta = "Cliente (voz sintética → Whisper)" if es_cliente else "Clara (TTS con SSML)"
    d.text((x0 + 28, y0 + 16), etiqueta, font=_fuente("segoeuib.ttf", 20), fill=color)
    for i, linea in enumerate(lineas):
        d.text((x0 + 28, y0 + 52 + 44 * i), linea, font=cuerpo, fill=COLORES["tinta"])

    # Panel interno: intención, parámetros, latencia
    mono = _fuente("consola.ttf", 19)
    y = ALTO - 120
    d.line([48, y - 16, ANCHO - 48, y - 16], fill=(226, 221, 211), width=2)
    params = ", ".join(f"{k}={v}" for k, v in f.get("parametros", {}).items())
    d.text((48, y), f"intención: {f.get('intent') or '—'}", font=mono, fill=COLORES["marca"])
    d.text((48, y + 30), f"parámetros: {params or '—'}"[:110], font=mono, fill=COLORES["suave"])
    d.text((48, y + 60), (f"latencia del turno: STT {f['stt'] * 1000:.0f} ms + Dialogflow/webhook {f['dialogflow'] * 1000:.0f} ms"
                          f" + TTS {f['tts'] * 1000:.0f} ms = {f['total']:.2f} s"), font=mono, fill=COLORES["suave"])
    return img


def montar_video(salida: Path, escenas: list[dict], duracion: float) -> None:
    """Renderiza las diapositivas con PIL y las une con el demuxer concat de ffmpeg."""
    carpeta = salida / "diapositivas"
    carpeta.mkdir(exist_ok=True)
    n_pasos = max(e["fila"]["paso"] for e in escenas)
    lista = []
    for i, escena in enumerate(escenas):
        fin = escenas[i + 1]["inicio"] if i + 1 < len(escenas) else duracion
        nombre = f"{i:03d}.png"
        diapositiva(escena, n_pasos).save(carpeta / nombre)
        lista += [f"file '{nombre}'", f"duration {fin - escena['inicio']:.3f}"]
    lista.append(f"file '{nombre}'")  # el demuxer concat necesita repetir la última
    (carpeta / "lista.txt").write_text(chr(10).join(lista), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", "diapositivas/lista.txt", "-i", "conversacion.wav",
                    "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "demo.mp4"], cwd=salida, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", type=Path, default=Path("demo/salida"))
    parser.add_argument("--sin-video", action="store_true")
    args = parser.parse_args()
    ejecutar(args.salida, con_video=not args.sin_video)
