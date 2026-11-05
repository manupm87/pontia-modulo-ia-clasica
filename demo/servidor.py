"""
Servidor de la demo web: habla con el asistente desde el navegador (micrófono o texto).

    uvicorn demo.servidor:app --port 8080      ->  http://localhost:8080

Flujo de un turno de voz:
    micrófono (webm) -> Whisper (STT) -> Dialogflow (emulador local o Google) -> webhook
    -> respuesta SSML -> TTS -> audio que reproduce el navegador

Muestra en pantalla la transcripción, la intención, los parámetros, los contextos activos y
la latencia de cada etapa (requisito: < 5 s por turno).

La "demo guiada" sintetiza las frases del usuario con otra voz y las manda por el mismo
camino de audio, para poder enseñar la conversación de voz completa sin micrófono.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo.guion import GUION
from dialogflow_local.cliente import Turno, crear_cliente

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
ESTATICOS = Path(__file__).resolve().parent / "static"
VOZ_USUARIO = "es-ES-AlvaroNeural"  # voz "del cliente" en la demo guiada (distinta a la del bot)

app = FastAPI(title="Demo Seguros Alonso")
cliente = crear_cliente()


def turno_a_json(turno: Turno, stt: float | None = None, transcripcion: str | None = None) -> dict:
    """Serializa el turno y genera el audio de respuesta (TTS) midiendo cada etapa."""
    from voz.tts import sintetizar

    latencias = dict(turno.latencias or {})
    if stt is not None:
        latencias["stt"] = stt
    audio_b64 = None
    if turno.texto:
        t0 = time.perf_counter()
        wav = turno.audio or sintetizar(turno.ssml or turno.texto)
        latencias["tts"] = time.perf_counter() - t0
        audio_b64 = base64.b64encode(wav).decode()
    # "total" del cliente = Dialogflow + webhook; el turno completo suma además STT y TTS
    latencias["total_turno"] = (latencias.get("stt", 0)
                                + latencias.get("total", latencias.get("dialogflow", 0))
                                + latencias.get("tts", 0))
    botones = []
    for m in turno.mensajes or []:
        botones = (m.get("payload") or {}).get("botones") or botones
    return {
        "transcripcion": transcripcion if transcripcion is not None else turno.query_text,
        "intent": turno.intent,
        "confianza": round(turno.confianza or 0, 2),
        "parametros": turno.parametros,
        "contextos": turno.contextos,
        "texto": turno.texto,
        "ssml": turno.ssml,
        "botones": botones,
        "mensajes": turno.mensajes,
        "audio": audio_b64,
        "latencias": {k: round(v, 3) for k, v in latencias.items()},
    }


class EntradaTexto(BaseModel):
    sesion: str
    texto: str


class EntradaEvento(BaseModel):
    sesion: str
    nombre: str


@app.post("/api/texto")
def api_texto(e: EntradaTexto):
    return turno_a_json(cliente.texto(e.sesion, e.texto, payload={"canal": "voz"}))


@app.post("/api/evento")
def api_evento(e: EntradaEvento):
    return turno_a_json(cliente.evento(e.sesion, e.nombre, payload={"canal": "voz"}))


@app.post("/api/audio")
async def api_audio(sesion: str = Form(...), audio: UploadFile = File(...)):
    from voz.stt import transcribir

    datos = await audio.read()
    t0 = time.perf_counter()
    texto = transcribir(datos)
    stt = time.perf_counter() - t0
    if not texto.strip():
        return JSONResponse({"error": "No se ha entendido nada en el audio"}, status_code=422)
    turno = cliente.texto(sesion, texto, payload={"canal": "voz"})
    return turno_a_json(turno, stt=stt, transcripcion=texto)


@app.get("/api/guion")
def api_guion():
    return GUION


@app.post("/api/voz_usuario")
def api_voz_usuario(e: EntradaTexto):
    """Sintetiza una frase del guion con la voz del 'cliente' (para la demo guiada)."""
    from voz.tts import sintetizar

    wav = sintetizar(e.texto, backend="edge", voz=VOZ_USUARIO)
    return {"audio": base64.b64encode(wav).decode()}


@app.get("/api/documentos")
def api_documentos():
    """Resultado del OCR offline: documentos, campos extraídos con regex y avisos."""
    ruta = RAIZ / "data" / "documentos.json"
    if not ruta.exists():
        return []
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if isinstance(datos, dict):
        datos = datos.get("documentos", list(datos.values()))
    return [{k: d.get(k) for k in ("documento", "motor", "caracteres", "segundos", "campos", "avisos")}
            | {"extracto": (d.get("texto") or "")[:600]} for d in datos]


app.mount("/documentos", StaticFiles(directory=RAIZ / "documentos", check_dir=False), name="documentos")
app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")


@app.get("/")
def portada():
    return FileResponse(ESTATICOS / "index.html")
