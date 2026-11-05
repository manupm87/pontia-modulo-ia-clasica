"""Cliente común de Dialogflow ES con dos backends intercambiables.

* ``local``  -> :class:`ClienteLocal`: emulador en memoria (``dialogflow_local.emulador``) que
  llama al webhook real por HTTP (``WEBHOOK_URL``) o en proceso (app ASGI / función).
* ``google`` -> :class:`ClienteGoogle`: API real ``detectIntent`` (``google-cloud-dialogflow``)
  sobre el proyecto ``DIALOGFLOW_PROJECT_ID``; soporta audio (STT y TTS nativos).

Los dos devuelven un :class:`Turno` con la misma forma, así el resto del proyecto (cliente de
voz, bot de Telegram, demo, tests) no sabe qué backend hay debajo::

    cliente = crear_cliente()                       # DIALOGFLOW_BACKEND=local|google
    turno = cliente.texto("sesion-1", "quiero una cita con el perito", {"canal": "voz"})
    print(turno.intent, turno.texto, turno.latencias)
"""

from __future__ import annotations

import html
import inspect
import io
import os
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

IDIOMA = "es"
WEBHOOK_URL_POR_DEFECTO = "http://127.0.0.1:8000/webhook"


# =============================================================================
# Resultado de un turno
# =============================================================================

@dataclass
class Turno:
    query_text: str                       # lo que entendió (transcripción si fue audio)
    intent: str                           # displayName ("" si nada)
    confianza: float
    parametros: dict
    contextos: list[dict]                 # [{"name": "ctx_solicitud_en_curso", "lifespan": 9, "parameters": {...}}]
    texto: str                            # respuesta en texto plano (sin etiquetas SSML)
    ssml: str | None                      # SSML si la respuesta lo trae
    mensajes: list[dict]                  # fulfillmentMessages crudos
    audio: bytes | None = None            # WAV de respuesta si lo generó Dialogflow (backend google)
    latencias: dict[str, float] = field(default_factory=dict)   # {"dialogflow", "webhook", "total"} en s


def ssml_a_texto(ssml: str) -> str:
    """Quita las etiquetas SSML y deshace las entidades XML (``&amp;`` -> ``&``)."""
    sin_etiquetas = re.sub(r"<[^>]+>", " ", ssml)
    return " ".join(html.unescape(sin_etiquetas).split())


def turno_desde_respuesta(respuesta: dict, audio: bytes | None = None,
                          latencias: dict[str, float] | None = None) -> Turno:
    """Convierte un DetectIntentResponse (JSON v2, local o de Google) en un :class:`Turno`."""
    qr = respuesta.get("queryResult", {}) or {}
    mensajes = list(qr.get("fulfillmentMessages", []) or [])
    cumplimiento = qr.get("fulfillmentText", "") or ""

    # SSML: el fulfillmentText si empieza por <speak>, o el campo "ssml" de un payload.
    ssml = cumplimiento if cumplimiento.lstrip().startswith("<speak") else None
    if ssml is None:
        for m in mensajes:
            posible = (m.get("payload") or {}).get("ssml")
            if isinstance(posible, str) and posible.strip():
                ssml = posible
                break

    # Texto plano: el primer mensaje de texto (sin plataforma); si no, el fulfillmentText sin SSML.
    texto = ""
    for m in mensajes:
        if "text" in m and not m.get("platform") and (m["text"].get("text") or []):
            texto = " ".join(m["text"]["text"])
            break
    if not texto:
        texto = ssml_a_texto(cumplimiento) if cumplimiento.lstrip().startswith("<") else cumplimiento
    if texto.lstrip().startswith("<speak"):
        texto = ssml_a_texto(texto)

    contextos = []
    for c in qr.get("outputContexts", []) or []:
        contextos.append({
            "name": c.get("name", "").rsplit("/contexts/", 1)[-1],
            "lifespan": int(c.get("lifespanCount", 0) or 0),
            "parameters": dict(c.get("parameters", {}) or {}),
        })
    return Turno(
        query_text=qr.get("queryText", ""),
        intent=(qr.get("intent") or {}).get("displayName", ""),
        confianza=float(qr.get("intentDetectionConfidence", 0.0) or 0.0),
        parametros=dict(qr.get("parameters", {}) or {}),
        contextos=contextos,
        texto=texto,
        ssml=ssml,
        mensajes=mensajes,
        audio=audio,
        latencias=latencias or {},
    )


# =============================================================================
# Interfaz común
# =============================================================================

class ClienteDialogflow:
    """Interfaz común de los backends."""

    def texto(self, session_id: str, texto: str, payload: dict | None = None) -> Turno:
        raise NotImplementedError

    def evento(self, session_id: str, nombre: str, parametros: dict | None = None,
               payload: dict | None = None) -> Turno:
        raise NotImplementedError

    def audio(self, session_id: str, wav_bytes: bytes, payload: dict | None = None) -> Turno:
        raise NotImplementedError


# =============================================================================
# Backend local (emulador)
# =============================================================================

def _es_app_asgi(objeto: Any) -> bool:
    """¿Es una aplicación ASGI (FastAPI/Starlette) en lugar de una función normal?"""
    try:
        from starlette.applications import Starlette
        if isinstance(objeto, Starlette):
            return True
    except ImportError:  # pragma: no cover
        pass
    llamada = objeto if inspect.isfunction(objeto) or inspect.ismethod(objeto) else getattr(objeto, "__call__", None)
    if llamada is None or not inspect.iscoroutinefunction(llamada):
        return False
    try:
        return len(inspect.signature(llamada).parameters) == 3     # (scope, receive, send)
    except (TypeError, ValueError):
        return False


def preparar_webhook(webhook: Any = None, url: str | None = None) -> Callable[[dict], dict]:
    """Devuelve una función ``peticion(dict) -> respuesta(dict)`` para el emulador.

    * ``webhook`` es una app ASGI (p. ej. ``webhook.main.app``): se llama en proceso con
      ``fastapi.testclient.TestClient`` (útil en tests, sin levantar uvicorn).
    * ``webhook`` es una función ``dict -> dict``: se usa tal cual.
    * Si no se pasa nada: POST HTTP a ``url`` / ``WEBHOOK_URL`` (por defecto 127.0.0.1:8000; en Windows "localhost" prueba antes IPv6 y añade ~2 s).
    """
    url = url or os.environ.get("WEBHOOK_URL", WEBHOOK_URL_POR_DEFECTO)
    if webhook is not None and _es_app_asgi(webhook):
        from fastapi.testclient import TestClient
        cliente_http = TestClient(webhook)
        ruta = urlparse(url).path or "/webhook"

        def llamar_asgi(peticion: dict) -> dict:
            r = cliente_http.post(ruta, json=peticion)
            r.raise_for_status()
            return r.json()
        return llamar_asgi
    if webhook is not None:
        return webhook

    import httpx
    sesion_http = httpx.Client(timeout=5.0, headers={"X-Tunnel-Skip-AntiPhishing-Page": "true"})

    def llamar_http(peticion: dict) -> dict:
        r = sesion_http.post(url, json=peticion)
        r.raise_for_status()
        return r.json()
    return llamar_http


class ClienteLocal(ClienteDialogflow):
    """Backend local: emulador de Dialogflow ES + webhook real."""

    def __init__(self, webhook: Any = None, ruta_agente: str | None = None, url_webhook: str | None = None,
                 ahora: Any = None, **opciones):
        from dialogflow_local.emulador import RUTA_AGENTE_POR_DEFECTO, EmuladorDialogflow
        self.emulador = EmuladorDialogflow(
            ruta_agente or RUTA_AGENTE_POR_DEFECTO,
            webhook=preparar_webhook(webhook, url_webhook),
            ahora=ahora, **opciones,
        )

    def _turno(self, respuesta: dict, t0: float) -> Turno:
        total = time.perf_counter() - t0
        webhook = respuesta.get("_emulador", {}).get("latencia_webhook", 0.0)
        return turno_desde_respuesta(respuesta, latencias={
            "dialogflow": max(total - webhook, 0.0), "webhook": webhook, "total": total})

    def texto(self, session_id: str, texto: str, payload: dict | None = None) -> Turno:
        t0 = time.perf_counter()
        return self._turno(self.emulador.detectar_texto(session_id, texto, payload), t0)

    def evento(self, session_id: str, nombre: str, parametros: dict | None = None,
               payload: dict | None = None) -> Turno:
        t0 = time.perf_counter()
        return self._turno(self.emulador.detectar_evento(session_id, nombre, parametros, payload), t0)

    def audio(self, session_id: str, wav_bytes: bytes, payload: dict | None = None) -> Turno:
        """En local no hay STT de Dialogflow: se transcribe con faster-whisper (voz.stt)."""
        from voz.stt import transcribir
        t0 = time.perf_counter()
        transcripcion = transcribir(wav_bytes, idioma=IDIOMA)
        t_stt = time.perf_counter() - t0
        turno = self.texto(session_id, transcripcion, payload)
        turno.latencias["stt"] = t_stt
        turno.latencias["total"] = time.perf_counter() - t0
        return turno


# =============================================================================
# Backend Google (API real)
# =============================================================================

# Pistas para el STT de Google: nombres de las consonantes que puede llevar una matrícula,
# las matrículas de ejemplo y vocabulario del dominio.
PISTAS_STT = ["be", "ce", "de", "efe", "ge", "hache", "jota", "ka", "ele", "eme", "ene", "pe", "erre",
              "ese", "te", "uve", "uve doble", "equis", "i griega", "zeta",
              "1234 BCD", "5678 FGH", "9012 JKL", "peritaje", "perito", "franquicia", "póliza"]


class ClienteGoogle(ClienteDialogflow):
    """Backend real: ``SessionsClient.detect_intent`` de google-cloud-dialogflow (ES v2).

    Requiere ``GOOGLE_APPLICATION_CREDENTIALS`` (clave JSON de una cuenta de servicio con el rol
    "Dialogflow API Client") y ``DIALOGFLOW_PROJECT_ID``.
    """

    def __init__(self, proyecto: str | None = None, voz: str | None = None, sintetizar: bool = False):
        from google.cloud import dialogflow
        self.df = dialogflow
        self.proyecto = proyecto or os.environ.get("DIALOGFLOW_PROJECT_ID", "")
        if not self.proyecto:
            raise RuntimeError("Falta DIALOGFLOW_PROJECT_ID para el backend google")
        self.voz = voz or os.environ.get("DIALOGFLOW_VOZ", "es-ES-Standard-A")
        self.sintetizar = sintetizar          # si True, también pide audio en texto() y evento()
        self.sesiones = dialogflow.SessionsClient()

    # ------------------------------------------------------------------ utilidades
    def _parametros_consulta(self, payload: dict | None):
        if not payload:
            return None
        from google.protobuf import struct_pb2
        estructura = struct_pb2.Struct()
        estructura.update(payload)
        return self.df.QueryParameters(payload=estructura)

    def _config_audio_salida(self):
        """TTS nativo de Dialogflow: WAV LINEAR16 con una voz es-ES (interpreta SSML)."""
        return self.df.OutputAudioConfig(
            audio_encoding=self.df.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16,
            synthesize_speech_config=self.df.SynthesizeSpeechConfig(
                voice=self.df.VoiceSelectionParams(name=self.voz)),
        )

    def _detectar(self, session_id: str, query_input, payload: dict | None, input_audio: bytes | None = None,
                  con_audio: bool = False) -> Turno:
        from google.protobuf.json_format import MessageToDict
        peticion = {"session": self.sesiones.session_path(self.proyecto, session_id), "query_input": query_input}
        parametros = self._parametros_consulta(payload)
        if parametros is not None:
            peticion["query_params"] = parametros
        if input_audio is not None:
            peticion["input_audio"] = input_audio
        if con_audio:
            peticion["output_audio_config"] = self._config_audio_salida()
        t0 = time.perf_counter()
        respuesta = self.sesiones.detect_intent(request=peticion)
        total = time.perf_counter() - t0
        datos = MessageToDict(type(respuesta).pb(respuesta), preserving_proto_field_name=False)
        # Dialogflow informa de la latencia del webhook en diagnosticInfo (si la hay).
        diag = (datos.get("queryResult", {}).get("diagnosticInfo") or {})
        webhook = float(diag.get("webhook_latency_ms", 0) or 0) / 1000
        return turno_desde_respuesta(datos, audio=bytes(respuesta.output_audio) or None, latencias={
            "dialogflow": max(total - webhook, 0.0), "webhook": webhook, "total": total})

    # ------------------------------------------------------------------ API
    def texto(self, session_id: str, texto: str, payload: dict | None = None) -> Turno:
        entrada = self.df.QueryInput(text=self.df.TextInput(text=texto, language_code=IDIOMA))
        return self._detectar(session_id, entrada, payload, con_audio=self.sintetizar)

    def evento(self, session_id: str, nombre: str, parametros: dict | None = None,
               payload: dict | None = None) -> Turno:
        from google.protobuf import struct_pb2
        estructura = struct_pb2.Struct()
        estructura.update(parametros or {})
        entrada = self.df.QueryInput(event=self.df.EventInput(name=nombre, parameters=estructura,
                                                              language_code=IDIOMA))
        return self._detectar(session_id, entrada, payload, con_audio=self.sintetizar)

    def audio(self, session_id: str, wav_bytes: bytes, payload: dict | None = None) -> Turno:
        """Audio WAV (PCM 16 bits mono) -> STT de Dialogflow -> intención -> TTS de Dialogflow."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            frecuencia, ancho = w.getframerate(), w.getsampwidth()
            pcm = w.readframes(w.getnframes())
        # Medio segundo de silencio al final: sin él, el STT corta la última sílaba
        # (comprobado: "1234 BCD" se transcribía como "1234 be ce").
        pcm += bytes(int(frecuencia * 0.5) * ancho)
        entrada = self.df.QueryInput(audio_config=self.df.InputAudioConfig(
            audio_encoding=self.df.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            language_code=IDIOMA, sample_rate_hertz=frecuencia,
            # Adaptación del STT: las letras deletreadas de las matrículas (sin vocales)
            # y las matrículas de ejemplo. Sin esto Google descarta las letras.
            speech_contexts=[self.df.SpeechContext(phrases=PISTAS_STT, boost=15)]))
        return self._detectar(session_id, entrada, payload, input_audio=pcm, con_audio=True)


# =============================================================================
# Fábrica
# =============================================================================

def crear_cliente(backend: str | None = None, **opciones) -> ClienteDialogflow:
    """Crea el cliente del backend indicado (o ``DIALOGFLOW_BACKEND``, por defecto ``local``)."""
    backend = (backend or os.environ.get("DIALOGFLOW_BACKEND", "local")).strip().lower()
    if backend == "local":
        return ClienteLocal(**opciones)
    if backend == "google":
        return ClienteGoogle(**opciones)
    raise ValueError(f"Backend de Dialogflow desconocido: {backend!r} (usa 'local' o 'google')")
