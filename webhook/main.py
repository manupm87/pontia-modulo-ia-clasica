"""
Webhook de Dialogflow ES para el asistente de voz de Seguros Alonso (FastAPI).

Arrancar (desde la raíz del repositorio):
    .venv\\Scripts\\activate
    uvicorn webhook.main:app --port 8000

Después: exponer el puerto 8000 (devtunnel / ngrok) y pegar la URL + /webhook en Fulfillment.

Responsabilidades:
- Flujo multi-turno de la cita de peritaje (slot filling con webhook, validación regex,
  confirmación sí/no, corrección de datos a mitad de flujo mediante eventos).
- Consulta documental (RAG sobre el texto OCR) con control de alucinaciones.
- Gestión de interrupciones: si el usuario pregunta algo en mitad del flujo, se responde
  y se retoma el punto en el que estaba.
- Transferencia a agente humano.
- Respuestas en texto, SSML (voz) y payload de Telegram con botones.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from webhook import negocio
from webhook.respuestas import (
    CTX_FOLLOWUP,
    CTX_SOLICITUD,
    EVENTO_SOLICITAR_CITA,
    Frase,
    construir,
    evento,
)

log = logging.getLogger("webhook")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PARAMETROS_CITA = ["matricula", "fecha", "franja", "ciudad", "tipo_dano"]
REQUERIDOS_CITA = ["matricula", "fecha", "franja", "ciudad"]
TELEFONO_ATENCION = "910 555 222"

# El motor documental se carga UNA vez al arrancar: el OCR ya se hizo offline y aquí solo
# se abre el JSON y la colección de ChromaDB. Nunca dentro de la petición.
ESTADO: dict = {"motor": None, "error_motor": None}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    try:
        from webhook.rag import MotorDocumental
        ESTADO["motor"] = MotorDocumental()
        try:
            from webhook.llm import precalentar
            precalentar()  # carga el modelo en memoria para que la primera pregunta no tarde
        except Exception as e:  # el LLM puede no estar disponible al arrancar
            log.warning("No se pudo precalentar el LLM: %s", e)
    except Exception as e:
        ESTADO["error_motor"] = str(e)
        log.error("Motor documental no disponible: %s", e)
    yield


app = FastAPI(title="Webhook Seguros Alonso", lifespan=ciclo_de_vida)


# ---------------------------------------------------------------------------
# Lectura de la petición de Dialogflow
# ---------------------------------------------------------------------------

@dataclass
class Peticion:
    intencion: str
    parametros: dict
    texto: str
    sesion: str
    canal: str
    completos: bool
    contextos: dict[str, dict] = field(default_factory=dict)  # nombre corto -> contexto

    def contexto(self, nombre: str) -> dict | None:
        return self.contextos.get(nombre)

    def params_contexto(self, nombre: str) -> dict:
        """
        Parámetros de un contexto, recuperando la copia de seguridad "guardado_<x>".

        Dialogflow escribe en sus contextos de salida TODOS los parámetros de la intención
        activa, también los vacíos: al activarse corregir_dato con solo la fecha, la matrícula
        del contexto pasa a "". Por eso el webhook guarda además cada dato como guardado_<x>,
        una clave que ninguna intención escribe, y aquí se usa cuando el valor normal está vacío.
        """
        ctx = self.contextos.get(nombre) or {}
        params = dict(ctx.get("parameters", {}) or {})
        for clave in PARAMETROS_CITA:
            if params.get(clave) in (None, "", []) and params.get(f"guardado_{clave}"):
                params[clave] = params[f"guardado_{clave}"]
        return params


def leer_peticion(body: dict) -> Peticion:
    resultado = body.get("queryResult", {})
    original = body.get("originalDetectIntentRequest", {}) or {}
    payload = original.get("payload", {}) or {}
    canal = payload.get("canal") or ("telegram" if original.get("source") == "telegram" else "texto")
    contextos = {}
    for ctx in resultado.get("outputContexts", []) or []:
        nombre_corto = ctx.get("name", "").rsplit("/", 1)[-1]
        # lifespanCount ausente o 0 = contexto que se está cerrando
        if ctx.get("lifespanCount", 0) > 0:
            contextos[nombre_corto] = ctx
    return Peticion(
        intencion=resultado.get("intent", {}).get("displayName", ""),
        parametros=resultado.get("parameters", {}) or {},
        texto=resultado.get("queryText", ""),
        sesion=body.get("session", "sesion-desconocida"),
        canal=canal,
        completos=resultado.get("allRequiredParamsPresent", False),
        contextos=contextos,
    )


def responder(p: Peticion, frase: Frase, **kwargs) -> dict:
    return construir(frase, p.sesion, canal=p.canal, **kwargs)


def solo_parametros_cita(params: dict) -> dict:
    """Quita las claves auxiliares de Dialogflow ('matricula.original', etc.)."""
    return {k: params.get(k) for k in PARAMETROS_CITA if params.get(k) not in (None, "", [])}


PATRON_MATRICULA_EN_TEXTO = re.compile(r"\b(\d{4})\s?([BCDFGHJKLMNPRSTVWXYZ]{3})\b")


def matricula_en_texto(texto: str) -> str | None:
    """
    Busca una matrícula DICTADA en el texto reconocido por el STT.

    El STT de Google transcribe las matrículas de formas que la entidad regex de Dialogflow
    no reconoce: "1234 be ce de", "1.234 BCD", "cinco seis siete ocho efe ge hache". Las
    normalizamos ("1234 BCD") y aplicamos la regex de matrícula sobre el resultado.
    """
    from voz.stt import normalizar_matricula_hablada
    m = PATRON_MATRICULA_EN_TEXTO.search(normalizar_matricula_hablada(texto or ""))
    return f"{m.group(1)} {m.group(2)}" if m else None


# ---------------------------------------------------------------------------
# Bienvenida
# ---------------------------------------------------------------------------

def bienvenida(p: Peticion) -> dict:
    f = Frase()
    f.decir("¡Hola! Soy Clara, la asistente virtual de Seguros Alonso.").pausa(250)
    f.decir("Puedo pedirte cita con un perito para revisar los daños de tu coche, "
            "o resolver dudas sobre tus coberturas y condiciones.").pausa(200)
    f.decir("¿En qué te ayudo?")
    return responder(p, f, botones=["Pedir cita con un perito", "¿Qué cubre el todo riesgo?",
                                    "Hablar con un agente"])


# ---------------------------------------------------------------------------
# Flujo de la cita: slot filling con webhook
# ---------------------------------------------------------------------------

PREGUNTAS = {
    "matricula": "¿Cuál es la matrícula del vehículo?",
    "fecha": "¿Qué día te viene bien para la cita?",
    "franja": "¿Prefieres por la mañana o por la tarde?",
    "ciudad": "¿En qué ciudad quieres que te vea el perito?",
}

# Arranque de frase típico de una pregunta (el STT no siempre pone "¿").
PATRON_PREGUNTA = re.compile(
    r"^\s*¿|\?\s*$|^\s*(qu[eé]|cu[aá]nt[oa]s?|c[oó]mo|cu[aá]l(es)?|cu[aá]ndo|d[oó]nde|"
    r"tengo que|hay que|me cubre|cubre|incluye|necesito llevar)\b",
    re.IGNORECASE,
)


def es_pregunta(texto: str) -> bool:
    return bool(PATRON_PREGUNTA.search(texto or ""))


def siguiente_pendiente(params: dict) -> str | None:
    for nombre in REQUERIDOS_CITA:
        if not params.get(nombre):
            return nombre
    return None


def turno_slot_filling(p: Peticion) -> dict:
    """
    Dialogflow nos llama en cada turno de la recogida de datos ("webhook for slot filling").

    - Guardamos SIEMPRE lo recogido hasta ahora en ctx_solicitud_en_curso. Dialogflow ES real
      abandona el slot filling si la frase encaja mejor con otra intención (p. ej. una pregunta
      que va a consulta_documental) y con él se pierden sus contextos internos. Nuestro contexto
      sobrevive, así que tras la interrupción se puede retomar: consulta_documental pregunta el
      dato pendiente y la respuesta entra por corregir_dato, que relanza SOLICITAR_CITA con todo.
    - Si la pregunta sí llega aquí (emulador local, o Dialogflow no encontró otra intención),
      la respondemos con los documentos y repetimos la pregunta pendiente.
    - Si ya sabemos de quién es la matrícula, personalizamos la siguiente pregunta.
    """
    # Reentrada tras una interrupción: si el contexto guarda datos que esta nueva ejecución de
    # solicitar_cita no tiene, los recuperamos relanzando el evento con todo fusionado.
    anteriores = solo_parametros_cita(p.params_contexto(CTX_SOLICITUD))
    actuales = solo_parametros_cita(p.parametros)
    if any(k not in actuales for k in anteriores):
        return evento(EVENTO_SOLICITAR_CITA, {**anteriores, **actuales})

    # Matrícula dictada que la entidad regex no reconoció ("1234 be ce de"): la rescatamos
    # del texto y reentramos con el evento para que Dialogflow la tenga como parámetro.
    if not actuales.get("matricula") and (dictada := matricula_en_texto(p.texto)):
        return evento(EVENTO_SOLICITAR_CITA, {**anteriores, **actuales, "matricula": dictada})

    pendiente = siguiente_pendiente(p.parametros)
    if pendiente is None:
        return {}
    contextos = [contexto_solicitud(actuales)]

    if es_pregunta(p.texto) and not re.search(r"\d{4}\s?[A-Z]{3}", p.texto, re.I):
        f = respuesta_documental(p.texto)
        f.pausa(300).decir("Seguimos con tu cita.").decir(PREGUNTAS[pendiente])
        return responder(p, f, contextos=contextos)

    valida, matricula = negocio.validar_matricula(p.parametros.get("matricula"))
    poliza = negocio.POLIZAS.get(matricula) if valida else None
    f = Frase()
    if poliza and pendiente != "matricula":
        nombre = poliza["titular"].split()[0]
        if pendiente == "fecha":
            f.decir(f"Gracias, {nombre}. He encontrado tu {poliza['vehiculo']}.").pausa(200)
        else:
            f.decir("Perfecto.")
    f.decir(PREGUNTAS[pendiente])
    return responder(p, f, contextos=contextos)


def resumen_cita(ev: negocio.Evaluacion) -> Frase:
    assert ev.poliza and ev.fecha and ev.franja and ev.taller
    f = Frase()
    nombre = ev.poliza["titular"].split()[0]
    f.decir(f"{nombre}, te resumo la solicitud:").pausa(250)
    f.decir(f"cita de peritaje para tu {ev.poliza['vehiculo']}, matrícula ")
    f.decir(negocio.matricula_legible(ev.matricula) + ",",
            f'<say-as interpret-as="characters">{ev.matricula[:4]}</say-as><break time="150ms"/>'
            f'<say-as interpret-as="characters">{ev.matricula[4:]}</say-as>,')
    f.decir(f"{negocio.fecha_en_texto(ev.fecha)} {negocio.FRANJAS[ev.franja]['nombre']}, "
            f"en {ev.taller['taller']}, {ev.ciudad}.").pausa(300)
    if ev.tipo_dano:
        f.decir(f"Motivo: {negocio.TIPOS_DANO[ev.tipo_dano]}.").pausa(200)
    f.enfasis("¿Confirmo la cita?")
    return f


def contexto_solicitud(params: dict, vida: int = 10) -> tuple[str, int, dict]:
    """Contexto del flujo con los datos y su copia "guardado_<x>" (ver Peticion.params_contexto)."""
    copia = {f"guardado_{k}": v for k, v in params.items() if k in PARAMETROS_CITA}
    return (CTX_SOLICITUD, vida, {**params, **copia})


def solicitar_cita(p: Peticion) -> dict:
    if not p.completos:
        return turno_slot_filling(p)

    params = solo_parametros_cita(p.parametros)
    ev = negocio.evaluar_solicitud(params)

    if ev.valida:
        params["matricula"] = ev.matricula
        return responder(
            p, resumen_cita(ev), botones=["Sí", "No"],
            contextos=[contexto_solicitud(params), (CTX_FOLLOWUP, 2, params)],
        )

    # Hay algo que corregir: lo explicamos y dejamos el contexto abierto para que el
    # usuario lo corrija ("mejor el jueves") sin empezar de cero. Cerramos el follow-up
    # para que un "sí" no confirme una cita inválida.
    problema = ev.problemas[0]
    f = Frase(problema.mensaje).pausa(250)
    botones = None
    if problema.campo == "matricula":
        f.decir("¿Me dices otra vez la matrícula?")
    elif problema.campo == "tipo_dano":
        f.decir("Si es por la luna, dímelo; si no, puedo pasarte con un agente.")
        botones = ["Es por la luna", "Hablar con un agente"]
    elif problema.campo == "fecha":
        f.decir("¿Qué otro día te viene bien?")
    elif problema.campo == "ciudad":
        f.decir("¿En qué otra ciudad te viene bien?")
    else:
        f.decir(PREGUNTAS.get(problema.campo, "¿Qué dato quieres cambiar?"))
    return responder(
        p, f, botones=botones,
        contextos=[contexto_solicitud(params), (CTX_FOLLOWUP, 0, {})],
    )


def parametros_flujo(p: Peticion) -> dict:
    """Datos de la cita en curso: contexto principal, o el follow-up como respaldo."""
    for nombre in (CTX_SOLICITUD, CTX_FOLLOWUP):
        params = solo_parametros_cita(p.params_contexto(nombre))
        if params:
            return params
    return solo_parametros_cita(p.parametros)


def confirmar_si(p: Peticion) -> dict:
    params = parametros_flujo(p)
    ev = negocio.evaluar_solicitud(params)
    if not ev.valida:  # defensa: nunca creamos una cita inválida
        return responder(p, Frase(ev.problemas[0].mensaje + " ¿Qué dato quieres cambiar?"),
                         contextos=[contexto_solicitud(params)])
    cita = negocio.crear_cita(ev)
    f = Frase("¡Hecho!").pausa(200)
    f.codigo("Tu cita es la ", cita["codigo"], ".").pausa(300)
    f.decir(f"Te atenderá {cita['perito']}, del equipo de peritos, {cita['fecha_texto']} a las "
            f"{cita['hora']} en {cita['taller']}, {cita['direccion']}, {cita['ciudad']}.").pausa(250)
    f.lento("Lleva el permiso de circulación, tu DNI y, si lo tienes, el parte amistoso.").pausa(200)
    f.decir("¿Te ayudo con algo más?")
    return responder(
        p, f, payload_extra={"cita": cita},
        contextos=[contexto_solicitud({}, 0), (CTX_FOLLOWUP, 0, {})],
    )


def confirmar_no(p: Peticion) -> dict:
    params = parametros_flujo(p)
    f = Frase("De acuerdo, no la reservo todavía.").pausa(200)
    f.decir("Si quieres cambiar algún dato, dímelo, por ejemplo: mejor el jueves, o por la tarde.")
    f.decir("Si no, la solicitud queda cancelada.")
    return responder(p, f, contextos=[contexto_solicitud(params, 3)])


def corregir_dato(p: Peticion) -> dict:
    """
    Corrección a mitad de flujo mediante EVENTOS: fusionamos el dato nuevo con lo que ya
    había en el contexto y relanzamos la intención solicitar_cita con el evento
    SOLICITAR_CITA. Dialogflow rellena los parámetros desde el evento (#SOLICITAR_CITA.x),
    vuelve a llamarnos, revalidamos todo y se pide de nuevo la confirmación.
    """
    anteriores = parametros_flujo(p) if p.contexto(CTX_SOLICITUD) else {}
    nuevos = solo_parametros_cita(p.parametros)
    if not nuevos.get("matricula") and (dictada := matricula_en_texto(p.texto)):
        nuevos["matricula"] = dictada
    if not nuevos:
        f = Frase("¿Qué dato quieres cambiar: la matrícula, el día, la franja o la ciudad?")
        return responder(p, f, contextos=[contexto_solicitud(anteriores)])
    fusion = {**anteriores, **nuevos}
    log.info("Corrección %s -> relanzo %s con %s", nuevos, EVENTO_SOLICITAR_CITA, fusion)
    return evento(EVENTO_SOLICITAR_CITA, fusion)


# ---------------------------------------------------------------------------
# Consulta documental (RAG) e interrupciones
# ---------------------------------------------------------------------------

def respuesta_documental(pregunta: str) -> Frase:
    motor = ESTADO["motor"]
    if motor is None:
        return Frase("Ahora mismo no puedo consultar la documentación. "
                     "¿Quieres que te pase con un agente?")
    r = motor.responder(pregunta)
    log.info("RAG respondida=%s fuentes=%s %.2fs", r.respondida, r.fuentes, r.latencia)
    return Frase(r.texto)


def retomar_flujo(p: Peticion, f: Frase) -> tuple[Frase, list, list | None]:
    """Si había una cita a medias, la retomamos tras responder a la interrupción."""
    ctx = p.contexto(CTX_SOLICITUD)
    if not ctx:
        return f, [], None
    params = solo_parametros_cita(ctx.get("parameters", {}))
    pendiente = siguiente_pendiente(params)
    f.pausa(350)
    if pendiente:
        f.decir("Seguimos con tu cita.").decir(PREGUNTAS[pendiente])
        return f, [contexto_solicitud(params)], None
    ev = negocio.evaluar_solicitud(params)
    if ev.valida and p.contexto(CTX_FOLLOWUP):
        f.decir("Volviendo a tu cita:").extender(resumen_cita(ev))
        return f, [contexto_solicitud(params), (CTX_FOLLOWUP, 2, params)], ["Sí", "No"]
    f.decir("Seguimos con tu cita: ¿qué dato quieres cambiar?")
    return f, [contexto_solicitud(params)], None


def consulta_documental(p: Peticion) -> dict:
    f, contextos, botones = retomar_flujo(p, respuesta_documental(p.texto))
    return responder(p, f, contextos=contextos or None, botones=botones)


def fallback(p: Peticion) -> dict:
    # Con una cita a medias, una matrícula dictada que Dialogflow no entendió cae aquí.
    if p.contexto(CTX_SOLICITUD) and (dictada := matricula_en_texto(p.texto)):
        return evento(EVENTO_SOLICITAR_CITA, {**parametros_flujo(p), "matricula": dictada})
    # Antes de rendirnos, probamos con los documentos: muchas preguntas libres caen aquí.
    return consulta_documental(p)


# ---------------------------------------------------------------------------
# Transferencia a agente humano
# ---------------------------------------------------------------------------

def transferir_agente(p: Peticion) -> dict:
    datos = parametros_flujo(p) if p.contextos else {}
    ticket = negocio.crear_ticket_agente({**datos, "sesion": p.sesion})
    f = Frase("Claro, te paso con una persona del equipo de atención al cliente.").pausa(250)
    if datos.get("matricula"):
        f.decir("Le dejo anotados los datos que me has dado para que no tengas que repetirlos.")
    f.codigo("Tu número de ticket es el ", ticket, ".").pausa(250)
    f.decir(f"Si se corta, puedes llamar al {TELEFONO_ATENCION}, de lunes a viernes "
            "de 9 a 14 y de 16 a 19 horas.")
    return responder(
        p, f,
        payload_extra={"liveAgentHandoff": True, "ticket": ticket, "resumen": datos},
        contextos=[contexto_solicitud({}, 0), (CTX_FOLLOWUP, 0, {})],
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

MANEJADORES = {
    "Default Welcome Intent": bienvenida,
    "solicitar_cita": solicitar_cita,
    "solicitar_cita - si": confirmar_si,
    "solicitar_cita - no": confirmar_no,
    "corregir_dato": corregir_dato,
    "consulta_documental": consulta_documental,
    "transferir_agente": transferir_agente,
    "Default Fallback Intent": fallback,
}


@app.post("/webhook")
async def webhook(request: Request):
    inicio = time.perf_counter()
    body = await request.json()
    p = leer_peticion(body)
    log.info("INTENCION=%s CANAL=%s TEXTO=%r PARAMS=%s CTX=%s", p.intencion, p.canal, p.texto,
             solo_parametros_cita(p.parametros), list(p.contextos))

    manejador = MANEJADORES.get(p.intencion)
    try:
        respuesta = manejador(p) if manejador else responder(
            p, Frase("Todavía no sé gestionar esa petición. ¿Quieres que te pase con un agente?"))
    except Exception:
        # Nunca dejar al usuario escuchando un silencio
        log.exception("Error en el manejador de %s", p.intencion)
        respuesta = responder(p, Frase("Perdona, he tenido un problema técnico. "
                                       "¿Quieres que te pase con un agente?"))

    duracion = time.perf_counter() - inicio
    log.info("Respuesta en %.2fs", duracion)
    return JSONResponse(respuesta, headers={"X-Tiempo-Webhook": f"{duracion:.3f}"})


@app.get("/")
def raiz():
    """Comprobación rápida de que el túnel llega y de qué hay cargado."""
    motor = ESTADO["motor"]
    return {
        "estado": "ok",
        "motor_documental": motor is not None,
        "error_motor": ESTADO["error_motor"],
        "documentos": getattr(motor, "num_documentos", None),
        "citas_creadas": len(negocio.CITAS),
    }
