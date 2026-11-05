"""Tests del emulador local de Dialogflow ES con un webhook falso (no usa el webhook real).

El webhook falso imita lo imprescindible del real:
  * slot filling: devuelve texto vacío -> el emulador usa el prompt del agente;
  * solicitar_cita completa: pide confirmación con los datos;
  * corregir_dato: fusiona los parámetros con ctx_solicitud_en_curso y reenvía el evento
    SOLICITAR_CITA (followupEventInput), como dice docs/DISENO.md.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI, Request

from dialogflow_local.cliente import ClienteLocal, Turno, turno_desde_respuesta
from dialogflow_local.emulador import EmuladorDialogflow

MADRID = ZoneInfo("Europe/Madrid")
AHORA = datetime(2026, 11, 5, 10, 0, tzinfo=MADRID)        # jueves 5 de noviembre de 2026
LUNES = "2026-11-09T12:00:00+01:00"
MANANA = "2026-11-06T12:00:00+01:00"
JUEVES = "2026-11-12T12:00:00+01:00"


# ----------------------------------------------------------------------------- webhook falso
def webhook_falso(peticion: dict) -> dict:
    qr = peticion["queryResult"]
    intent = qr["intent"]["displayName"]
    p = qr["parameters"]
    webhook_falso.peticiones.append(peticion)
    if intent == "Default Welcome Intent":
        return {"fulfillmentText": "Bienvenido a Seguros Alonso."}
    if intent == "solicitar_cita":
        if not qr["allRequiredParamsPresent"]:
            return {"fulfillmentText": ""}        # que el emulador use el prompt
        return {"fulfillmentText": f"¿Confirmo la cita para {p['matricula']} el {p['fecha'][:10]} "
                                   f"por la {p['franja']} en {p['ciudad']}?"}
    if intent == "solicitar_cita - si":
        return {"fulfillmentText": "Cita confirmada: PER-12345."}
    if intent == "corregir_dato":
        # Como en Dialogflow real, corregir_dato deja vacíos en ctx_solicitud_en_curso los
        # parámetros que no trae; el follow-up conserva los datos completos del resumen.
        anteriores = {}
        for sufijo in ("solicitar_cita-followup", "ctx_solicitud_en_curso"):
            for c in qr["outputContexts"]:
                if c["name"].endswith(f"/contexts/{sufijo}"):
                    anteriores.update({k: v for k, v in c["parameters"].items()
                                       if v and not k.endswith(".original")})
        fusion = {**anteriores, **{k: v for k, v in p.items() if v}}
        return {"followupEventInput": {"name": "SOLICITAR_CITA", "parameters": fusion, "languageCode": "es"}}
    if intent == "Default Fallback Intent":
        return {"fulfillmentText": "No consta en la documentación."}
    return {"fulfillmentText": f"[{intent}]"}


webhook_falso.peticiones = []


@pytest.fixture
def emu():
    webhook_falso.peticiones.clear()
    return EmuladorDialogflow(webhook=webhook_falso, ahora=AHORA, semilla=1)


def turno(respuesta: dict) -> Turno:
    return turno_desde_respuesta(respuesta)


def contextos(t: Turno) -> dict[str, dict]:
    return {c["name"]: c for c in t.contextos}


# ----------------------------------------------------------------------------- tests
def test_bienvenida_por_evento(emu):
    t = turno(emu.detectar_evento("s1", "WELCOME"))
    assert t.intent == "Default Welcome Intent"
    assert t.texto == "Bienvenido a Seguros Alonso."


def test_slot_filling_secuencia(emu):
    t = turno(emu.detectar_texto("s1", "quiero que un perito vea el golpe de mi coche"))
    assert t.intent == "solicitar_cita"
    assert t.parametros["tipo_dano"] == "chapa"
    assert t.texto == "¿Cuál es la matrícula del vehículo?"
    assert "solicitar_cita_dialog_context" in contextos(t)

    t = turno(emu.detectar_texto("s1", "es el 1234 BCD"))
    assert t.parametros["matricula"] == "1234 BCD"
    assert t.texto == "¿Qué día te viene bien para la cita?"

    t = turno(emu.detectar_texto("s1", "el lunes"))
    assert t.parametros["fecha"] == LUNES
    assert t.texto == "¿Prefieres por la mañana o por la tarde?"

    t = turno(emu.detectar_texto("s1", "tarde"))                 # respuesta corta al prompt
    assert t.parametros["franja"] == "tarde"
    assert t.texto == "¿En qué ciudad quieres que te vea el perito?"

    t = turno(emu.detectar_texto("s1", "en Valencia"))
    assert t.parametros["ciudad"] == "Valencia"
    assert t.texto.startswith("¿Confirmo la cita para 1234 BCD el 2026-11-09 por la tarde en Valencia")
    ctx = contextos(t)
    assert ctx["ctx_solicitud_en_curso"]["lifespan"] == 10
    assert ctx["ctx_solicitud_en_curso"]["parameters"]["matricula"] == "1234 BCD"
    assert ctx["solicitar_cita-followup"]["lifespan"] == 2
    assert "solicitar_cita_dialog_context" not in ctx

    # El webhook se llamó en cada turno de slot filling, con allRequiredParamsPresent=False
    estados = [p["queryResult"]["allRequiredParamsPresent"] for p in webhook_falso.peticiones]
    assert estados == [False, False, False, False, True]


def test_respuesta_vacia_del_webhook_usa_prompt_y_reintenta(emu):
    emu.detectar_texto("s1", "quiero pedir una cita de peritaje")
    t = turno(emu.detectar_texto("s1", "pues no sé qué decirte"))
    assert t.intent == "solicitar_cita"
    assert t.texto == "¿Cuál es la matrícula del vehículo?"     # se repite el prompt


def test_frase_completa_de_una_vez(emu):
    t = turno(emu.detectar_texto(
        "s1", "necesito cita para peritar el 1234 BCD en Madrid el lunes por la mañana"))
    assert t.intent == "solicitar_cita"
    assert t.parametros == {"matricula": "1234 BCD", "fecha": LUNES, "franja": "manana",
                            "ciudad": "Madrid", "tipo_dano": ""}
    assert t.texto.startswith("¿Confirmo la cita")
    assert webhook_falso.peticiones[-1]["queryResult"]["allRequiredParamsPresent"] is True


@pytest.mark.parametrize("texto, fecha, franja", [
    ("mañana por la tarde", MANANA, "tarde"),
    ("el lunes por la mañana", LUNES, "manana"),
    ("por la mañana", None, "manana"),
    ("pasado mañana", "2026-11-07T12:00:00+01:00", None),
    ("el próximo jueves", JUEVES, None),
    ("15 de noviembre", "2026-11-15T12:00:00+01:00", None),
    ("el día 20", "2026-11-20T12:00:00+01:00", None),
    ("20/11", "2026-11-20T12:00:00+01:00", None),
    ("el 3 de enero", "2027-01-03T12:00:00+01:00", None),
])
def test_fechas_y_franjas(emu, texto, fecha, franja):
    encontradas = {c.tipo: c.valor for c in emu.extractor.extraer(texto, AHORA)}
    assert encontradas.get("@sys.date") == fecha
    assert encontradas.get("@franja") == franja


def test_entidad_regex_y_ciudad(emu):
    encontradas = {c.tipo: c.valor for c in emu.extractor.extraer("el 9012jkl en a coruña", AHORA)}
    assert encontradas == {"@matricula": "9012JKL", "@sys.geo-city": "A Coruña"}
    # Las vocales no forman parte de una matrícula actual
    assert emu.extractor.extraer("el 1234 ABE", AHORA, tipos={"@matricula"}) == []


def test_followup_si_solo_tras_el_padre(emu):
    # Sin contexto de follow-up, "sí" no es la confirmación
    t = turno(emu.detectar_texto("s1", "sí"))
    assert t.intent == "Default Fallback Intent"

    emu.detectar_texto("s2", "necesito cita para peritar el 1234 BCD en Madrid el lunes por la mañana")
    t = turno(emu.detectar_texto("s2", "sí, adelante"))
    assert t.intent == "solicitar_cita - si"
    assert t.texto == "Cita confirmada: PER-12345."
    # El agente deja el contexto vivo (lifespan 2) para que sus parámetros lleguen al webhook;
    # es el webhook real quien lo cierra al crear la cita (este webhook falso no lo hace).
    assert contextos(t)["ctx_solicitud_en_curso"]["lifespan"] == 2


def test_followup_no(emu):
    emu.detectar_texto("s1", "necesito cita para peritar el 1234 BCD en Madrid el lunes por la mañana")
    t = turno(emu.detectar_texto("s1", "no, espera"))
    assert t.intent == "solicitar_cita - no"
    assert contextos(t)["ctx_solicitud_en_curso"]["lifespan"] == 3


def test_corregir_dato_reentra_con_parametros_fusionados(emu):
    emu.detectar_texto("s1", "necesito cita para peritar el 1234 BCD en Madrid el lunes por la mañana")
    t = turno(emu.detectar_texto("s1", "mejor el jueves"))
    # corregir_dato -> followupEventInput SOLICITAR_CITA -> solicitar_cita con todo rellenado
    assert t.intent == "solicitar_cita"
    assert t.parametros["fecha"] == JUEVES
    assert t.parametros["matricula"] == "1234 BCD"
    assert t.parametros["ciudad"] == "Madrid"
    assert t.texto.startswith("¿Confirmo la cita para 1234 BCD el 2026-11-12")
    intents = [p["queryResult"]["intent"]["displayName"] for p in webhook_falso.peticiones[-2:]]
    assert intents == ["corregir_dato", "solicitar_cita"]
    # Y después se puede confirmar
    assert turno(emu.detectar_texto("s1", "vale")).intent == "solicitar_cita - si"


def test_corregir_matricula(emu):
    emu.detectar_texto("s1", "necesito cita para peritar el 1234 BCD en Madrid el lunes por la mañana")
    t = turno(emu.detectar_texto("s1", "no, la matrícula es 5678 FGH"))
    assert t.intent == "solicitar_cita"
    assert t.parametros["matricula"] == "5678 FGH"
    assert t.parametros["fecha"] == LUNES


def test_caducidad_de_contextos(emu):
    emu.detectar_texto("s1", "necesito cita para peritar el 1234 BCD en Madrid el lunes por la mañana")
    # Dos turnos ajenos agotan el follow-up (lifespan 2)
    emu.detectar_texto("s1", "¿cuál es la franquicia de mi póliza?")
    t = turno(emu.detectar_texto("s1", "¿tengo asistencia en carretera?"))
    ctx = contextos(t)
    assert "solicitar_cita-followup" not in ctx
    assert ctx["ctx_solicitud_en_curso"]["lifespan"] == 8
    assert turno(emu.detectar_texto("s1", "sí")).intent != "solicitar_cita - si"


def test_contexto_de_webhook_con_nombre_completo(emu):
    def webhook(peticion):
        return {"fulfillmentText": "ok", "outputContexts": [{
            "name": peticion["session"] + "/contexts/mi_contexto", "lifespanCount": 2,
            "parameters": {"x": 1}}]}
    emu.webhook = webhook
    t = turno(emu.detectar_texto("s1", "hola"))
    assert contextos(t)["mi_contexto"] == {"name": "mi_contexto", "lifespan": 2, "parameters": {"x": 1}}


def test_fallback(emu):
    t = turno(emu.detectar_texto("s1", "mi perro se llama Toby"))
    assert t.intent == "Default Fallback Intent"
    assert t.texto == "No consta en la documentación."


def test_consulta_documental_y_transferencia(emu):
    assert turno(emu.detectar_texto("s1", "¿me cubre la rotura de lunas?")).intent == "consulta_documental"
    assert turno(emu.detectar_texto("s1", "quiero hablar con una persona")).intent == "transferir_agente"


def test_peticion_webhook_formato_es_v2(emu):
    emu.detectar_texto("s1", "hola", payload={"canal": "voz"})
    p = webhook_falso.peticiones[-1]
    assert set(p) == {"responseId", "queryResult", "originalDetectIntentRequest", "session"}
    assert p["session"] == "projects/seguros-alonso-local/agent/sessions/s1"
    assert p["originalDetectIntentRequest"] == {"source": "local", "payload": {"canal": "voz"}}
    qr = p["queryResult"]
    for clave in ("queryText", "parameters", "allRequiredParamsPresent", "fulfillmentText",
                  "fulfillmentMessages", "outputContexts", "intent", "intentDetectionConfidence", "languageCode"):
        assert clave in qr
    assert qr["intent"]["name"].startswith("projects/seguros-alonso-local/agent/intents/")


def test_timeout_del_webhook_usa_respuesta_estatica():
    def lento(_):
        time.sleep(1)
        return {"fulfillmentText": "tarde"}
    emu = EmuladorDialogflow(webhook=lento, ahora=AHORA, timeout_webhook=0.2)
    r = emu.detectar_evento("s1", "WELCOME")
    assert r["webhookStatus"]["code"] == 4
    assert r["queryResult"]["fulfillmentText"].startswith("Hola, soy el asistente virtual")


def test_cliente_local_con_app_asgi_y_ssml():
    app = FastAPI()

    @app.post("/webhook")
    async def recibir(request: Request):
        await request.json()
        return {"fulfillmentText": "<speak>Hola <break time=\"300ms\"/> &amp; bienvenido</speak>",
                "fulfillmentMessages": [{"payload": {"ssml": "<speak>Hola</speak>"}}]}

    cliente = ClienteLocal(webhook=app, ahora=AHORA)
    t = cliente.evento("s1", "WELCOME", payload={"canal": "voz"})
    assert t.intent == "Default Welcome Intent"
    assert t.ssml.startswith("<speak>")
    assert t.texto == "Hola & bienvenido"
    assert set(t.latencias) == {"dialogflow", "webhook", "total"}
