"""Pruebas del webhook y de la lógica de negocio (sin red: el motor documental es falso)."""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from webhook import main, negocio
from webhook.rag import RespuestaDocumental

SESION = "projects/demo/agent/sessions/test-1"
HOY = date.today()


def proximo(dia_semana: int) -> date:
    """Próximo día de la semana (0=lunes) a partir de mañana."""
    d = HOY + timedelta(days=1)
    while d.weekday() != dia_semana:
        d += timedelta(days=1)
    return d


class MotorFalso:
    num_documentos = 5

    def responder(self, pregunta):
        if "franquicia" in pregunta:
            return RespuestaDocumental("La franquicia del todo riesgo es de 300 euros.", True,
                                       ["02_tabla_coberturas.png"], 0.1)
        return RespuestaDocumental("No dispongo de esa información en la documentación de "
                                   "Seguros Alonso. ¿Quieres que te pase con un agente?", False, [], 0.1)


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setitem(main.ESTADO, "motor", MotorFalso())
    return TestClient(main.app)  # sin "with": no ejecuta el arranque (no carga Chroma ni LLM)


def peticion(intent, params=None, texto="", completos=True, contextos=None, canal=None):
    return {
        "session": SESION,
        "queryResult": {
            "queryText": texto,
            "parameters": params or {},
            "allRequiredParamsPresent": completos,
            "intent": {"displayName": intent},
            "outputContexts": [
                {"name": f"{SESION}/contexts/{n}", "lifespanCount": v, "parameters": p}
                for n, v, p in (contextos or [])
            ],
            "languageCode": "es",
        },
        "originalDetectIntentRequest": {"source": "test", "payload": {"canal": canal} if canal else {}},
    }


def params_validos(**cambios):
    p = {"matricula": "1234 BCD", "fecha": f"{proximo(0).isoformat()}T12:00:00+01:00",
         "franja": "manana", "ciudad": "Madrid"}
    p.update(cambios)
    return p


# --- negocio ---------------------------------------------------------------

@pytest.mark.parametrize("valor,ok", [("1234 BCD", True), ("1234-bcd", True), ("1234 ABC", False),
                                      ("123 BCD", False), ("1234 BCQ", False), ("", False)])
def test_regex_matricula(valor, ok):
    assert negocio.validar_matricula(valor)[0] is ok


def test_reglas_fecha():
    assert negocio.evaluar_solicitud(params_validos()).valida
    sabado = proximo(5).isoformat()
    ev = negocio.evaluar_solicitud(params_validos(fecha=sabado))
    assert ev.problemas[0].campo == "fecha" and "fin de semana" in ev.problemas[0].mensaje
    ev = negocio.evaluar_solicitud(params_validos(fecha=HOY.isoformat()))
    assert ev.problemas[0].campo == "fecha"


def test_poliza_terceros_solo_lunas():
    ev = negocio.evaluar_solicitud(params_validos(matricula="5678FGH"))
    assert ev.problemas[0].campo == "tipo_dano"
    assert negocio.evaluar_solicitud(params_validos(matricula="5678FGH", tipo_dano="lunas")).valida


def test_ciudad_sin_taller():
    ev = negocio.evaluar_solicitud(params_validos(ciudad="Teruel"))
    assert ev.problemas[0].campo == "ciudad"


# --- webhook ---------------------------------------------------------------

def test_bienvenida_ssml_en_canal_voz(cliente):
    r = cliente.post("/webhook", json=peticion("Default Welcome Intent", canal="voz")).json()
    assert r["fulfillmentText"].startswith("<speak>")
    assert "Clara" in r["fulfillmentMessages"][0]["text"]["text"][0]


def test_slot_filling_personaliza(cliente):
    r = cliente.post("/webhook", json=peticion("solicitar_cita", {"matricula": "1234BCD"},
                                               "1234 BCD", completos=False)).json()
    assert "Ana" in r["fulfillmentText"] and "día" in r["fulfillmentText"]


def test_slot_filling_interrupcion_documental(cliente):
    r = cliente.post("/webhook", json=peticion("solicitar_cita", {"matricula": "1234BCD"},
                                               "¿cuál es la franquicia?", completos=False)).json()
    assert "300 euros" in r["fulfillmentText"] and "Seguimos con tu cita" in r["fulfillmentText"]


def test_solicitud_completa_pide_confirmacion(cliente):
    r = cliente.post("/webhook", json=peticion("solicitar_cita", params_validos())).json()
    assert "¿Confirmo la cita?" in r["fulfillmentText"]
    nombres = {c["name"].rsplit("/", 1)[-1]: c["lifespanCount"] for c in r["outputContexts"]}
    assert nombres == {"ctx_solicitud_en_curso": 10, "solicitar_cita-followup": 2}


def test_solicitud_invalida_cierra_followup(cliente):
    r = cliente.post("/webhook", json=peticion("solicitar_cita", params_validos(matricula="0000BBB"))).json()
    assert "No encuentro" in r["fulfillmentText"]
    vidas = {c["name"].rsplit("/", 1)[-1]: c["lifespanCount"] for c in r["outputContexts"]}
    assert vidas["solicitar_cita-followup"] == 0


def test_correccion_dispara_evento(cliente):
    ctx = [("ctx_solicitud_en_curso", 9, params_validos())]
    nuevo = f"{proximo(3).isoformat()}T12:00:00+01:00"
    r = cliente.post("/webhook", json=peticion("corregir_dato", {"fecha": nuevo}, "mejor el jueves",
                                               contextos=ctx)).json()
    assert r["followupEventInput"]["name"] == "SOLICITAR_CITA"
    assert r["followupEventInput"]["parameters"]["fecha"] == nuevo
    assert r["followupEventInput"]["parameters"]["matricula"] == "1234 BCD"


def test_confirmacion_crea_cita(cliente):
    ctx = [("ctx_solicitud_en_curso", 9, params_validos()), ("solicitar_cita-followup", 1, {})]
    r = cliente.post("/webhook", json=peticion("solicitar_cita - si", {}, "sí", contextos=ctx)).json()
    cita = r["fulfillmentMessages"][1]["payload"]["cita"]
    assert cita["codigo"].startswith("PER-") and cita["titular"] == "Ana Ruiz"
    assert cita["codigo"] in negocio.CITAS


def test_consulta_con_flujo_en_curso_retoma(cliente):
    ctx = [("ctx_solicitud_en_curso", 8, params_validos()), ("solicitar_cita-followup", 1, {})]
    r = cliente.post("/webhook", json=peticion("consulta_documental", {}, "¿y la franquicia?",
                                               contextos=ctx)).json()
    assert "300 euros" in r["fulfillmentText"] and "¿Confirmo la cita?" in r["fulfillmentText"]


def test_fallback_no_inventa(cliente):
    r = cliente.post("/webhook", json=peticion("Default Fallback Intent", {}, "¿cubre el seguro de vida?")).json()
    assert "No dispongo" in r["fulfillmentText"]


def test_transferencia_agente(cliente):
    r = cliente.post("/webhook", json=peticion("transferir_agente", {}, "quiero hablar con una persona")).json()
    payload = r["fulfillmentMessages"][1]["payload"]
    assert payload["liveAgentHandoff"] is True and payload["ticket"].startswith("TCK-")


def test_correccion_recupera_copia_guardada(cliente):
    """Dialogflow real deja matricula="" en el contexto al activar corregir_dato: se usa la copia."""
    nuevo = f"{proximo(3).isoformat()}T12:00:00+01:00"
    ctx = [("ctx_solicitud_en_curso", 9, {"matricula": "", "fecha": nuevo, "franja": "",
                                          "guardado_matricula": "1234 BCD", "guardado_franja": "tarde"})]
    r = cliente.post("/webhook", json=peticion("corregir_dato", {"fecha": nuevo}, "mejor el jueves",
                                               contextos=ctx)).json()
    params = r["followupEventInput"]["parameters"]
    assert params["matricula"] == "1234 BCD" and params["franja"] == "tarde" and params["fecha"] == nuevo


def test_reentrada_tras_interrupcion_recupera_datos(cliente):
    """Si Dialogflow vuelve a entrar en solicitar_cita de cero, el webhook relanza el evento con lo guardado."""
    ctx = [("ctx_solicitud_en_curso", 9, {"guardado_matricula": "1234 BCD"})]
    r = cliente.post("/webhook", json=peticion("solicitar_cita", {"franja": "manana"}, "por la mañana",
                                               completos=False, contextos=ctx)).json()
    assert r["followupEventInput"]["parameters"] == {"matricula": "1234 BCD", "franja": "manana"}


@pytest.mark.parametrize("texto", ["la matrícula es 1234 be ce de", "la matrícula es 1.234 BCD",
                                   "es la uno dos tres cuatro be ce de"])
def test_matricula_dictada_en_slot_filling(cliente, texto):
    """El STT de Google transcribe la matrícula de formas que la entidad regex no reconoce."""
    r = cliente.post("/webhook", json=peticion("solicitar_cita", {}, texto, completos=False)).json()
    assert r["followupEventInput"]["parameters"]["matricula"] == "1234 BCD"


def test_si_sin_solicitud_abierta(cliente):
    r = cliente.post("/webhook", json=peticion("solicitar_cita - si", {}, "sí")).json()
    assert "no tengo ninguna cita pendiente" in r["fulfillmentText"]


def test_fin_de_semana_una_sola_pregunta(cliente):
    r = cliente.post("/webhook", json=peticion("solicitar_cita", params_validos(fecha=proximo(5).isoformat()))).json()
    assert r["fulfillmentText"].count("?") == 1, r["fulfillmentText"]


def test_fallback_con_flujo_rescata_respuesta_suelta(cliente):
    """Dialogflow real a veces manda "el lunes por la mañana" al fallback: el flujo sigue."""
    ctx = [("ctx_solicitud_en_curso", 9, {"guardado_matricula": "1234 BCD"})]
    r = cliente.post("/webhook", json=peticion("Default Fallback Intent", {}, "el lunes por la mañana",
                                               contextos=ctx)).json()
    params = r["followupEventInput"]["parameters"]
    assert params["matricula"] == "1234 BCD" and params["franja"] == "manana" and params["fecha"]
