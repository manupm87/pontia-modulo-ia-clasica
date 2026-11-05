"""Pruebas del puente de Telegram sin Telegram: cliente de Dialogflow falso."""

from dialogflow_local.cliente import Turno
from telegram_bot import Asistente


class ClienteFalso:
    def __init__(self):
        self.llamadas = []

    def _turno(self, texto):
        return Turno(query_text=texto, intent="solicitar_cita", confianza=0.9, parametros={},
                     contextos=[], texto="¿Confirmo la cita?", ssml="<speak>¿Confirmo la cita?</speak>",
                     mensajes=[{"text": {"text": ["¿Confirmo la cita?"]}},
                               {"payload": {"botones": ["Sí", "No"]}}],
                     audio=None, latencias={"total": 0.1})

    def texto(self, sesion, texto, payload=None):
        self.llamadas.append((sesion, texto, payload))
        return self._turno(texto)

    def evento(self, sesion, nombre, parametros=None, payload=None):
        self.llamadas.append((sesion, nombre, payload))
        return self._turno("")


def test_texto_con_botones_y_sesion_por_chat():
    cliente = ClienteFalso()
    salida = Asistente(cliente).texto(42, "sí")
    assert salida.texto == "¿Confirmo la cita?"
    assert salida.botones == ["Sí", "No"]
    assert salida.voz_ogg is None
    assert cliente.llamadas[0][0] == "telegram-42"
    assert cliente.llamadas[0][2] == {"canal": "telegram"}


def test_start_dispara_bienvenida():
    cliente = ClienteFalso()
    Asistente(cliente).inicio(7)
    assert cliente.llamadas[0][1] == "WELCOME"
