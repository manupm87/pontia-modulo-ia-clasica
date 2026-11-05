"""
Construcción de respuestas de webhook en formato Dialogflow ES.

Una misma respuesta se genera a la vez en tres "sabores":

- texto plano  -> chat, Telegram, consola (fulfillmentText y mensaje de texto)
- SSML         -> voz: pausas, énfasis y lectura de matrículas/códigos letra a letra
- Telegram     -> payload con botones (inline keyboard) para la integración de Telegram

Para no escribir cada frase dos veces, la clase `Frase` va acumulando trozos con su versión
en texto y su versión en SSML.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape

CTX_SOLICITUD = "ctx_solicitud_en_curso"
CTX_FOLLOWUP = "solicitar_cita-followup"
EVENTO_SOLICITAR_CITA = "SOLICITAR_CITA"


# ---------------------------------------------------------------------------
# Fragmentos SSML
# ---------------------------------------------------------------------------

def ssml_deletreo(codigo: str) -> str:
    """
    Lee un código carácter a carácter con una pausa corta entre grupos.
    '1234BCD' -> '1 2 3 4, B C D'. Mucho más claro al teléfono que "mil doscientos...".
    """
    grupos = re.findall(r"\d+|[A-Za-z]+", codigo)
    partes = [f'<say-as interpret-as="characters">{escape(g)}</say-as>' for g in grupos]
    return '<break time="150ms"/>'.join(partes)


class Frase:
    """Acumula una respuesta en paralelo como texto plano y como SSML."""

    def __init__(self, texto: str = ""):
        self._texto: list[str] = []
        self._ssml: list[str] = []
        if texto:
            self.decir(texto)

    def decir(self, texto: str, ssml: str | None = None) -> "Frase":
        """Añade una frase. Si no se da SSML, se usa el texto escapado."""
        self._texto.append(texto)
        self._ssml.append(ssml if ssml is not None else escape(texto))
        return self

    def pausa(self, ms: int = 300) -> "Frase":
        self._ssml.append(f'<break time="{ms}ms"/>')
        return self

    def codigo(self, prefijo: str, codigo: str, sufijo: str = "") -> "Frase":
        """Frase con un código que se deletrea en voz: 'Tu cita es la PER-12345.'"""
        return self.decir(
            f"{prefijo}{codigo}{sufijo}",
            f"{escape(prefijo)}{ssml_deletreo(codigo)}{escape(sufijo)}",
        )

    def enfasis(self, texto: str) -> "Frase":
        return self.decir(texto, f'<emphasis level="moderate">{escape(texto)}</emphasis>')

    def lento(self, texto: str) -> "Frase":
        return self.decir(texto, f'<prosody rate="90%">{escape(texto)}</prosody>')

    def extender(self, otra: "Frase") -> "Frase":
        self._texto.extend(otra._texto)
        self._ssml.extend(otra._ssml)
        return self

    @property
    def texto(self) -> str:
        return " ".join(t.strip() for t in self._texto if t.strip())

    @property
    def ssml(self) -> str:
        cuerpo = " ".join(s for s in self._ssml if s)
        return f"<speak>{cuerpo}</speak>"


# ---------------------------------------------------------------------------
# Respuesta completa de webhook
# ---------------------------------------------------------------------------

def nombre_contexto(sesion: str, nombre: str) -> str:
    return f"{sesion}/contexts/{nombre}"


def construir(
    frase: Frase,
    sesion: str,
    canal: str = "texto",
    contextos: list[tuple[str, int, dict]] | None = None,
    botones: list[str] | None = None,
    payload_extra: dict | None = None,
) -> dict:
    """
    Arma el JSON de respuesta del webhook.

    - canal "voz": fulfillmentText lleva SSML (el TTS nativo de Dialogflow lo interpreta).
    - resto de canales: fulfillmentText en texto plano.
    - contextos: [(nombre, lifespan, parametros)] -> outputContexts con el nombre completo.
    """
    texto = frase.texto
    ssml = frase.ssml
    payload = {"texto": texto, "ssml": ssml, "botones": botones or []}
    if payload_extra:
        payload.update(payload_extra)

    telegram: dict = {"text": texto}
    if botones:
        telegram["reply_markup"] = {
            "inline_keyboard": [[{"text": b, "callback_data": b} for b in botones]]
        }

    respuesta: dict = {
        "fulfillmentText": ssml if canal == "voz" else texto,
        "fulfillmentMessages": [
            {"text": {"text": [texto]}},
            {"payload": payload},
            {"platform": "TELEGRAM", "payload": {"telegram": telegram}},
        ],
        "source": "webhook-seguros-alonso",
    }
    if botones:
        # Respuestas rápidas también en el formato estándar (las pinta la integración nativa)
        respuesta["fulfillmentMessages"].append(
            {"platform": "TELEGRAM", "quickReplies": {"title": texto[:60], "quickReplies": botones}}
        )
    if contextos:
        respuesta["outputContexts"] = [
            {"name": nombre_contexto(sesion, nombre), "lifespanCount": vida, "parameters": params or {}}
            for nombre, vida, params in contextos
        ]
    return respuesta


def evento(nombre: str, parametros: dict, idioma: str = "es") -> dict:
    """Respuesta que dispara un evento (Dialogflow ejecuta la intención asociada)."""
    return {
        "followupEventInput": {
            "name": nombre,
            "parameters": parametros,
            "languageCode": idioma,
        }
    }
