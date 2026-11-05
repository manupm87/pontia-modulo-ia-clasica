"""
Puente Telegram <-> Dialogflow para el asistente de Seguros Alonso (mejora opcional).

Qué hace:
- Mensajes de texto -> Dialogflow -> respuesta con botones (inline keyboard).
- Notas de voz -> Whisper (STT) -> Dialogflow -> respuesta en texto + NOTA DE VOZ (TTS con SSML,
  voz clonada si TTS_BACKEND=xtts/elevenlabs).
- Botones: al pulsarlos se envía su texto como si el usuario lo hubiera escrito.
- /start dispara el evento WELCOME.

Alternativa sin código: la integración nativa de Telegram de Dialogflow ES. El webhook ya
devuelve un payload de plataforma TELEGRAM con los botones, así que también funciona ahí,
pero la integración nativa no acepta notas de voz; por eso existe este puente.

Uso:
    set TELEGRAM_BOT_TOKEN=...            (token de @BotFather)
    set DIALOGFLOW_BACKEND=local|google
    python telegram_bot.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from dialogflow_local.cliente import Turno, crear_cliente

load_dotenv()
log = logging.getLogger("telegram_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# "voz": responder con nota de voz solo si el usuario habló; "siempre"; "nunca"
MODO_VOZ = os.environ.get("TELEGRAM_RESPUESTA_VOZ", "voz")


@dataclass
class Salida:
    """Lo que hay que mandar a Telegram, independiente de la librería (se prueba con pytest)."""
    texto: str
    botones: list[str] = field(default_factory=list)
    voz_ogg: bytes | None = None
    transcripcion: str | None = None
    turno: Turno | None = None


def botones_de(turno: Turno) -> list[str]:
    for m in turno.mensajes or []:
        botones = (m.get("payload") or {}).get("botones")
        if botones:
            return list(botones)
    return []


class Asistente:
    """Lógica del puente: una sesión de Dialogflow por chat de Telegram."""

    def __init__(self, cliente=None):
        self.cliente = cliente or crear_cliente()

    @staticmethod
    def sesion(chat_id: int | str) -> str:
        return f"telegram-{chat_id}"

    def _salida(self, turno: Turno, con_voz: bool, transcripcion: str | None = None) -> Salida:
        salida = Salida(turno.texto, botones_de(turno), transcripcion=transcripcion, turno=turno)
        if con_voz and turno.texto:
            from voz.tts import a_ogg_opus, sintetizar
            wav = turno.audio or sintetizar(turno.ssml or turno.texto)
            salida.voz_ogg = a_ogg_opus(wav)
        return salida

    def inicio(self, chat_id) -> Salida:
        turno = self.cliente.evento(self.sesion(chat_id), "WELCOME", payload={"canal": "telegram"})
        return self._salida(turno, MODO_VOZ == "siempre")

    def texto(self, chat_id, texto: str) -> Salida:
        turno = self.cliente.texto(self.sesion(chat_id), texto, payload={"canal": "telegram"})
        return self._salida(turno, MODO_VOZ == "siempre")

    def voz(self, chat_id, audio_ogg: bytes) -> Salida:
        from voz.stt import transcribir
        transcripcion = transcribir(audio_ogg)
        turno = self.cliente.texto(self.sesion(chat_id), transcripcion, payload={"canal": "voz"})
        return self._salida(turno, MODO_VOZ != "nunca", transcripcion)


# ---------------------------------------------------------------------------
# Adaptador python-telegram-bot
# ---------------------------------------------------------------------------

def main() -> None:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes,
                              MessageHandler, filters)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Falta TELEGRAM_BOT_TOKEN (créalo con @BotFather y ponlo en .env)")

    asistente = Asistente()

    async def enviar(update: Update, salida: Salida) -> None:
        mensaje = update.effective_message
        teclado = None
        if salida.botones:
            teclado = InlineKeyboardMarkup(
                [[InlineKeyboardButton(b, callback_data=b[:64]) for b in salida.botones]])
        if salida.transcripcion:
            await mensaje.reply_text(f"🗣️ «{salida.transcripcion}»")
        if salida.voz_ogg:
            await mensaje.reply_voice(salida.voz_ogg, caption=salida.texto[:1000], reply_markup=teclado)
        else:
            await mensaje.reply_text(salida.texto, reply_markup=teclado)

    async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await enviar(update, await asyncio.to_thread(asistente.inicio, update.effective_chat.id))

    async def on_texto(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        texto = update.effective_message.text
        await enviar(update, await asyncio.to_thread(asistente.texto, update.effective_chat.id, texto))

    async def on_voz(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_chat.send_action("record_voice")
        fichero = await update.effective_message.voice.get_file()
        audio = bytes(await fichero.download_as_bytearray())
        await enviar(update, await asyncio.to_thread(asistente.voz, update.effective_chat.id, audio))

    async def on_boton(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        consulta = update.callback_query
        await consulta.answer()
        await consulta.edit_message_reply_markup(None)  # quitamos los botones ya usados
        await enviar(update, await asyncio.to_thread(asistente.texto, update.effective_chat.id, consulta.data))

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.VOICE, on_voz))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_texto))
    app.add_handler(CallbackQueryHandler(on_boton))
    log.info("Bot de Telegram escuchando (backend Dialogflow: %s)",
             os.environ.get("DIALOGFLOW_BACKEND", "local"))
    app.run_polling()


if __name__ == "__main__":
    main()
