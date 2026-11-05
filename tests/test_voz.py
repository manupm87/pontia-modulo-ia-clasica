"""Pruebas del canal de voz: SSML, normalización de matrículas y viaje de ida y vuelta TTS -> STT."""

from __future__ import annotations

import socket
from datetime import date

import pytest

from voz.ssml import (a_ssml, deletrear_matricula, enfasis, fecha_hablada, hora_hablada, pausa,
                      prosodia, segmentar, speak, ssml_a_texto, ssml_para_elevenlabs, texto)
from voz.stt import normalizar_matricula_hablada


def _textos(segmentos):
    return [s["texto"] for s in segmentos if s["tipo"] == "texto"]


# ---------------------------------------------------------------------------
# Construcción de SSML
# ---------------------------------------------------------------------------

def test_a_ssml_escapa_y_envuelve():
    assert a_ssml("Coches & motos <3") == "<speak>Coches &amp; motos &lt;3</speak>"
    ya = "<speak>Hola</speak>"
    assert a_ssml(ya) == ya


def test_deletrear_matricula_separa_cifras_y_letras():
    s = deletrear_matricula("1234bcd")
    assert s.count('interpret-as="characters"') == 2
    assert ">1234<" in s and ">BCD<" in s and "<break" in s
    assert ssml_a_texto(speak("Matrícula", s)) == "Matrícula 1234 BCD"


def test_fecha_hablada():
    hoy = date(2026, 11, 5)
    assert fecha_hablada(date(2026, 11, 9), hoy=hoy) == "el lunes 9 de noviembre"
    assert fecha_hablada("2026-11-06T12:00:00+01:00", hoy=hoy, relativa=True) == \
        "mañana, viernes 6 de noviembre"
    assert fecha_hablada("2027-01-04", hoy=hoy) == "el lunes 4 de enero de 2027"


def test_hora_hablada():
    assert hora_hablada("17:00") == "las cinco de la tarde"
    assert hora_hablada("9:30") == "las nueve y media de la mañana"
    assert hora_hablada("13:00") == "la una del mediodía"
    assert hora_hablada("13:45", articulo=False) == "dos menos cuarto del mediodía"


def test_segmentar_lee_las_horas_en_voz():
    segs = segmentar("<speak>Te espera a las 17:00, de 9:00 a 14:00 h.</speak>")
    assert _textos(segs) == ["Te espera a las cinco de la tarde, de nueve de la mañana a dos del mediodía."]
    # para mostrar se conserva el formato escrito
    assert ssml_a_texto("<speak>a las 17:00</speak>") == "a las 17:00"


# ---------------------------------------------------------------------------
# SSML -> texto y segmentos
# ---------------------------------------------------------------------------

def test_ssml_a_texto_quita_etiquetas():
    ssml = speak(texto("Hola."), pausa(300), enfasis("Atención"), prosodia("despacio", rate="80%"))
    assert ssml_a_texto(ssml) == "Hola. Atención despacio"
    # SSML roto: se quitan las etiquetas igualmente
    assert ssml_a_texto("<speak>Hola <break time='1s'> mundo") == "Hola mundo"


def test_segmentar_pausas_y_prosodia():
    ssml = ('<speak>Hola. <break time="300ms"/> <prosody rate="slow" pitch="+2st">'
            'despacio <emphasis level="strong">muy</emphasis></prosody>'
            '<break strength="strong"/>fin</speak>')
    segs = segmentar(ssml)
    assert segs[0] == {"tipo": "texto", "texto": "Hola.", "rate": "+0%", "pitch": "+0Hz",
                       "volume": "+0%"}
    assert segs[1] == {"tipo": "pausa", "ms": 300}
    assert segs[2]["texto"] == "despacio" and segs[2]["rate"] == "-20%" and segs[2]["pitch"] == "+24Hz"
    # el énfasis se acumula sobre la prosodia del padre
    assert segs[3]["texto"] == "muy" and segs[3]["rate"] == "-30%" and segs[3]["volume"] == "+20%"
    assert segs[4] == {"tipo": "pausa", "ms": 700}
    assert segs[5]["texto"] == "fin"


def test_segmentar_say_as_characters_deletrea():
    segs = segmentar(speak("Cita para el", deletrear_matricula("1234BCD") + "."))
    assert _textos(segs) == ["Cita para el uno dos tres cuatro", "be ce de."]
    assert {"tipo": "pausa", "ms": 200} in segs


def test_segmentar_say_as_codigo_cita():
    segs = segmentar('<speak>Tu código es <say-as interpret-as="characters">PER-12345</say-as></speak>')
    assert _textos(segs) == ["Tu código es pe e erre, uno dos tres cuatro cinco"]


def test_segmentar_s_p_sub_y_rate_absoluto():
    segs = segmentar('<speak><p><s>Uno.</s><s>Dos.</s></p><sub alias="Seguros Alonso">SA</sub>'
                     '<prosody rate="90%">lento</prosody></speak>')
    assert segs[0]["texto"] == "Uno." and segs[1] == {"tipo": "pausa", "ms": 150}
    assert segs[2]["texto"] == "Dos." and segs[3] == {"tipo": "pausa", "ms": 400}
    assert segs[4]["texto"] == "Seguros Alonso"
    assert segs[5]["rate"] == "-10%"


def test_segmentar_texto_plano_e_invalido():
    assert _textos(segmentar("Texto & plano")) == ["Texto & plano"]
    assert _textos(segmentar("<speak>Sin <cerrar")) == ["Sin"]
    assert segmentar("") == []


def test_ssml_para_elevenlabs_conserva_breaks():
    salida = ssml_para_elevenlabs(speak("Hola", pausa(500), enfasis("adiós")))
    assert salida == 'Hola <break time="0.50s" /> adiós'


# ---------------------------------------------------------------------------
# Matrículas dictadas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dicho, esperado", [
    ("Quiero cita para el 1234 B C D.", "Quiero cita para el 1234 BCD."),
    ("uno dos tres cuatro be ce de", "1234 BCD"),
    ("la matrícula es 5678-fgh", "la matrícula es 5678 FGH"),
    ("12 34, b, c, d por favor", "1234 BCD por favor"),
    ("1234BCD", "1234 BCD"),
    ("1.234 BCD", "1234 BCD"),
    ("nueve cero uno dos jota ka ele", "9012 JKL"),
    ("cuatro cinco seis siete uve doble equis i griega", "4567 WXY"),
])
def test_normalizar_matricula_hablada(dicho, esperado):
    assert normalizar_matricula_hablada(dicho) == esperado


@pytest.mark.parametrize("texto_sin_matricula", [
    "el 1234 de mayo",                 # "de" seguido de algo que no es letra
    "tengo 3 coches de Madrid",
    "1234 BCE",                        # vocal: no es una matrícula válida
    "",
])
def test_normalizar_no_toca_otros_textos(texto_sin_matricula):
    assert normalizar_matricula_hablada(texto_sin_matricula) == texto_sin_matricula


# ---------------------------------------------------------------------------
# Ida y vuelta TTS (edge, necesita red) -> STT (whisper local)
# ---------------------------------------------------------------------------

def _hay_red() -> bool:
    try:
        socket.create_connection(("speech.platform.bing.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _hay_red(), reason="sin red: edge-tts necesita el servicio de Microsoft")
def test_ida_y_vuelta_tts_stt():
    pytest.importorskip("faster_whisper")
    from voz.stt import transcribir
    from voz.tts import a_ogg_opus, duracion, sintetizar

    wav = sintetizar("Quiero pedir cita para el 1234 BCD", backend="edge")
    assert wav[:4] == b"RIFF" and 1.5 < duracion(wav) < 8
    assert "1234 BCD" in transcribir(wav)
    # también desde una nota de voz de Telegram (OGG/Opus)
    assert "1234 BCD" in transcribir(a_ogg_opus(wav))
