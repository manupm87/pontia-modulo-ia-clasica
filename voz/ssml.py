"""
Utilidades SSML para que el asistente suene natural.

Tres usos:

1. **Construir** SSML desde el webhook con pequeñas funciones que devuelven fragmentos ya
   escapados: `pausa(300)`, `enfasis("importante")`, `deletrear_matricula("1234BCD")`,
   `fecha_hablada(date)`... y `speak(...)` / `a_ssml(...)` para envolver en `<speak>`.
2. **Mostrar**: `ssml_a_texto(ssml)` quita las etiquetas (para consola, Telegram, logs).
3. **Sintetizar con motores que no entienden SSML** (edge-tts, XTTS): `segmentar(ssml)` lo
   convierte en una lista de trozos de texto con su prosodia y pausas:

       [{"tipo": "texto", "texto": "Hola", "rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
        {"tipo": "pausa", "ms": 300}, ...]

   El TTS sintetiza cada trozo con sus parámetros y mete silencios en las pausas.

Todo es tolerante a SSML mal formado: si no se puede parsear, se trata como texto plano.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import date, datetime
from xml.sax.saxutils import escape, quoteattr

# ---------------------------------------------------------------------------
# Lectura en voz de letras y cifras (para <say-as interpret-as="characters">)
# ---------------------------------------------------------------------------

CIFRAS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]

# Nombre de cada letra en español, tal como se dice al deletrear por teléfono.
LETRAS = {
    "A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "efe", "G": "ge",
    "H": "hache", "I": "i", "J": "jota", "K": "ka", "L": "ele", "M": "eme", "N": "ene",
    "Ñ": "eñe", "O": "o", "P": "pe", "Q": "cu", "R": "erre", "S": "ese", "T": "te",
    "U": "u", "V": "uve", "W": "uve doble", "X": "equis", "Y": "i griega", "Z": "zeta",
}

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def deletrear_texto(codigo: str) -> str:
    """
    Convierte un código en palabras que se leen carácter a carácter.

    Los grupos de cifras y de letras se separan con una coma (pausa corta natural):
    "1234BCD" -> "uno dos tres cuatro, be ce de";  "PER-12345" -> "pe e erre, uno dos tres cuatro cinco".
    """
    grupos = re.findall(r"\d+|[A-Za-zÑñ]+", codigo)
    hablados = []
    for grupo in grupos:
        if grupo.isdigit():
            hablados.append(" ".join(CIFRAS[int(c)] for c in grupo))
        else:
            hablados.append(" ".join(LETRAS.get(c.upper(), c) for c in grupo))
    return ", ".join(hablados)


# ---------------------------------------------------------------------------
# 1. Construcción de SSML (fragmentos ya escapados)
# ---------------------------------------------------------------------------

def a_ssml(texto: str) -> str:
    """Envuelve texto plano en `<speak>` escapando XML. Si ya es SSML, lo devuelve tal cual."""
    if es_ssml(texto):
        return texto.strip()
    return f"<speak>{escape(texto.strip())}</speak>"


def es_ssml(texto: str) -> bool:
    return texto.lstrip().startswith("<speak")


def speak(*fragmentos: str) -> str:
    """Une fragmentos SSML (ya escapados, p. ej. salidas de `pausa`, `enfasis`...) en un `<speak>`."""
    return "<speak>" + " ".join(f.strip() for f in fragmentos if f and f.strip()) + "</speak>"


def texto(t: str) -> str:
    """Texto literal escapado para meterlo dentro de un SSML."""
    return escape(t)


def pausa(ms: int | None = 300, fuerza: str | None = None) -> str:
    """`<break>`: por tiempo (`pausa(400)`) o por fuerza (`pausa(fuerza="strong")`)."""
    if fuerza:
        return f"<break strength={quoteattr(fuerza)}/>"
    return f'<break time="{int(ms or 0)}ms"/>'


def enfasis(t: str, nivel: str = "moderate") -> str:
    """Pronuncia `t` con énfasis (strong | moderate | reduced)."""
    return f"<emphasis level={quoteattr(nivel)}>{escape(t)}</emphasis>"


def prosodia(t: str, rate: str | None = None, pitch: str | None = None,
             volume: str | None = None) -> str:
    """`<prosody>`: velocidad ("90%", "slow", "+10%"), tono ("+2st", "low") y volumen ("loud")."""
    attrs = "".join(
        f" {nombre}={quoteattr(valor)}"
        for nombre, valor in (("rate", rate), ("pitch", pitch), ("volume", volume)) if valor
    )
    return f"<prosody{attrs}>{escape(t)}</prosody>" if attrs else escape(t)


def frase(t: str) -> str:
    """Una oración (`<s>`): el motor hace una pausa breve al final."""
    return f"<s>{escape(t)}</s>"


def deletrear(codigo: str) -> str:
    """Lee un código carácter a carácter: `<say-as interpret-as="characters">`."""
    return f'<say-as interpret-as="characters">{escape(codigo)}</say-as>'


def deletrear_matricula(matricula: str) -> str:
    """
    Matrícula española leída como la dice una persona: "1 2 3 4, B C D".

    Cifras y letras van en dos `<say-as>` separados por una pausa corta, para que el oyente
    distinga los dos bloques. Sirve también para otros códigos ("PER-12345").
    """
    grupos = re.findall(r"\d+|[A-Za-zÑñ]+", matricula.upper())
    if not grupos:
        return escape(matricula)
    return pausa(200).join(deletrear(g) for g in grupos)


def fecha_hablada(fecha: date | datetime | str, hoy: date | None = None,
                  relativa: bool = False) -> str:
    """
    Fecha como se dice en voz: "el lunes 9 de noviembre".

    - Acepta `date`, `datetime` o cadena ISO (el formato de `@sys.date`: "2026-11-09T12:00:00+01:00").
    - Añade el año solo si no es el actual.
    - Con `relativa=True` dice "mañana, lunes 9 de noviembre" / "hoy, ..." cuando corresponde.
    """
    if isinstance(fecha, str):
        fecha = date.fromisoformat(fecha.strip()[:10])
    elif isinstance(fecha, datetime):
        fecha = fecha.date()
    hoy = hoy or date.today()

    dicho = f"{DIAS_SEMANA[fecha.weekday()]} {fecha.day} de {MESES[fecha.month - 1]}"
    if fecha.year != hoy.year:
        dicho += f" de {fecha.year}"

    if relativa:
        dias = (fecha - hoy).days
        if dias == 0:
            return f"hoy, {dicho}"
        if dias == 1:
            return f"mañana, {dicho}"
        if dias == 2:
            return f"pasado mañana, {dicho}"
    return f"el {dicho}"


HORAS = ["doce", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
         "diez", "once"]


def hora_hablada(hora: str, articulo: bool = True) -> str:
    """
    Hora como se dice en voz: "17:00" -> "las cinco de la tarde", "9:30" -> "las nueve y media
    de la mañana", "13:45" -> "las dos menos cuarto del mediodía". Los motores TTS leen "17:00"
    de forma poco natural (o directamente mal), por eso `segmentar` lo aplica automáticamente.
    """
    h, m = (int(x) for x in hora.split(":")[:2])
    if m > 30 and m % 15 == 0:          # "menos cuarto" se refiere a la hora siguiente
        h, minutos = (h + 1) % 24, " menos cuarto"
    else:
        minutos = {0: "", 15: " y cuarto", 30: " y media"}.get(m, f" y {m}")
    if 6 <= h <= 11:
        periodo = " de la mañana"
    elif 12 <= h <= 14:
        periodo = " del mediodía"
    elif 15 <= h <= 20:
        periodo = " de la tarde"
    elif 1 <= h <= 5:
        periodo = " de la madrugada"
    else:
        periodo = " de la noche"
    nombre = HORAS[h % 12]
    art = ("la " if nombre == "una" else "las ") if articulo else ""
    return f"{art}{nombre}{minutos}{periodo}"


_HORA = re.compile(r"\b(?:(las?)\s+)?([01]?\d|2[0-3]):([0-5]\d)(?:\s*h\b)?", re.IGNORECASE)


def _horas_habladas(t: str) -> str:
    """'a las 17:00' -> 'a las cinco de la tarde' (respeta el artículo que ya hubiera)."""
    def cambiar(m: re.Match) -> str:
        return hora_hablada(f"{m.group(2)}:{m.group(3)}", articulo=bool(m.group(1)))
    return _HORA.sub(cambiar, t)


# ---------------------------------------------------------------------------
# Parseo tolerante
# ---------------------------------------------------------------------------

def _parsear(ssml: str) -> ET.Element | None:
    """Parsea SSML (con o sin `<speak>`). Devuelve None si no es XML válido."""
    contenido = re.sub(r"^\s*<\?xml[^>]*\?>", "", ssml or "").strip()
    if not es_ssml(contenido):
        contenido = f"<speak>{contenido}</speak>"
    try:
        return ET.fromstring(contenido)
    except ET.ParseError:
        return None


def _nombre(elem: ET.Element) -> str:
    """Nombre de etiqueta sin espacio de nombres ('{http://www.w3.org/...}speak' -> 'speak')."""
    return elem.tag.rsplit("}", 1)[-1].lower() if isinstance(elem.tag, str) else ""


def _quitar_etiquetas(ssml: str) -> str:
    """Último recurso para SSML roto: elimina cualquier cosa entre < > y desescapa."""
    sin = re.sub(r"<break[^>]*>", " ", ssml or "")
    sin = re.sub(r"<[^>]*(>|$)", " ", sin)  # también etiquetas sin cerrar al final
    return _limpiar_espacios(html.unescape(sin))


def _limpiar_espacios(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s+([,.;:!?…)\]])", r"\1", t)


# ---------------------------------------------------------------------------
# 2. SSML -> texto para mostrar
# ---------------------------------------------------------------------------

def ssml_a_texto(ssml: str) -> str:
    """
    Quita las etiquetas para mostrar la respuesta. Las matrículas deletreadas se muestran
    como se escriben ("1234 BCD"), no como se pronuncian.
    """
    if not ssml:
        return ""
    raiz = _parsear(ssml)
    if raiz is None:
        return _quitar_etiquetas(ssml)

    trozos: list[str] = []

    def recorrer(elem: ET.Element) -> None:
        nombre = _nombre(elem)
        if nombre in ("break", "mark"):
            trozos.append(" ")
        elif nombre == "desc":
            pass  # descripción de un <audio>: no es parte de lo que se dice
        else:
            trozos.append(elem.text or "")
            for hijo in elem:
                recorrer(hijo)
                trozos.append(hijo.tail or "")
            if nombre in ("s", "p", "sentence", "paragraph"):
                trozos.append(" ")

    recorrer(raiz)
    return _limpiar_espacios("".join(trozos))


# ---------------------------------------------------------------------------
# 3. SSML -> segmentos para motores sin soporte SSML
# ---------------------------------------------------------------------------

# Duraciones de <break strength="..."> (ms), aproximando lo que hace Google TTS.
PAUSA_POR_FUERZA = {"none": 0, "x-weak": 100, "weak": 200, "medium": 400,
                    "strong": 700, "x-strong": 1000}
PAUSA_ORACION_MS = 150   # al cerrar <s>
PAUSA_PARRAFO_MS = 400   # al cerrar <p>
PAUSA_MAXIMA_MS = 5000

# Palabras clave de <prosody>, traducidas a variaciones relativas.
RATE_CLAVE = {"x-slow": -40, "slow": -20, "medium": 0, "default": 0, "fast": 20, "x-fast": 40}
PITCH_CLAVE_HZ = {"x-low": -20, "low": -10, "medium": 0, "default": 0, "high": 10, "x-high": 20}
VOLUME_CLAVE = {"silent": -100, "x-soft": -50, "soft": -25, "medium": 0, "default": 0,
                "loud": 25, "x-loud": 50}
HZ_POR_SEMITONO = 12     # ~6 % de una voz de 200 Hz
HZ_POR_PORCENTAJE = 2    # 1 % de 200 Hz

# Cómo se traduce <emphasis level="..."> (velocidad %, volumen %, tono Hz).
ENFASIS = {"strong": (-10, 20, 6), "moderate": (-5, 10, 3), "none": (0, 0, 0),
           "reduced": (5, -15, -3)}


@dataclass(frozen=True)
class _Estilo:
    """Prosodia acumulada en la rama actual del árbol SSML (valores relativos)."""
    rate: int = 0      # %
    pitch: int = 0     # Hz
    volume: int = 0    # %

    def como_dict(self) -> dict:
        rate = max(-50, min(100, self.rate))
        pitch = max(-50, min(50, self.pitch))
        volume = max(-100, min(100, self.volume))
        return {"rate": f"{rate:+d}%", "pitch": f"{pitch:+d}Hz", "volume": f"{volume:+d}%"}


def _numero(valor: str) -> float | None:
    try:
        return float(re.sub(r"[^0-9.+-]", "", valor))
    except ValueError:
        return None


def _rate(valor: str) -> int:
    v = valor.strip().lower()
    if v in RATE_CLAVE:
        return RATE_CLAVE[v]
    n = _numero(v)
    if n is None:
        return 0
    if v.endswith("%"):
        # "+10%" es relativo; "90%" (sin signo) es absoluto respecto a la velocidad normal.
        return int(n) if v[0] in "+-" else int(n - 100)
    return int((n - 1) * 100)  # multiplicador: "1.2" -> +20 %


def _pitch(valor: str) -> int:
    v = valor.strip().lower()
    if v in PITCH_CLAVE_HZ:
        return PITCH_CLAVE_HZ[v]
    n = _numero(v)
    if n is None or v[0] not in "+-":
        return 0  # valores absolutos ("150Hz") no se pueden trasladar a un desplazamiento
    if v.endswith("st"):
        return int(n * HZ_POR_SEMITONO)
    if v.endswith("%"):
        return int(n * HZ_POR_PORCENTAJE)
    return int(n)  # "+5Hz"


def _volume(valor: str) -> int:
    v = valor.strip().lower()
    if v in VOLUME_CLAVE:
        return VOLUME_CLAVE[v]
    n = _numero(v)
    if n is None:
        return 0
    if v.endswith("db"):
        return int((10 ** (n / 20) - 1) * 100)
    return int(n) if v[0] in "+-" else int(n - 100)


def _duracion_pausa(elem: ET.Element) -> int:
    tiempo = (elem.get("time") or "").strip().lower()
    if tiempo:
        n = _numero(tiempo) or 0
        ms = n if tiempo.endswith("ms") else n * 1000
        return int(max(0, min(ms, PAUSA_MAXIMA_MS)))
    return PAUSA_POR_FUERZA.get((elem.get("strength") or "medium").lower(), 400)


def segmentar(ssml: str) -> list[dict]:
    """
    Convierte SSML (o texto plano) en segmentos de texto con prosodia y pausas.

    - `<prosody>` y `<emphasis>` anidados se acumulan (rate/pitch/volume relativos).
    - `<break time="300ms"/>` o `strength="..."` -> pausa.
    - `<say-as interpret-as="characters|spell-out|verbatim">` -> deletreado en palabras.
    - `<sub alias="...">` -> se dice el alias.  `<s>`/`<p>` -> pausa breve al cerrar.
    - Las horas "17:00" se convierten en "cinco de la tarde" (ver `hora_hablada`).
    - Si el SSML no es válido, devuelve el texto sin etiquetas como un único segmento.
    """
    raiz = _parsear(ssml)
    if raiz is None:
        plano = _quitar_etiquetas(ssml)
        return [{"tipo": "texto", "texto": plano, **_Estilo().como_dict()}] if plano else []

    salida: list[dict] = []

    def poner_texto(t: str | None, estilo: _Estilo) -> None:
        if t and t.strip():
            salida.append({"tipo": "texto", "texto": _horas_habladas(t), **estilo.como_dict()})

    def poner_pausa(ms: int) -> None:
        if ms > 0:
            salida.append({"tipo": "pausa", "ms": ms})

    def recorrer(elem: ET.Element, estilo: _Estilo) -> None:
        nombre = _nombre(elem)

        if nombre == "break":
            poner_pausa(_duracion_pausa(elem))
            return
        if nombre in ("mark", "desc"):
            return
        if nombre == "say-as":
            contenido = "".join(elem.itertext())
            modo = (elem.get("interpret-as") or "").lower()
            if modo in ("characters", "spell-out", "verbatim"):
                contenido = deletrear_texto(contenido)
            poner_texto(contenido, estilo)
            return
        if nombre == "sub":
            poner_texto(elem.get("alias") or "".join(elem.itertext()), estilo)
            return

        if nombre == "prosody":
            estilo = replace(
                estilo,
                rate=estilo.rate + _rate(elem.get("rate", "")),
                pitch=estilo.pitch + _pitch(elem.get("pitch", "")),
                volume=estilo.volume + _volume(elem.get("volume", "")),
            )
        elif nombre == "emphasis":
            d_rate, d_vol, d_pitch = ENFASIS.get((elem.get("level") or "moderate").lower(),
                                                 ENFASIS["moderate"])
            estilo = replace(estilo, rate=estilo.rate + d_rate, volume=estilo.volume + d_vol,
                             pitch=estilo.pitch + d_pitch)

        poner_texto(elem.text, estilo)
        for hijo in elem:
            recorrer(hijo, estilo)
            poner_texto(hijo.tail, estilo)

        if nombre in ("s", "sentence"):
            poner_pausa(PAUSA_ORACION_MS)
        elif nombre in ("p", "paragraph"):
            poner_pausa(PAUSA_PARRAFO_MS)

    recorrer(raiz, _Estilo())
    return _compactar(salida)


def _compactar(segmentos: list[dict]) -> list[dict]:
    """Une textos contiguos con la misma prosodia y pausas contiguas (se queda la mayor);
    quita las pausas de los bordes."""
    resultado: list[dict] = []
    for seg in segmentos:
        if seg["tipo"] == "texto":
            seg = {**seg, "texto": _limpiar_espacios(seg["texto"])}
            if not re.search(r"\w", seg["texto"]):
                # Solo puntuación (p. ej. el "." tras una matrícula): se pega al texto anterior.
                if resultado and resultado[-1]["tipo"] == "texto":
                    resultado[-1]["texto"] = _limpiar_espacios(resultado[-1]["texto"] + seg["texto"])
                continue
        previo = resultado[-1] if resultado else None
        if previo and previo["tipo"] == seg["tipo"] == "pausa":
            previo["ms"] = max(previo["ms"], seg["ms"])  # </s></p> seguidos: manda la mayor
        elif (previo and previo["tipo"] == seg["tipo"] == "texto"
              and all(previo[k] == seg[k] for k in ("rate", "pitch", "volume"))):
            previo["texto"] = _limpiar_espacios(previo["texto"] + " " + seg["texto"])
        else:
            resultado.append(dict(seg))

    while resultado and resultado[0]["tipo"] == "pausa":
        resultado.pop(0)
    while resultado and resultado[-1]["tipo"] == "pausa":
        resultado.pop()
    return resultado


def ssml_para_elevenlabs(ssml: str) -> str:
    """
    ElevenLabs solo entiende `<break time="1.5s" />` (máx. 3 s): se conservan las pausas,
    el resto de etiquetas se aplana (los deletreos se convierten en palabras).
    """
    partes = []
    for seg in segmentar(ssml):
        if seg["tipo"] == "pausa":
            partes.append(f'<break time="{min(seg["ms"], 3000) / 1000:.2f}s" />')
        else:
            partes.append(seg["texto"])
    return " ".join(partes)
