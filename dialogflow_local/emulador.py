"""Emulador local de Dialogflow ES.

Carga el export oficial del agente (``agente_dialogflow/``) y reproduce la semántica de
Dialogflow ES lo bastante bien como para probar el webhook real de extremo a extremo sin
conexión a Google:

* **Detección de intenciones** a partir de las frases de entrenamiento: similitud de
  n-gramas de caracteres (TF-IDF + coseno) combinada con ``rapidfuzz`` sobre texto
  normalizado (minúsculas, sin tildes) y con las entidades sustituidas por marcadores
  (``"cita para el 1234 BCD"`` -> ``"cita para el entmatricula"``), igual que una frase
  anotada. Umbral = ``mlMinConfidence`` del agente.
* **Contextos**: una intención con contextos de entrada solo es candidata si TODOS están
  activos, y entonces recibe un plus frente a las globales. Los lifespans se decrementan
  en cada petición; lifespan 0 elimina el contexto. Los parámetros de la intención se
  copian a sus contextos de salida (con ``<param>.original``).
* **Follow-ups**, **eventos** (``WELCOME``, ``SOLICITAR_CITA``...) e intención **fallback**.
* **Entidades**: regex, entidades propias con sinónimos (sin tildes, coincidencia más
  larga), ``@sys.date`` en español (devuelve ``2026-11-09T12:00:00+01:00`` como Dialogflow)
  y ``@sys.geo-city``.
* **Slot filling** con prompts, ``defaultValue`` (``#EVENTO.param`` y ``#contexto.param``) y
  llamada al webhook en cada turno si ``webhookForSlotFilling``.
* **Webhook** con la petición en formato ES v2 exacto, timeout de 5 s (si falla se usa la
  respuesta estática), ``outputContexts`` y ``followupEventInput`` (salto a la intención
  del evento, como hace Dialogflow).

El estado de cada sesión (contextos y slot filling en curso) se guarda en memoria.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import random
import re
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from rapidfuzz import fuzz

# Tipo del webhook: recibe la petición ES v2 (dict) y devuelve la respuesta (dict).
Webhook = Callable[[dict], dict]

RUTA_AGENTE_POR_DEFECTO = Path(__file__).resolve().parent.parent / "agente_dialogflow"

TIMEOUT_WEBHOOK = 5.0          # Dialogflow ES corta el webhook a los 5 segundos
MAX_SALTOS_EVENTO = 3          # máximo de followupEventInput encadenados en una petición
SUELO_SIMILITUD = 0.2          # similitud bruta típica entre frases sin relación (ver Clasificador)
PLUS_CONTEXTO = 0.12          # plus de puntuación para intenciones con contexto de entrada
UMBRAL_INTERRUPCION = 0.7      # confianza para abandonar un slot filling por otra intención
PALABRAS_CANCELAR = {"cancelar", "cancela", "olvidalo", "olvidala", "dejalo", "dejala", "salir", "para"}


# =============================================================================
# Normalización de texto
# =============================================================================

def normalizar(texto: str) -> str:
    """Pasa a minúsculas, quita tildes y cambia la puntuación por espacios.

    Conserva la LONGITUD del texto (un carácter de entrada -> un carácter de salida), de modo
    que las posiciones encontradas en el texto normalizado sirven para recortar el original.
    Se conserva ``/`` para fechas del tipo ``20/11``.
    """
    salida = []
    for c in texto:
        base = unicodedata.normalize("NFD", c)[0].lower()[:1] or " "
        salida.append(base if (base.isalnum() or base == "/") else " ")
    return "".join(salida)


def compactar(texto: str) -> str:
    """Colapsa espacios repetidos."""
    return " ".join(texto.split())


def patron_frase(frase_normalizada: str) -> str:
    """Patrón regex que encuentra una frase normalizada como palabras completas."""
    palabras = [re.escape(p) for p in frase_normalizada.split()]
    return r"(?<![a-z0-9])" + r"\s+".join(palabras) + r"(?![a-z0-9])"


def marcador(tipo: str) -> str:
    """Marcador que sustituye a una entidad: ``@sys.date`` -> ``entsysdate``."""
    return "ent" + re.sub(r"[^a-z0-9]", "", tipo.lower())


# =============================================================================
# Entidades
# =============================================================================

@dataclass
class Coincidencia:
    """Entidad encontrada en el texto (posiciones sobre el texto normalizado)."""
    inicio: int
    fin: int
    tipo: str        # "@matricula", "@sys.date"...
    valor: Any       # valor resuelto ("manana", "2026-11-09T12:00:00+01:00", "Madrid"...)
    original: str    # trozo del texto original
    prioridad: int = 5


# Ciudades para @sys.geo-city: nombre canónico -> variantes que puede decir el usuario.
CIUDADES: dict[str, list[str]] = {
    "Madrid": [], "Barcelona": [], "Valencia": [], "Sevilla": [], "Bilbao": [], "Zaragoza": [],
    "Málaga": [], "Gijón": [], "Oviedo": [], "A Coruña": ["la coruña", "coruña"],
    "Salamanca": [], "Murcia": [], "Toledo": [], "Alicante": [], "Granada": [], "Córdoba": [],
    "Valladolid": [], "Vigo": [], "Santander": [], "Pamplona": ["iruña"],
    "San Sebastián": ["donostia"], "Vitoria": ["vitoria-gasteiz", "gasteiz"],
    "Palma": ["palma de mallorca"], "Las Palmas de Gran Canaria": ["las palmas"],
    "Santa Cruz de Tenerife": [], "Cádiz": [], "Huelva": [], "Almería": [], "Jaén": [],
    "Badajoz": [], "Cáceres": [], "Logroño": [], "Burgos": [], "León": [],
    "Santiago de Compostela": ["santiago"], "Pontevedra": [], "Lugo": [], "Ourense": ["orense"],
    "Girona": ["gerona"], "Lleida": ["lérida"], "Tarragona": [], "Castellón de la Plana": ["castellón"],
    "Albacete": [], "Ciudad Real": [], "Cuenca": [], "Guadalajara": [], "Segovia": [], "Ávila": [],
    "Soria": [], "Teruel": [], "Huesca": [], "Palencia": [], "Zamora": [], "Cartagena": [],
    "Elche": [], "Jerez de la Frontera": ["jerez"], "Marbella": [], "Avilés": [], "Getafe": [],
    "Móstoles": [], "Alcalá de Henares": [], "Mérida": [], "Ceuta": [], "Melilla": [],
}

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
NUMEROS = {"un": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
           "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "quince": 15}

# Prioridad al resolver solapes de igual longitud (menor = gana).
PRIORIDAD = {"preferida": 0, "regex": 1, "propia": 2, "fecha": 3, "ciudad": 4}


class ExtractorEntidades:
    """Encuentra entidades (propias y de sistema) en un texto en español."""

    def __init__(self, entidades: dict[str, dict], zona: ZoneInfo):
        self.zona = zona
        self.regex: list[tuple[str, re.Pattern]] = []          # (tipo, patrón)
        self.sinonimos: list[tuple[str, str, re.Pattern]] = []  # (tipo, valor, patrón)
        self.valores: dict[str, list[tuple[str, re.Pattern]]] = {}  # valores "a pelo" por tipo
        for nombre, ent in entidades.items():
            tipo = "@" + nombre
            for entrada in ent["entradas"]:
                if ent["regex"]:
                    # Sobre texto normalizado (minúsculas): se compila sin distinguir mayúsculas.
                    self.regex.append((tipo, re.compile(
                        r"(?<![a-z0-9])(?:" + entrada["value"] + r")(?![a-z0-9])", re.IGNORECASE)))
                    continue
                for sin in set(entrada.get("synonyms") or [entrada["value"]]):
                    sin_n = compactar(normalizar(sin))
                    if sin_n:
                        self.sinonimos.append((tipo, entrada["value"], re.compile(patron_frase(sin_n))))
                valor_n = compactar(normalizar(entrada["value"]))
                self.valores.setdefault(tipo, []).append((entrada["value"], re.compile(patron_frase(valor_n))))
        self.ciudades: list[tuple[str, re.Pattern]] = []
        for canonica, variantes in CIUDADES.items():
            for v in {canonica, *variantes}:
                self.ciudades.append((canonica, re.compile(patron_frase(compactar(normalizar(v))))))

    # ------------------------------------------------------------------ API
    def extraer(self, texto: str, ahora: datetime, preferida: str | None = None,
                tipos: set[str] | None = None) -> list[Coincidencia]:
        """Devuelve las entidades del texto, sin solapes, ordenadas por posición.

        ``preferida``: tipo de entidad que se está pidiendo en el slot filling. Gana los empates
        y, si la respuesta es muy corta, se acepta también el valor "a pelo" (``"tarde"``).
        ``tipos``: si se indica, solo se devuelven entidades de esos tipos.
        """
        norm = normalizar(texto)
        candidatas: list[Coincidencia] = []

        for tipo, patron in self.regex:
            for m in patron.finditer(norm):
                valor = compactar(texto[m.start():m.end()]).upper()
                candidatas.append(Coincidencia(m.start(), m.end(), tipo, valor, texto[m.start():m.end()],
                                               PRIORIDAD["regex"]))
        for tipo, valor, patron in self.sinonimos:
            for m in patron.finditer(norm):
                candidatas.append(Coincidencia(m.start(), m.end(), tipo, valor, texto[m.start():m.end()],
                                               PRIORIDAD["propia"]))
        if preferida in self.valores and len(norm.split()) <= 2:
            # Respuesta corta a un prompt ("tarde", "mañana"): vale el valor de la entrada tal cual.
            for valor, patron in self.valores[preferida]:
                for m in patron.finditer(norm):
                    candidatas.append(Coincidencia(m.start(), m.end(), preferida, valor,
                                                   texto[m.start():m.end()], PRIORIDAD["preferida"]))
        for ini, fin, valor in self._fechas(norm, ahora):
            candidatas.append(Coincidencia(ini, fin, "@sys.date", valor, texto[ini:fin], PRIORIDAD["fecha"]))
        for canonica, patron in self.ciudades:
            for m in patron.finditer(norm):
                candidatas.append(Coincidencia(m.start(), m.end(), "@sys.geo-city", canonica,
                                               texto[m.start():m.end()], PRIORIDAD["ciudad"]))

        if tipos is not None:
            candidatas = [c for c in candidatas if c.tipo in tipos]
        for c in candidatas:
            if c.tipo == preferida:
                c.prioridad = PRIORIDAD["preferida"]
        # Resolución de solapes: primero la coincidencia más larga y, a igual longitud, la de
        # mayor prioridad. Así "por la mañana" (franja) gana a "mañana" (fecha).
        candidatas.sort(key=lambda c: (-(c.fin - c.inicio), c.prioridad, c.inicio))
        elegidas: list[Coincidencia] = []
        for c in candidatas:
            if all(c.fin <= e.inicio or c.inicio >= e.fin for e in elegidas):
                elegidas.append(c)
        return sorted(elegidas, key=lambda c: c.inicio)

    # ------------------------------------------------------------------ @sys.date
    def _iso(self, dia: date) -> str:
        """Formato de Dialogflow ES para @sys.date: mediodía en la zona horaria del agente."""
        return datetime(dia.year, dia.month, dia.day, 12, 0, 0, tzinfo=self.zona).isoformat()

    def _fechas(self, norm: str, ahora: datetime) -> list[tuple[int, int, str]]:
        """Fechas en español sobre texto normalizado -> [(inicio, fin, iso)]."""
        hoy = ahora.astimezone(self.zona).date()
        res: list[tuple[int, int, str]] = []
        b, e = r"(?<![a-z0-9])", r"(?![a-z0-9])"
        articulo = r"(?:(?:el|al|del|este|para\s+el|para)\s+)?"

        def anadir(m: re.Match, dia: date | None):
            if dia is not None:
                res.append((m.start(), m.end(), self._iso(dia)))

        for m in re.finditer(b + r"(?:para\s+)?pasado\s+manana" + e, norm):
            anadir(m, hoy + timedelta(days=2))
        # "mañana" como fecha. "por la mañana", "de mañana"... los captura antes @franja por
        # ser coincidencias más largas (ver resolución de solapes).
        for m in re.finditer(b + r"(?:para\s+)?manana" + e, norm):
            anadir(m, hoy + timedelta(days=1))
        for m in re.finditer(b + r"(?:para\s+)?hoy" + e, norm):
            anadir(m, hoy)
        # "el lunes", "el próximo jueves", "este viernes", "el martes que viene", "al viernes"
        dias = "|".join(DIAS_SEMANA)
        for m in re.finditer(b + articulo + r"(?:proximo\s+)?(" + dias + r")(?:\s+que\s+viene)?" + e, norm):
            objetivo = DIAS_SEMANA.index(m.group(1))
            delta = (objetivo - hoy.weekday()) % 7 or 7   # siguiente aparición (nunca hoy)
            anadir(m, hoy + timedelta(days=delta))
        # "15 de noviembre", "el día 3 de diciembre de 2026"
        meses = "|".join(MESES)
        for m in re.finditer(b + articulo + r"(?:dia\s+)?(\d{1,2})\s+de\s+(" + meses + r")"
                             r"(?:\s+(?:de|del)\s+(\d{4}))?" + e, norm):
            anadir(m, self._dia_mes(hoy, int(m.group(1)), MESES.index(m.group(2)) + 1,
                                    int(m.group(3)) if m.group(3) else None))
        # "20/11", "20/11/2026"
        for m in re.finditer(b + articulo + r"(?:dia\s+)?(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?" + e, norm):
            anio = int(m.group(3)) if m.group(3) else None
            if anio is not None and anio < 100:
                anio += 2000
            anadir(m, self._dia_mes(hoy, int(m.group(1)), int(m.group(2)), anio))
        # "el día 20" (de este mes, o del siguiente si ya ha pasado)
        for m in re.finditer(b + articulo + r"dia\s+(\d{1,2})" + e, norm):
            anadir(m, self._dia_suelto(hoy, int(m.group(1))))
        # "dentro de 3 días", "en dos semanas"
        cantidad = r"(\d{1,2}|" + "|".join(NUMEROS) + r")"
        for m in re.finditer(b + r"(?:dentro\s+de|en)\s+" + cantidad + r"\s+(dias?|semanas?)" + e, norm):
            n = int(m.group(1)) if m.group(1).isdigit() else NUMEROS[m.group(1)]
            anadir(m, hoy + timedelta(days=n * (7 if m.group(2).startswith("semana") else 1)))
        for m in re.finditer(b + r"(?:la\s+)?semana\s+que\s+viene" + e, norm):
            anadir(m, hoy + timedelta(days=7 - hoy.weekday()))   # lunes de la semana siguiente
        return res

    @staticmethod
    def _dia_mes(hoy: date, dia: int, mes: int, anio: int | None) -> date | None:
        """Día/mes; sin año, la próxima vez que llegue esa fecha."""
        try:
            fecha = date(anio or hoy.year, mes, dia)
            if anio is None and fecha < hoy:
                fecha = date(hoy.year + 1, mes, dia)
            return fecha
        except ValueError:
            return None

    @staticmethod
    def _dia_suelto(hoy: date, dia: int) -> date | None:
        """"El día 20": este mes si aún no ha pasado; si no, el mes siguiente."""
        anio, mes = hoy.year, hoy.month
        for _ in range(2):
            try:
                fecha = date(anio, mes, dia)
                if fecha >= hoy:
                    return fecha
            except ValueError:
                pass
            anio, mes = (anio + 1, 1) if mes == 12 else (anio, mes + 1)
        return None


# =============================================================================
# Modelo del agente
# =============================================================================

@dataclass
class Intencion:
    """Una intención del export, con lo necesario para emularla."""
    id: str
    nombre: str
    contextos_entrada: list[str]
    contextos_salida: list[tuple[str, int]]
    parametros: list[dict]
    respuestas: list[str]
    accion: str
    webhook: bool
    webhook_slot: bool
    fallback: bool
    eventos: list[str]
    padre: str | None
    reset_contextos: bool = False
    frases: list[list[dict]] = field(default_factory=list)   # partes de cada frase de entrenamiento

    @property
    def tipos(self) -> set[str]:
        """Tipos de entidad que maneja la intención (parámetros y anotaciones)."""
        tipos = {p["dataType"] for p in self.parametros}
        for frase in self.frases:
            tipos.update(parte["meta"] for parte in frase if parte.get("meta"))
        return tipos


def cargar_agente(ruta: Path) -> tuple[dict, dict[str, dict], list[Intencion]]:
    """Lee agent.json, entities/ e intents/ del export de Dialogflow ES."""
    ruta = Path(ruta)
    agente = json.loads((ruta / "agent.json").read_text(encoding="utf-8"))
    idioma = agente.get("language", "es")

    entidades: dict[str, dict] = {}
    for fichero in sorted((ruta / "entities").glob("*.json")):
        if "_entries_" in fichero.stem:
            continue
        ent = json.loads(fichero.read_text(encoding="utf-8"))
        entradas_f = fichero.with_name(f"{fichero.stem}_entries_{idioma}.json")
        entradas = json.loads(entradas_f.read_text(encoding="utf-8")) if entradas_f.exists() else []
        entidades[ent["name"]] = {"regex": ent.get("isRegexp", False), "entradas": entradas}

    intenciones: list[Intencion] = []
    for fichero in sorted((ruta / "intents").glob("*.json")):
        if "_usersays_" in fichero.stem:
            continue
        d = json.loads(fichero.read_text(encoding="utf-8"))
        resp = (d.get("responses") or [{}])[0]
        textos: list[str] = []
        for msg in resp.get("messages", []):
            if str(msg.get("type")) == "0" and msg.get("lang", idioma) == idioma:
                speech = msg.get("speech", [])
                textos.extend([speech] if isinstance(speech, str) else speech)
        usersays = fichero.with_name(f"{fichero.stem}_usersays_{idioma}.json")
        frases = []
        if usersays.exists():
            frases = [u["data"] for u in json.loads(usersays.read_text(encoding="utf-8"))]
        intenciones.append(Intencion(
            id=d["id"], nombre=d["name"], contextos_entrada=list(d.get("contexts", [])),
            contextos_salida=[(c["name"], int(c.get("lifespan", 5))) for c in resp.get("affectedContexts", [])],
            parametros=list(resp.get("parameters", [])), respuestas=[t for t in textos if t],
            accion=resp.get("action", ""), webhook=d.get("webhookUsed", False),
            webhook_slot=d.get("webhookForSlotFilling", False), fallback=d.get("fallbackIntent", False),
            eventos=[e["name"] for e in d.get("events", [])], padre=d.get("parentId"),
            reset_contextos=resp.get("resetContexts", False), frases=frases,
        ))
    return agente, entidades, intenciones


# =============================================================================
# Clasificador de intenciones
# =============================================================================

class Clasificador:
    """Similitud entre la consulta y las frases de entrenamiento de cada intención.

    Cada frase se convierte en una "plantilla": texto normalizado donde las partes anotadas
    (y cualquier entidad de los tipos que maneja la intención) se sustituyen por un marcador.
    La consulta se transforma igual, pero solo con los tipos de cada intención, para que
    "¿cubre el granizo?" siga pareciéndose a las frases de consulta_documental (que no tiene
    parámetros) aunque "granizo" sea un valor de @tipo_dano.

    similitud = 0.6 * coseno(TF-IDF de n-gramas de caracteres 3-4) + 0.4 * token_sort_ratio
    confianza = similitud recalibrada descontando el "suelo" de frases sin relación
    """

    def __init__(self, intenciones: list[Intencion], extractor: ExtractorEntidades, ahora: datetime):
        self.extractor = extractor
        self.plantillas: dict[str, list[str]] = {}
        for it in intenciones:
            tipos = it.tipos
            plantillas = []
            for frase in it.frases:
                trozos = []
                for parte in frase:
                    if parte.get("meta") and parte.get("alias"):
                        trozos.append(marcador(parte["meta"]))
                    else:
                        trozos.append(self.sustituir(parte["text"], tipos, ahora))
                plantillas.append(compactar(" ".join(trozos)))
            self.plantillas[it.nombre] = [p for p in plantillas if p]
        # IDF global sobre todas las plantillas
        documentos = [p for lista in self.plantillas.values() for p in lista]
        df: Counter = Counter()
        for doc in documentos:
            df.update(set(self._ngramas(doc)))
        n = max(len(documentos), 1)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1.0 for g, c in df.items()}
        self.idf_max = math.log(1 + n) + 1.0
        self.vectores = {nombre: [self._vector(p) for p in lista] for nombre, lista in self.plantillas.items()}

    def sustituir(self, texto: str, tipos: set[str], ahora: datetime) -> str:
        """Texto normalizado con las entidades de ``tipos`` cambiadas por su marcador."""
        norm = normalizar(texto)
        trozos, pos = [], 0
        for c in self.extractor.extraer(texto, ahora, tipos=tipos):
            trozos.append(norm[pos:c.inicio])
            trozos.append(f" {marcador(c.tipo)} ")
            pos = c.fin
        trozos.append(norm[pos:])
        return compactar("".join(trozos))

    @staticmethod
    def _ngramas(texto: str) -> Counter:
        gramas: Counter = Counter()
        for palabra in texto.split():
            p = f" {palabra} "
            for n in (3, 4):
                gramas.update(p[i:i + n] for i in range(len(p) - n + 1))
            gramas["w:" + palabra] += 1
        return gramas

    def _vector(self, texto: str) -> dict[str, float]:
        v = {g: (1 + math.log(c)) * self.idf.get(g, self.idf_max) for g, c in self._ngramas(texto).items()}
        norma = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norma for g, x in v.items()}

    def similitud(self, consulta: str, nombre_intencion: str) -> float:
        """Máxima similitud de la consulta (ya sustituida) con las plantillas de la intención."""
        if not consulta:
            return 0.0
        vq = self._vector(consulta)
        mejor = 0.0
        for plantilla, vp in zip(self.plantillas[nombre_intencion], self.vectores[nombre_intencion]):
            if plantilla == consulta:
                return 1.0
            coseno = sum(x * vp.get(g, 0.0) for g, x in vq.items())
            mejor = max(mejor, 0.6 * coseno + 0.4 * fuzz.token_sort_ratio(consulta, plantilla) / 100)
        return mejor

    def confianza(self, consulta: str, nombre_intencion: str) -> float:
        """Similitud recalibrada a una "confianza" comparable a la de Dialogflow.

        Dos frases en español sin relación ya comparten n-gramas comunes ("que", " el "...) y
        obtienen ~0.2-0.35 de similitud bruta; se descuenta ese suelo para que el umbral
        ``mlMinConfidence`` (0.3) del agente separe bien lo reconocido de lo desconocido.
        """
        bruta = self.similitud(consulta, nombre_intencion)
        return max(0.0, (bruta - SUELO_SIMILITUD) / (1 - SUELO_SIMILITUD))


# =============================================================================
# Sesiones
# =============================================================================

@dataclass
class SlotFilling:
    """Slot filling en curso: intención, valores recogidos y evento que la disparó."""
    intencion: Intencion
    valores: dict[str, Any]
    originales: dict[str, str]
    evento: tuple[str, dict] | None


@dataclass
class Sesion:
    contextos: dict[str, dict] = field(default_factory=dict)   # nombre -> {"lifespan", "parameters"}
    slot: SlotFilling | None = None

    def activos(self) -> set[str]:
        return {n for n, c in self.contextos.items() if c["lifespan"] > 0}

    def decrementar(self) -> None:
        """Una petición más: todos los contextos pierden 1 de vida; a 0 desaparecen."""
        for nombre in list(self.contextos):
            self.contextos[nombre]["lifespan"] -= 1
            if self.contextos[nombre]["lifespan"] <= 0:
                del self.contextos[nombre]

    def fijar(self, nombre: str, lifespan: int, parametros: dict | None = None) -> None:
        """Crea/actualiza un contexto. Los parámetros vacíos no borran valores anteriores."""
        if lifespan <= 0:
            self.contextos.pop(nombre, None)
            return
        ctx = self.contextos.setdefault(nombre, {"lifespan": lifespan, "parameters": {}})
        ctx["lifespan"] = lifespan
        for k, v in (parametros or {}).items():
            if v not in ("", None, [], {}) or k not in ctx["parameters"]:
                ctx["parameters"][k] = v


# =============================================================================
# Emulador
# =============================================================================

class EmuladorDialogflow:
    """Emulador de Dialogflow ES (API ``detectIntent`` v2) con estado de sesión en memoria."""

    def __init__(self, ruta_agente: str | Path = RUTA_AGENTE_POR_DEFECTO, webhook: Webhook | None = None,
                 ahora: datetime | Callable[[], datetime] | None = None, proyecto: str = "seguros-alonso-local",
                 timeout_webhook: float = TIMEOUT_WEBHOOK, semilla: int | None = None):
        self.agente, entidades, self.intenciones = cargar_agente(Path(ruta_agente))
        self.idioma = self.agente.get("language", "es")
        self.zona = ZoneInfo(self.agente.get("defaultTimezone", "Europe/Madrid"))
        self.umbral = float(self.agente.get("mlMinConfidence", 0.3))
        self.webhook = webhook
        self.timeout_webhook = timeout_webhook
        self.proyecto = proyecto
        self._ahora = ahora
        self._azar = random.Random(semilla)
        self._hilos = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook")
        self.sesiones: dict[str, Sesion] = {}
        self.extractor = ExtractorEntidades(entidades, self.zona)
        self.clasificador = Clasificador(self.intenciones, self.extractor, self.ahora())
        self.por_nombre = {i.nombre: i for i in self.intenciones}

    # ------------------------------------------------------------------ utilidades
    def ahora(self) -> datetime:
        """Momento actual (inyectable en tests)."""
        if callable(self._ahora):
            return self._ahora()
        return self._ahora or datetime.now(self.zona)

    def sesion(self, session_id: str) -> Sesion:
        return self.sesiones.setdefault(session_id, Sesion())

    def reiniciar(self, session_id: str) -> None:
        self.sesiones.pop(session_id, None)

    def _ruta_sesion(self, session_id: str) -> str:
        return f"projects/{self.proyecto}/agent/sessions/{session_id}"

    def _ruta_contexto(self, session_id: str, nombre: str) -> str:
        return f"{self._ruta_sesion(session_id)}/contexts/{nombre}"

    # ------------------------------------------------------------------ API pública
    def detectar_texto(self, session_id: str, texto: str, payload: dict | None = None) -> dict:
        """Equivalente a ``detectIntent`` con ``queryInput.text``. Devuelve la respuesta ES v2."""
        ses = self.sesion(session_id)
        ctx = _Peticion(session_id, texto, payload or {})

        if ses.slot is not None:
            respuesta = self._turno_slot_filling(ses, ctx)
            if respuesta is not None:
                return respuesta

        intencion, confianza = self.clasificar(texto, ses.activos())
        ses.decrementar()
        valores, originales = self._parametros_de_consulta(intencion, texto)
        return self._ejecutar(ses, ctx, intencion, valores, originales, confianza, evento=None)

    def detectar_evento(self, session_id: str, nombre: str, parametros: dict | None = None,
                        payload: dict | None = None) -> dict:
        """Equivalente a ``detectIntent`` con ``queryInput.event``."""
        ses = self.sesion(session_id)
        intencion = self._intencion_de_evento(nombre)
        ses.slot = None                     # un evento cancela el slot filling en curso
        self._borrar_contextos_dialogo(ses)
        ses.decrementar()
        ctx = _Peticion(session_id, nombre, payload or {})
        return self._ejecutar(ses, ctx, intencion, {}, {}, 1.0, evento=(nombre, dict(parametros or {})))

    def clasificar(self, texto: str, activos: set[str], excluir: str | None = None) -> tuple[Intencion, float]:
        """Mejor intención para el texto dados los contextos activos (o la fallback)."""
        ranking = self.ranking(texto, activos)
        ranking = [(p, i) for p, i in ranking if i.nombre != excluir]
        if ranking and ranking[0][0] >= self.umbral:
            return ranking[0][1], round(min(ranking[0][0], 1.0), 4)
        return self._fallback(activos), 1.0

    def ranking(self, texto: str, activos: set[str]) -> list[tuple[float, Intencion]]:
        """Puntuación de todas las intenciones candidatas, de mayor a menor."""
        ahora = self.ahora()
        resultado = []
        for it in self.intenciones:
            if it.fallback or not self.clasificador.plantillas.get(it.nombre):
                continue
            if not set(it.contextos_entrada) <= activos:
                continue      # faltan contextos de entrada: no es candidata
            consulta = self.clasificador.sustituir(texto, it.tipos, ahora)
            puntuacion = self.clasificador.confianza(consulta, it.nombre)
            if it.contextos_entrada and puntuacion >= self.umbral:
                # Como en ES, las intenciones "en contexto" tienen preferencia sobre las globales
                # (pero el plus no convierte en válida una coincidencia que no lo era).
                puntuacion += PLUS_CONTEXTO
            resultado.append((puntuacion, it))
        resultado.sort(key=lambda x: -x[0])
        return resultado

    # ------------------------------------------------------------------ búsqueda de intenciones
    def _fallback(self, activos: set[str]) -> Intencion:
        candidatas = [i for i in self.intenciones if i.fallback and set(i.contextos_entrada) <= activos]
        candidatas.sort(key=lambda i: -len(i.contextos_entrada))   # la más específica primero
        if not candidatas:
            raise RuntimeError("El agente no tiene intención fallback")
        return candidatas[0]

    def _intencion_de_evento(self, nombre: str) -> Intencion:
        for it in self.intenciones:
            if nombre in it.eventos:
                return it
        raise ValueError(f"Ninguna intención atiende el evento {nombre!r}")

    # ------------------------------------------------------------------ parámetros
    def _parametros_de_consulta(self, intencion: Intencion, texto: str,
                                preferida: str | None = None) -> tuple[dict, dict]:
        """Asigna las entidades encontradas en el texto a los parámetros de la intención."""
        coincidencias = self.extractor.extraer(texto, self.ahora(), preferida=preferida,
                                               tipos={p["dataType"] for p in intencion.parametros})
        usadas: set[int] = set()
        valores, originales = {}, {}
        for p in intencion.parametros:
            mias = [(n, c) for n, c in enumerate(coincidencias) if c.tipo == p["dataType"] and n not in usadas]
            if not mias:
                continue
            if p.get("isList"):
                usadas.update(n for n, _ in mias)
                valores[p["name"]] = [c.valor for _, c in mias]
                originales[p["name"]] = [c.original for _, c in mias]
            else:
                n, c = mias[0]
                usadas.add(n)
                valores[p["name"]] = c.valor
                originales[p["name"]] = c.original
        return valores, originales

    def _resolver_referencia(self, ses: Sesion, ref: str, evento: tuple[str, dict] | None,
                             valores: dict) -> Any:
        """Resuelve ``$param``, ``#EVENTO.param`` y ``#contexto.param`` como Dialogflow."""
        if not isinstance(ref, str) or not ref:
            return ref
        m = re.fullmatch(r"\$([\w\-.]+)", ref)
        if m:
            return valores.get(m.group(1), "")
        m = re.fullmatch(r"#([\w\-]+)\.([\w\-.]+)", ref)
        if m:
            origen, param = m.groups()
            if evento is not None and evento[0] == origen:
                return evento[1].get(param, "")
            ctx = ses.contextos.get(origen)
            return ctx["parameters"].get(param, "") if ctx else ""
        return ref   # valor literal

    def _completar_parametros(self, ses: Sesion, intencion: Intencion, valores: dict,
                              evento: tuple[str, dict] | None) -> dict:
        """Valores finales: lo dicho por el usuario o, si falta, el defaultValue."""
        finales = {}
        for p in intencion.parametros:
            nombre = p["name"]
            valor = valores.get(nombre, "")
            if valor in ("", None, []) and p.get("value", f"${nombre}") != f"${nombre}":
                valor = self._resolver_referencia(ses, p["value"], evento, valores)
            if valor in ("", None, []) and p.get("defaultValue"):
                valor = self._resolver_referencia(ses, p["defaultValue"], evento, valores)
            finales[nombre] = valor if valor is not None else ""
        return finales

    # ------------------------------------------------------------------ slot filling
    def _turno_slot_filling(self, ses: Sesion, ctx: "_Peticion") -> dict | None:
        """Turno con un slot filling en curso. Devuelve None si el usuario se sale del flujo."""
        slot = ses.slot
        it = slot.intencion
        pendientes = [p for p in it.parametros if p.get("required") and slot.valores.get(p["name"]) in ("", None, [])]
        preferida = pendientes[0]["dataType"] if pendientes else None
        nuevos, originales = self._parametros_de_consulta(it, ctx.texto, preferida=preferida)

        if not nuevos:
            palabras = set(compactar(normalizar(ctx.texto)).split())
            if palabras and palabras <= PALABRAS_CANCELAR | {"la", "lo", "solicitud", "cita", "todo"}:
                # Dialogflow sale del slot filling si el usuario cancela.
                ses.slot = None
                self._borrar_contextos_dialogo(ses)
                ses.decrementar()
                return self._respuesta(ctx, it, slot.valores, False, "De acuerdo, cancelo la solicitud.",
                                       [{"text": {"text": ["De acuerdo, cancelo la solicitud."]}}], 1.0, ses)
            otra, confianza = self.clasificar(ctx.texto, ses.activos(), excluir=it.nombre)
            if not otra.fallback and confianza >= UMBRAL_INTERRUPCION:
                # Interrupción: el usuario cambia claramente de tema (p. ej. una consulta).
                ses.slot = None
                self._borrar_contextos_dialogo(ses)
                return None

        slot.valores.update(nuevos)
        slot.originales.update(originales)
        ses.decrementar()
        return self._ejecutar(ses, ctx, it, slot.valores, slot.originales, 1.0, evento=slot.evento)

    def _borrar_contextos_dialogo(self, ses: Sesion) -> None:
        for nombre in [n for n in ses.contextos if "_dialog_" in n]:
            del ses.contextos[nombre]

    # ------------------------------------------------------------------ ejecución de una intención
    def _ejecutar(self, ses: Sesion, ctx: "_Peticion", it: Intencion, valores: dict, originales: dict,
                  confianza: float, evento: tuple[str, dict] | None, saltos: int = 0) -> dict:
        """Rellena parámetros, aplica contextos, llama al webhook y construye la respuesta."""
        finales = self._completar_parametros(ses, it, valores, evento)
        pendientes = [p for p in it.parametros if p.get("required") and finales[p["name"]] in ("", None, [])]
        completos = not pendientes
        con_original = dict(finales)
        for k, v in originales.items():
            con_original[f"{k}.original"] = v

        if completos:
            ses.slot = None
            self._borrar_contextos_dialogo(ses)
            if it.reset_contextos:
                ses.contextos.clear()
            for nombre, vida in it.contextos_salida:
                ses.fijar(nombre, vida, con_original)
            texto_estatico = self._respuesta_estatica(ses, it, finales)
            llamar = it.webhook
        else:
            # Slot filling: Dialogflow guarda el estado en contextos "<intent>_dialog_*".
            ses.slot = SlotFilling(it, dict(valores), dict(originales), evento)
            base = re.sub(r"\W+", "_", it.nombre.lower())
            self._borrar_contextos_dialogo(ses)
            ses.fijar(f"{base}_dialog_context", 2, con_original)
            ses.fijar(f"{base}_dialog_params_{pendientes[0]['name']}", 1, con_original)
            prompts = [pr["value"] for pr in pendientes[0].get("prompts", []) if pr.get("lang", self.idioma) == self.idioma]
            texto_estatico = self._azar.choice(prompts) if prompts else ""
            llamar = it.webhook and it.webhook_slot

        texto, mensajes = texto_estatico, [{"text": {"text": [texto_estatico]}}] if texto_estatico else []
        estado_webhook, payload_webhook = None, None
        if llamar and self.webhook is not None:
            peticion = self._peticion_webhook(ses, ctx, it, finales, completos, texto, mensajes, confianza)
            datos, estado_webhook, latencia = self._llamar_webhook(peticion)
            ctx.latencia_webhook += latencia
            if datos is not None:
                # Contextos devueltos por el webhook (nombre completo o corto)
                for c in datos.get("outputContexts", []) or []:
                    nombre = c.get("name", "").rsplit("/contexts/", 1)[-1]
                    if nombre:
                        ses.fijar(nombre, int(c.get("lifespanCount", 0) or 0), c.get("parameters") or {})
                payload_webhook = datos.get("payload")
                # Evento de seguimiento: Dialogflow dispara la intención del evento y devuelve SU respuesta.
                evento_sig = datos.get("followupEventInput")
                if evento_sig and evento_sig.get("name") and saltos < MAX_SALTOS_EVENTO:
                    destino = self._intencion_de_evento(evento_sig["name"])
                    ses.slot = None
                    self._borrar_contextos_dialogo(ses)
                    return self._ejecutar(ses, ctx, destino, {}, {}, 1.0,
                                          evento=(evento_sig["name"], dict(evento_sig.get("parameters") or {})),
                                          saltos=saltos + 1)
                texto_wh = datos.get("fulfillmentText") or ""
                mensajes_wh = datos.get("fulfillmentMessages") or []
                if not texto_wh:
                    texto_wh = next((" ".join(m["text"].get("text", [])) for m in mensajes_wh
                                     if "text" in m and m["text"].get("text")), "")
                if texto_wh or mensajes_wh:
                    texto, mensajes = texto_wh, (mensajes_wh or [{"text": {"text": [texto_wh]}}])
        return self._respuesta(ctx, it, finales, completos, texto, mensajes, confianza, ses,
                               estado_webhook, payload_webhook)

    def _respuesta_estatica(self, ses: Sesion, it: Intencion, valores: dict) -> str:
        """Respuesta de texto de la intención, con ``$param`` y ``#ctx.param`` sustituidos."""
        if not it.respuestas:
            return ""
        plantilla = self._azar.choice(it.respuestas)

        def sustituir(m: re.Match) -> str:
            return str(self._resolver_referencia(ses, m.group(0), None, valores) or "")
        return re.sub(r"\$[\w\-]+|#[\w\-]+\.[\w\-]+", sustituir, plantilla)

    # ------------------------------------------------------------------ webhook
    def _contextos_salida(self, ses: Sesion, session_id: str) -> list[dict]:
        return [{"name": self._ruta_contexto(session_id, n), "lifespanCount": c["lifespan"],
                 "parameters": dict(c["parameters"])}
                for n, c in ses.contextos.items() if c["lifespan"] > 0]

    def _peticion_webhook(self, ses: Sesion, ctx: "_Peticion", it: Intencion, parametros: dict,
                          completos: bool, texto: str, mensajes: list, confianza: float) -> dict:
        """WebhookRequest de Dialogflow ES v2."""
        query_result = {
            "queryText": ctx.texto,
            "action": it.accion,
            "parameters": parametros,
            "allRequiredParamsPresent": completos,
            "fulfillmentText": texto,
            "fulfillmentMessages": mensajes,
            "outputContexts": self._contextos_salida(ses, ctx.session_id),
            "intent": {"name": f"projects/{self.proyecto}/agent/intents/{it.id}", "displayName": it.nombre},
            "intentDetectionConfidence": confianza,
            "languageCode": self.idioma,
        }
        if not it.accion:
            del query_result["action"]
        return {
            "responseId": str(uuid.uuid4()),
            "queryResult": query_result,
            "originalDetectIntentRequest": {"source": "local", "payload": dict(ctx.payload)},
            "session": self._ruta_sesion(ctx.session_id),
        }

    def _llamar_webhook(self, peticion: dict) -> tuple[dict | None, dict, float]:
        """Llama al webhook con timeout. Devuelve (respuesta | None, webhookStatus, latencia)."""
        t0 = time.perf_counter()
        futuro = self._hilos.submit(self.webhook, peticion)
        try:
            datos = futuro.result(timeout=self.timeout_webhook)
            if not isinstance(datos, dict):
                raise TypeError("el webhook no devolvió un objeto JSON")
            estado = {"code": 0, "message": "Webhook execution successful"}
        except concurrent.futures.TimeoutError:
            datos, estado = None, {"code": 4, "message": "Webhook call failed. Error: DEADLINE_EXCEEDED."}
        except Exception as exc:  # noqa: BLE001 - cualquier fallo del webhook -> respuesta estática
            datos, estado = None, {"code": 14, "message": f"Webhook call failed. Error: {exc}"}
        return datos, estado, time.perf_counter() - t0

    # ------------------------------------------------------------------ respuesta
    def _respuesta(self, ctx: "_Peticion", it: Intencion, parametros: dict, completos: bool, texto: str,
                   mensajes: list, confianza: float, ses: Sesion, estado_webhook: dict | None = None,
                   payload_webhook: dict | None = None) -> dict:
        """DetectIntentResponse de Dialogflow ES v2 (como JSON)."""
        query_result = {
            "queryText": ctx.texto,
            "action": it.accion,
            "parameters": parametros,
            "allRequiredParamsPresent": completos,
            "fulfillmentText": texto,
            "fulfillmentMessages": mensajes,
            "outputContexts": self._contextos_salida(ses, ctx.session_id),
            "intent": {"name": f"projects/{self.proyecto}/agent/intents/{it.id}", "displayName": it.nombre,
                       **({"isFallback": True} if it.fallback else {})},
            "intentDetectionConfidence": confianza,
            "diagnosticInfo": {"webhook_latency_ms": round(ctx.latencia_webhook * 1000, 1)},
            "languageCode": self.idioma,
        }
        if payload_webhook:
            query_result["webhookPayload"] = payload_webhook
        respuesta = {"responseId": str(uuid.uuid4()), "queryResult": query_result}
        if estado_webhook is not None:
            respuesta["webhookStatus"] = estado_webhook
        # Diagnóstico propio del emulador (no existe en Dialogflow): latencias en segundos.
        respuesta["_emulador"] = {"latencia_webhook": ctx.latencia_webhook,
                                  "latencia_total": time.perf_counter() - ctx.t0}
        return respuesta


@dataclass
class _Peticion:
    """Datos de una petición detectIntent en curso."""
    session_id: str
    texto: str
    payload: dict
    t0: float = field(default_factory=time.perf_counter)
    latencia_webhook: float = 0.0
