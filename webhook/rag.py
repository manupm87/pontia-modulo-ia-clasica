"""
Consulta documental (RAG) sobre los documentos OCR de Seguros Alonso, con control de alucinaciones.

Flujo de MotorDocumental.responder(pregunta):

  1. Recuperación: embedding de la pregunta (Ollama bge-m3) y búsqueda de los
     fragmentos más parecidos en ChromaDB (similitud coseno). Si ChromaDB o los
     embeddings no están disponibles, se usa una búsqueda por palabras clave
     (la del código del curso, mejorada) para no dejar al usuario sin respuesta.
  2. Primer filtro anti-alucinación (SIN llamar al LLM): si el fragmento más
     parecido no supera el umbral de similitud, la pregunta no trata de nada que
     esté en los documentos -> "no consta". Ahorra tiempo y evita que el LLM
     "rellene" con conocimiento propio.
  3. Generación: prompt en español que obliga a responder SOLO con el CONTEXTO y
     con dos líneas: EVIDENCIA (copia literal del trozo que responde) y RESPUESTA
     (1-3 frases aptas para voz) o exactamente NO_CONSTA si el dato no aparece.
     Pedir primero la evidencia hace que un modelo pequeño (7B) se equivoque
     bastante menos al leer el contexto.
  4. Post-control: si el modelo devuelve NO_CONSTA -> respuesta estándar de "no
     dispongo de esa información" + oferta de agente.
  5. Control de anclaje (grounding):
       a) la EVIDENCIA citada debe estar realmente en el contexto (si no, la ha inventado);
       b) toda cifra de la respuesta (importes, plazos, teléfonos...) debe aparecer
          en los fragmentos recuperados o en la propia pregunta.
     Si falla cualquiera de los dos, se descarta la respuesta.

Nunca lanza excepciones y el tiempo total (embedding + LLM) se limita a < 4 s,
porque Dialogflow corta el webhook a los 5 s.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from webhook import llm

log = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent
# 127.0.0.1 en vez de localhost: en Windows "localhost" prueba antes IPv6 y pierde ~2 s por conexión.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODELO_EMBEDDINGS = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
COLECCION = "seguros_alonso"

# --- Umbral de similitud ---
# Calibrado con 22 preguntas respondibles y 14 no respondibles (similitud coseno bge-m3 del MEJOR fragmento):
#   - respondibles: mínimo 0,43 ("¿Qué pasó en el siniestro de Ana?"), mediana ~0,63;
#   - fuera de dominio ("capital de Francia", "quién ganó la liga", "bitcoin", "CIF de Mapfre"): 0,30-0,38;
#   - del dominio pero sin respuesta ("seguro de moto", "seguro de vida", "camión"): 0,52-0,63, es decir,
#     MEZCLADAS con las respondibles: ningún umbral las separa sin perder preguntas buenas.
# Por eso 0,40: descarta lo claramente ajeno sin gastar una llamada al LLM, y el resto lo filtran
# el NO_CONSTA del LLM y el control de anclaje. Por debajo -> no consta, sin llamar al LLM.
UMBRAL_SIMILITUD = float(os.getenv("RAG_UMBRAL", "0.40"))
# Fragmentos que se pasan al LLM: los k mejores que además no se alejen mucho del mejor.
K_CONTEXTO = 4
MARGEN_CONTEXTO = 0.12
# Búsqueda por palabras (modo degradado): fracción mínima de palabras de la pregunta encontradas.
UMBRAL_PALABRAS = 0.5

PRESUPUESTO_SEGUNDOS = 3.8      # embedding + búsqueda + LLM
TIMEOUT_EMBEDDING = 1.0
LLM_TIMEOUT_MINIMO = 0.8
MAX_TOKENS = 260              # evidencia + respuesta

MENSAJE_NO_CONSTA = ("No dispongo de esa información en la documentación de Seguros Alonso. "
                     "¿Quieres que te pase con un agente?")
MENSAJE_ERROR = ("Ahora mismo no puedo consultar la documentación. "
                 "¿Quieres que te pase con un agente?")

SISTEMA = """Eres el asistente telefónico de Seguros Alonso. Respondes usando ÚNICAMENTE el CONTEXTO: fragmentos de la documentación oficial del seguro de automóvil para turismos de uso particular (condiciones generales, tabla de coberturas y precios, guía de peritaje y asistencia, la póliza SA-104233 y el parte de siniestro SIN-2026-004512).

Contesta SIEMPRE con este formato de dos líneas:
EVIDENCIA: <copia literal del trozo del CONTEXTO que responde a la pregunta, o NINGUNA>
RESPUESTA: <la respuesta para el cliente, o NO_CONSTA>

Reglas:
1. Primero busca la evidencia. Si el CONTEXTO contiene el dato, cópialo en EVIDENCIA y responde.
2. Si el CONTEXTO dice que algo no está incluido, no está cubierto o está excluido, ESO ES UNA RESPUESTA: explícalo.
3. Si el CONTEXTO no contiene el dato pedido, o la pregunta es sobre otro tipo de seguro (vida, hogar, salud...), otro tipo de vehículo distinto del turismo, u otro tema ajeno a esta documentación: EVIDENCIA: NINGUNA y RESPUESTA: NO_CONSTA.
4. No uses conocimiento propio. No inventes ni cambies cifras, precios, plazos ni teléfonos: cópialos tal como aparecen.
5. La RESPUESTA va en español, de tú, en una a tres frases completas, cortas y naturales, porque se leerá en voz alta. Sin markdown, sin listas, sin emojis.
6. No deduzcas ni supongas datos del cliente que no estén escritos en el CONTEXTO (edad, permiso de conducir, quién conduce, motivos). Di solo lo que el documento dice.
7. Si la pregunta es sobre la póliza de un cliente concreto (su nombre, "mi póliza", su coche o su matrícula), usa los datos de sus CONDICIONES PARTICULARES (prima, franquicia, conductores) antes que la tabla general de precios. Si solo tienes la tarifa general, di que es la tarifa general.

Ejemplos (documentos inventados, solo para mostrar el formato):
CONTEXTO: Cobertura «Remolque de caravana»: incluida en Todo riesgo; NO incluida en Terceros.
PREGUNTA: ¿Me cubre el remolque de caravana?
EVIDENCIA: Cobertura «Remolque de caravana»: incluida en Todo riesgo; NO incluida en Terceros.
RESPUESTA: Depende de la modalidad: el Todo riesgo lo incluye, pero el seguro a Terceros no.

CONTEXTO: Artículo 12. No se cubren los daños causados por animales domésticos transportados en el vehículo.
PREGUNTA: Si mi perro me raya el asiento, ¿lo cubrís?
EVIDENCIA: No se cubren los daños causados por animales domésticos transportados en el vehículo.
RESPUESTA: No, los daños causados por animales domésticos que viajan en el vehículo están excluidos.

CONTEXTO: Asistencia en viaje: llame al teléfono de asistencia que figura en su tarjeta.
PREGUNTA: ¿Tenéis seguro de viaje para mis vacaciones?
EVIDENCIA: NINGUNA
RESPUESTA: NO_CONSTA"""

PLANTILLA = """CONTEXTO:
{contexto}

PREGUNTA: {pregunta}"""

# Preguntas sobre la póliza de un cliente concreto -> priorizar sus condiciones particulares
DOC_POLIZA = "04_condiciones_particulares_SA-104233.png"
DOC_TARIFAS = "02_tabla_coberturas.png"
RE_POLIZA_CONCRETA = re.compile(
    r"\b(ana|ruiz|mi p[oó]liza|mi seguro|mi coche|mi le[oó]n|seat le[oó]n|sa-?104233|1234\s?bcd|"
    r"mi prima|pago yo|me cobr[aá]is|mi hijo|mi mujer|mi marido)\b", re.IGNORECASE)

RE_NO_CONSTA = re.compile(r"NO[\s_-]*CONSTA", re.IGNORECASE)
RE_EVIDENCIA = re.compile(r"EVIDENCIA\s*:\s*(.*?)\s*(?=RESPUESTA\s*:|$)", re.IGNORECASE | re.DOTALL)
RE_RESPUESTA = re.compile(r"RESPUESTA\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
# Fracción mínima de palabras de la EVIDENCIA que deben estar en el contexto (control de citas inventadas)
MIN_EVIDENCIA_EN_CONTEXTO = 0.7


@dataclass
class RespuestaDocumental:
    texto: str                 # respuesta para el usuario (2-3 frases, apta para voz)
    respondida: bool           # False si no consta en los documentos (o si hubo un error)
    fuentes: list[str] = field(default_factory=list)   # nombres de documento usados
    latencia: float = 0.0


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a al algo como con cual cuales cuando cuanto cuanta cuantos cuantas de del donde el ella en es esa ese eso
esta este esto hay la las le lo los mas me mi mis muy no o para pero por que quien se si sin sobre su sus
tengo tiene tu tus un una uno unos y ya yo puedo puede hace hacer seguro seguros alonso cubre quiero saber
""".split())


def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def palabras_clave(texto: str) -> list[str]:
    return [p for p in re.findall(r"[a-z0-9ñ]+", sin_tildes(texto)) if len(p) >= 3 and p not in STOPWORDS]


RE_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")


def _canon_numero(n: str) -> str:
    """'50.000.000' -> '50000000' ; '285,00' -> '285' ; '612,40' -> '612,4'."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", n):
        n = n.replace(".", "")
    if "," in n:
        n = n.rstrip("0").rstrip(",")
    return n.lstrip("0") or "0"   # '02' (día) == '2'



def numeros_sin_respaldo(respuesta: str, fuentes_texto: str) -> list[str]:
    """
    Cifras de la respuesta que NO aparecen en el contexto ni en la pregunta.
    Los teléfonos se comparan también sin espacios ("900123456" == "900 123 456").
    """
    conocidos = {_canon_numero(n) for n in RE_NUMERO.findall(fuentes_texto)}
    solo_cifras = re.sub(r"\D", "", fuentes_texto)
    sospechosos = []
    for n in RE_NUMERO.findall(respuesta):
        c = _canon_numero(n)
        if c in conocidos or (len(re.sub(r"\D", "", n)) >= 6 and re.sub(r"\D", "", n) in solo_cifras):
            continue
        sospechosos.append(n)
    return sospechosos


def separar_salida(salida: str) -> tuple[str, str]:
    """
    Separa la salida del LLM en (evidencia, respuesta). Si el modelo no respetó el
    formato, toda la salida se considera respuesta y la evidencia queda vacía.
    """
    m_ev, m_resp = RE_EVIDENCIA.search(salida), RE_RESPUESTA.search(salida)
    evidencia = m_ev.group(1).strip() if m_ev else ""
    respuesta = m_resp.group(1).strip() if m_resp else salida.strip()
    return evidencia, respuesta


def cobertura_palabras(fragmento: str, texto: str) -> float:
    """Fracción de las palabras de `fragmento` que aparecen en `texto` (sin tildes ni mayúsculas)."""
    palabras = re.findall(r"[a-z0-9ñ]+", sin_tildes(fragmento))
    if not palabras:
        return 0.0
    vocab = set(re.findall(r"[a-z0-9ñ]+", sin_tildes(texto)))
    return sum(1 for p in palabras if p in vocab) / len(palabras)


def limpiar_para_voz(texto: str) -> str:
    """
    Quita restos de markdown y saltos de línea (la respuesta se convierte en voz) y,
    si el modelo se quedó sin tokens a mitad de frase, recorta hasta la última frase completa.
    """
    t = re.sub(r"[*_#`>]+", "", texto)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in ".!?»)\"":
        corte = max(t.rfind(". "), t.rfind("? "), t.rfind("! "))
        t = t[:corte + 1] if corte > 0 else t + "."
    return t


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class MotorDocumental:
    """Recuperación (ChromaDB + embeddings, o palabras clave) + LLM + controles de alucinación."""

    def __init__(self, ruta_json: str | Path = "data/documentos.json", ruta_chroma: str | Path = "data/chroma"):
        ruta_json, ruta_chroma = Path(ruta_json), Path(ruta_chroma)
        if not ruta_json.is_absolute():
            ruta_json = RAIZ / ruta_json
        if not ruta_chroma.is_absolute():
            ruta_chroma = RAIZ / ruta_chroma

        self.documentos: list[dict] = json.loads(ruta_json.read_text(encoding="utf-8"))
        self.num_documentos = sum(1 for d in self.documentos if d.get("texto"))
        self._http = httpx.Client(timeout=httpx.Timeout(5.0, connect=1.0))

        # Fragmentos en memoria para el modo degradado (mismo troceado que el índice)
        from ocr.indexar import fragmentos_documento
        self.fragmentos = [f for d in self.documentos if d.get("texto") for f in fragmentos_documento(d)]
        self._palabras_fragmentos = [set(palabras_clave(f["texto"])) for f in self.fragmentos]

        self.coleccion = None
        self.modo = "palabras"
        try:
            import chromadb
            from chromadb.config import Settings
            cliente = chromadb.PersistentClient(path=str(ruta_chroma),
                                                settings=Settings(anonymized_telemetry=False))
            try:
                coleccion = cliente.get_collection(COLECCION)
            except Exception:
                # El índice no se versiona (se regenera y cambia en cada ejecución): si no existe,
                # se construye ahora a partir de data/documentos.json (unos segundos con Ollama).
                log.info("No hay índice vectorial en %s: lo construyo con ocr.indexar", ruta_chroma)
                from ocr import indexar
                indexar.main()
                coleccion = cliente.get_collection(COLECCION)
            if coleccion.count() == 0:
                raise RuntimeError("la colección está vacía")
            # Embedding de prueba: comprueba Ollama y deja bge-m3 cargado en memoria
            self._embeber("prueba", timeout=60)
            self._precalentar_embeddings()
            self.coleccion, self.modo = coleccion, "vectorial"
        except Exception as e:
            log.warning("RAG en modo palabras clave (sin ChromaDB/embeddings): %s", e)
        log.info("MotorDocumental listo: %d documentos, %d fragmentos, modo %s",
                 self.num_documentos, len(self.fragmentos), self.modo)

    # -- recuperación ------------------------------------------------------------

    def _embeber(self, texto: str, timeout: float = TIMEOUT_EMBEDDING) -> list[float]:
        r = self._http.post(f"{OLLAMA_URL}/api/embed",
                            json={"model": MODELO_EMBEDDINGS, "input": texto, "keep_alive": "60m"},
                            timeout=httpx.Timeout(timeout, connect=1.0))
        r.raise_for_status()
        return r.json()["embeddings"][0]

    def _precalentar_embeddings(self, max_palabras: int = 60) -> None:
        """
        Calcula un embedding para cada longitud de frase de 1 a `max_palabras` palabras.

        Medido en la RTX 5090: la PRIMERA vez que llega una entrada con un número de
        tokens nuevo, la GPU compila los kernels para ese tamaño y el embedding tarda
        ~1,5-2 s (en vez de ~15 ms). Pasar todas las longitudes típicas de una pregunta
        al arrancar quita ese coste de la primera petición real (~1 s en total).
        """
        t0 = time.perf_counter()
        for n in range(1, max_palabras + 1):
            try:
                self._embeber(" ".join(["seguro"] * n), timeout=10)
            except Exception:
                break
        log.info("Embeddings precalentados en %.1f s", time.perf_counter() - t0)

    def _recuperar_vectorial(self, pregunta: str, k: int, documento: str | None = None) -> list[dict]:
        filtro = {"documento": documento} if documento else None
        res = self.coleccion.query(query_embeddings=[self._embeber(pregunta)], n_results=k, where=filtro,
                                   include=["documents", "metadatas", "distances"])
        return [{"documento": m["documento"], "texto": t, "similitud": round(1 - d, 4)}
                for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def _recuperar_palabras(self, pregunta: str, k: int) -> list[dict]:
        """
        Respaldo sin embeddings: fracción de palabras de la pregunta presentes en el
        fragmento (sin tildes; se aceptan coincidencias por raíz de 5 letras para
        cubrir plurales: "franquicias" ~ "franquicia").
        """
        claves = palabras_clave(pregunta)
        if not claves:
            return []
        puntuados = []
        for frag, vocab in zip(self.fragmentos, self._palabras_fragmentos):
            raices = {v[:5] for v in vocab}
            aciertos = sum(1 for p in claves if p in vocab or p[:5] in raices)
            puntuados.append((aciertos / len(claves), frag))
        puntuados.sort(key=lambda x: x[0], reverse=True)
        return [{"documento": f["documento"], "texto": f["texto"], "similitud": round(s, 4)}
                for s, f in puntuados[:k]]

    def recuperar(self, pregunta: str, k: int = 4) -> list[dict]:
        """Devuelve [{"documento", "texto", "similitud"}] ordenados de más a menos parecido."""
        if self.coleccion is not None:
            try:
                return self._recuperar_vectorial(pregunta, k)
            except Exception as e:
                log.warning("Fallo en la búsqueda vectorial, uso palabras clave: %s", e)
        return self._recuperar_palabras(pregunta, k)

    # -- respuesta ---------------------------------------------------------------

    def responder(self, pregunta: str) -> RespuestaDocumental:
        """Responde con la documentación o dice explícitamente que no consta. Nunca lanza excepción."""
        t0 = time.perf_counter()

        def fin(texto: str, respondida: bool, fuentes: list[str] | None = None) -> RespuestaDocumental:
            return RespuestaDocumental(texto, respondida, fuentes or [], round(time.perf_counter() - t0, 3))

        try:
            pregunta = (pregunta or "").strip()
            if not pregunta:
                return fin(MENSAJE_NO_CONSTA, False)

            # 1) Recuperación
            vectorial = self.coleccion is not None
            try:
                resultados = self._recuperar_vectorial(pregunta, K_CONTEXTO) if vectorial else []
            except Exception as e:
                log.warning("Búsqueda vectorial no disponible (%s); uso palabras clave", e)
                vectorial = False
            if not vectorial:
                resultados = self._recuperar_palabras(pregunta, K_CONTEXTO)
            umbral = UMBRAL_SIMILITUD if vectorial else UMBRAL_PALABRAS

            # 2) Filtro por similitud: no hay nada parecido en los documentos
            if not resultados or resultados[0]["similitud"] < umbral:
                mejor = resultados[0]["similitud"] if resultados else 0
                log.info("RAG: no consta (mejor similitud %.3f < %.2f) para %r", mejor, umbral, pregunta)
                return fin(MENSAJE_NO_CONSTA, False)

            mejor = resultados[0]["similitud"]
            usados = [r for r in resultados if r["similitud"] >= max(umbral, mejor - MARGEN_CONTEXTO)]

            # 2b) Pregunta sobre una póliza concreta ("el Seat León de Ana", "mi póliza"): la tabla
            #     general de precios se parece más a la pregunta que las condiciones particulares,
            #     y el modelo contestaba con la tarifa genérica (589 €) en vez de la prima real
            #     (612,40 €). Añadimos siempre los fragmentos de esa póliza al contexto.
            if vectorial and RE_POLIZA_CONCRETA.search(pregunta):
                try:
                    propios = self._recuperar_vectorial(pregunta, 4, documento=DOC_POLIZA)
                    vistos = {r["texto"] for r in propios}
                    # La tabla general de tarifas sale de este contexto: su precio de catálogo
                    # no es lo que paga ese cliente, y el modelo lo prefería a la prima de la póliza.
                    usados = propios + [r for r in usados
                                        if r["texto"] not in vistos and r["documento"] != DOC_TARIFAS]
                except Exception as e:
                    log.warning("No se pudieron añadir las condiciones particulares: %s", e)
            contexto = "\n\n".join(f"[{i}] {r['texto']}" for i, r in enumerate(usados, 1))

            # 3) Generación con el tiempo que quede del presupuesto
            restante = PRESUPUESTO_SEGUNDOS - (time.perf_counter() - t0)
            salida = llm.generar(PLANTILLA.format(contexto=contexto, pregunta=pregunta), sistema=SISTEMA,
                                 max_tokens=MAX_TOKENS, timeout=max(LLM_TIMEOUT_MINIMO, restante))
            evidencia, respuesta = separar_salida(salida)

            # 4) El modelo reconoce que no está en el contexto (se mira ANTES de limpiar el texto,
            #    que quitaría el guion bajo de NO_CONSTA)
            if (not respuesta.strip() or RE_NO_CONSTA.search(respuesta)
                    or re.fullmatch(r"\W*ninguna\W*", evidencia, re.IGNORECASE)):
                log.info("RAG: el LLM indica NO_CONSTA para %r", pregunta)
                return fin(MENSAJE_NO_CONSTA, False)
            respuesta = limpiar_para_voz(respuesta)

            # 5) Control de anclaje (grounding):
            #    a) la EVIDENCIA citada tiene que estar de verdad en el contexto (no inventada);
            #    b) toda cifra de la respuesta tiene que aparecer en el contexto o en la pregunta.
            if evidencia and cobertura_palabras(evidencia, contexto) < MIN_EVIDENCIA_EN_CONTEXTO:
                log.warning("RAG: respuesta descartada, evidencia que no está en el contexto: %r", evidencia)
                return fin(MENSAJE_NO_CONSTA, False)
            inventadas = numeros_sin_respaldo(respuesta, contexto + "\n" + pregunta)
            if inventadas:
                log.warning("RAG: respuesta descartada, cifras sin respaldo %s: %r", inventadas, respuesta)
                return fin(MENSAJE_NO_CONSTA, False)

            # Fuentes: los documentos de los fragmentos que contienen la evidencia (o todos los usados)
            fuentes = [r["documento"] for r in usados
                       if evidencia and cobertura_palabras(evidencia, r["texto"]) >= MIN_EVIDENCIA_EN_CONTEXTO]
            fuentes = list(dict.fromkeys(fuentes or [r["documento"] for r in usados]))
            return fin(respuesta, True, fuentes)

        except TimeoutError as e:
            log.warning("RAG: tiempo agotado: %s", e)
            return fin(MENSAJE_ERROR, False)
        except Exception as e:  # nunca dejar al usuario en silencio
            log.exception("RAG: error inesperado: %s", e)
            return fin(MENSAJE_ERROR, False)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    motor = MotorDocumental()
    llm.precalentar()
    for p in sys.argv[1:] or ["¿Cuál es la franquicia del todo riesgo?"]:
        r = motor.responder(p)
        print(f"\nP: {p}\nR: {r.texto}\n   respondida={r.respondida} fuentes={r.fuentes} {r.latencia:.2f}s")
