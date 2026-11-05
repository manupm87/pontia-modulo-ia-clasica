"""
Indexado de los textos OCR en una base de datos vectorial (ChromaDB). Se ejecuta una vez, tras ocr.py.

Por qué una base vectorial en vez de meter todos los documentos en el prompt:
  - El prompt es más corto -> el LLM responde antes (tenemos < 5 s en total).
  - Se buscan fragmentos por SIGNIFICADO (embeddings), no por coincidencia
    literal: "coche" encuentra "vehículo", "grúa" encuentra "remolque".
  - Al LLM solo le llegan los fragmentos relevantes, lo que reduce que mezcle
    datos de otros apartados (otra fuente típica de alucinaciones).

Troceado ("chunking"):
  - Se respetan los títulos (Markdown '#', "Artículo N.") y los párrafos: cada
    fragmento pertenece a una sola sección.
  - Las secciones cortas se agrupan y las largas se parten en ventanas de
    ~400-700 caracteres con solape (la última frase se repite al inicio del
    siguiente fragmento para no cortar una idea por la mitad).
  - Las tablas se trocean POR FILAS y cada fila lleva la cabecera de la tabla,
    para que el fragmento se entienda solo ("Lunas | Terceros básico: ✗ ...").
  - Cada fragmento empieza con el título del documento del que sale.

Embeddings: modelo bge-m3 (multilingüe, buen rendimiento en español) servido por Ollama.
Almacén: ChromaDB persistente en data/chroma, distancia coseno.

Uso:
    python -m ocr.indexar
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DOCUMENTOS = RAIZ / "data" / "documentos.json"
RUTA_CHROMA = RAIZ / "data" / "chroma"
RUTA_FRAGMENTOS = RAIZ / "data" / "fragmentos.json"   # copia legible de lo indexado

# 127.0.0.1 y no "localhost": en Windows "localhost" intenta antes IPv6 (::1) y cada conexión nueva
# pierde ~2 s hasta caer a IPv4. Medido: 2,1 s con localhost frente a 0,07 s con 127.0.0.1.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODELO_EMBEDDINGS = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
COLECCION = "seguros_alonso"

TAM_MIN, TAM_MAX = 400, 700   # caracteres por fragmento

# Títulos legibles de cada documento (por el prefijo numérico del fichero)
TITULOS = {
    "01": "Condiciones generales del seguro de automóvil",
    "02": "Tabla de coberturas y precios del seguro de automóvil",
    "03": "Guía rápida de peritaje y asistencia en carretera",
    "04": "Condiciones particulares de la póliza SA-104233 (tomadora Ana Ruiz Martín)",
    "05": "Declaración de siniestro SIN-2026-004512 (parte manuscrito de Ana Ruiz Martín)",
}

RE_TITULO = re.compile(r"^\s*(#{1,6}\s+.+|\**\s*Art[ií]culo\s+\d+.*)$", re.IGNORECASE)


def es_titulo(linea: str) -> bool:
    """Título Markdown, "Artículo N." o línea corta toda en MAYÚSCULAS ("B. DATOS DEL SINIESTRO")."""
    t = linea.strip()
    if RE_TITULO.match(t):
        return True
    letras = [c for c in t if c.isalpha()]
    return 4 <= len(t) <= 90 and len(letras) >= 4 and not any(c.islower() for c in letras)


def es_tabla_de_datos(filas: list[list[str]]) -> bool:
    """
    Distingue una tabla de DATOS (cabecera + filas comparables, como la de coberturas)
    de una tabla de MAQUETACIÓN (el VLM a veces transcribe un formulario como tabla
    de pares "Etiqueta: valor"). Solo las primeras se trocean por filas.
    """
    cabecera, cuerpo = filas[0], [c for f in filas[1:] for c in f if c]
    if any(":" in c for c in cabecera):
        return False
    if len(cabecera) >= 3:
        return True
    con_dos_puntos = sum(1 for c in cuerpo if ":" in c) / max(1, len(cuerpo))
    return con_dos_puntos < 0.2 and all(len(c) < 40 for c in cabecera)


def titulo_documento(nombre: str) -> str:
    return TITULOS.get(nombre[:2], Path(nombre).stem.replace("_", " "))


def _celdas(linea: str) -> list[str]:
    return [c.strip() for c in linea.strip().strip("|").split("|")]


def _describir_celda(valor: str) -> str:
    """Hace explícitos los símbolos para el embedding y para el LLM."""
    v = valor.replace("✔", "✓").replace("✘", "✗")
    if v == "✓":
        return "✓ (incluida)"
    if v == "✗":
        return "✗ (no incluida)"
    return v


def trocear_tabla(lineas: list[str], prefijo: str) -> list[dict]:
    """
    Un fragmento por fila, con la cabecera "pegada" a cada celda.

    No se indexa la tabla completa en Markdown: se probó y el LLM de 7B leía mal
    la rejilla de ✓/✗ (atribuía una celda a la columna equivocada). Con una fila
    por fragmento y cada celda etiquetada ("Terceros básico: ✗ (no incluida)")
    no hay ambigüedad posible.
    """
    filas = [l for l in lineas if not re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", l)]
    if len(filas) < 2:
        return [{"texto": prefijo + "\n" + "\n".join(lineas), "tipo": "tabla"}]
    cabecera = _celdas(filas[0])
    if not es_tabla_de_datos([_celdas(f) for f in filas]):
        # Formulario maquetado como tabla: se convierte en líneas de texto normales
        texto = "\n".join(" · ".join(c for c in _celdas(f) if c) for f in filas)
        return [{"texto": f"{prefijo}\n{texto}", "tipo": "formulario"}]
    fragmentos = []
    for fila in filas[1:]:
        celdas = _celdas(fila)
        pares = [f"{c}: {_describir_celda(v)}" for c, v in zip(cabecera[1:], celdas[1:])]
        texto = (f"{prefijo}\nFila de la tabla «{celdas[0]}» ({cabecera[0]}: {celdas[0]}). "
                 + "; ".join(pares) + ".")
        fragmentos.append({"texto": texto, "tipo": "fila_tabla"})
    return fragmentos


def _partir_largo(texto: str) -> list[str]:
    """Parte un texto largo en ventanas de TAM_MIN-TAM_MAX caracteres con una frase de solape."""
    frases = re.split(r"(?<=[.;:])\s+|\n+", texto)
    trozos, actual = [], ""
    for frase in frases:
        if actual and len(actual) + len(frase) + 1 > TAM_MAX and len(actual) >= TAM_MIN // 2:
            trozos.append(actual.strip())
            solape = re.split(r"(?<=[.;])\s+", actual.strip())[-1]
            actual = (solape + " " if len(solape) < 200 else "") + frase
        else:
            actual = (actual + " " + frase).strip()
    if actual.strip():
        trozos.append(actual.strip())
    return trozos


def trocear(documento: str, texto: str) -> list[dict]:
    """Devuelve [{"documento", "texto", "tipo", "seccion"}] en orden de aparición."""
    prefijo = f"[{titulo_documento(documento)}]"
    # "Nº"/"N°" -> "Número": el VLM a veces usa el símbolo de grado y el LLM no lo reconoce
    texto = re.sub(r"\bN\s?[º°]\s?", "Número ", texto)
    lineas = texto.splitlines()

    # 1) Separar en bloques: secciones de texto (con su título) y tablas
    bloques: list[tuple[str, str, list[str]]] = []   # (tipo, titulo_seccion, lineas)
    seccion, actual = "", []
    i = 0
    while i < len(lineas):
        linea = lineas[i]
        if linea.strip().startswith("|"):
            if actual:
                bloques.append(("texto", seccion, actual))
                actual = []
            tabla = []
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                tabla.append(lineas[i])
                i += 1
            bloques.append(("tabla", seccion, tabla))
            continue
        if es_titulo(linea):
            if actual:
                bloques.append(("texto", seccion, actual))
            seccion = re.sub(r"^[#*\s]+|\*+$", "", linea).strip()
            actual = [linea.strip()]
        elif linea.strip():
            actual.append(linea.strip())
        i += 1
    if actual:
        bloques.append(("texto", seccion, actual))

    # 2) Convertir bloques en fragmentos de tamaño adecuado
    fragmentos: list[dict] = []
    pendiente, seccion_pendiente = "", ""

    def emitir(t, sec, tipo="texto"):
        for trozo in _partir_largo(t) if len(t) > TAM_MAX else [t]:
            fragmentos.append({"texto": f"{prefijo}\n{trozo}", "tipo": tipo, "seccion": sec})

    for tipo, sec, contenido in bloques:
        if tipo == "tabla":
            # Un título suelto antes de la tabla no se emite solo: ya va en el contexto de la tabla
            if pendiente and not ("\n" not in pendiente and es_titulo(pendiente)):
                emitir(pendiente, seccion_pendiente)
            pendiente = ""
            contexto_tabla = f"{prefijo} {sec}".strip()
            for f in trocear_tabla(contenido, contexto_tabla):
                fragmentos.append({**f, "seccion": sec})
            continue
        bloque = "\n".join(contenido)
        # Agrupar secciones cortas hasta llegar a TAM_MIN, sin pasar de TAM_MAX
        if pendiente and len(pendiente) + len(bloque) + 1 > TAM_MAX:
            emitir(pendiente, seccion_pendiente)
            pendiente = ""
        pendiente = (pendiente + "\n" + bloque).strip()
        seccion_pendiente = seccion_pendiente if pendiente != bloque else sec
        if len(pendiente) >= TAM_MIN:
            emitir(pendiente, seccion_pendiente)
            pendiente = ""
    if pendiente:
        emitir(pendiente, seccion_pendiente)

    for f in fragmentos:
        f["documento"] = documento
    return fragmentos


NOMBRES_CAMPOS = {
    "siniestros": "Número de siniestro", "polizas": "Número de póliza", "matriculas": "Matrícula",
    "dnis": "DNI (letra de control verificada)", "telefonos": "Teléfonos", "fechas": "Fechas",
    "importes": "Importes en euros", "citas": "Códigos de cita",
}


def ficha_campos(doc: dict) -> dict | None:
    """
    Fragmento extra con los datos estructurados que la fase de regex extrajo y VALIDÓ
    (ocr.py -> campos). Así el RAG también encuentra "el número de siniestro" o "el DNI"
    aunque en el texto OCR estén en un formulario difícil de leer para el LLM.
    """
    campos = doc.get("campos") or {}
    partes = []
    for clave, nombre in NOMBRES_CAMPOS.items():
        valores = campos.get(clave) or []
        if clave == "importes":
            valores = [f"{v:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".") for v in valores]
        elif clave == "fechas":
            valores = ["/".join(reversed(v.split("-"))) for v in valores]   # ISO -> dd/mm/aaaa
        if valores:
            partes.append(f"{nombre}: {', '.join(map(str, valores))}")
    if not partes:
        return None
    texto = f"[{titulo_documento(doc['documento'])}]\nDatos validados del documento. " + ". ".join(partes) + "."
    return {"texto": texto, "tipo": "campos_validados", "seccion": "Datos validados", "documento": doc["documento"]}


def fragmentos_documento(doc: dict) -> list[dict]:
    """Fragmentos de texto/tablas del documento + la ficha de campos validados por regex."""
    trozos = trocear(doc["documento"], doc["texto"])
    ficha = ficha_campos(doc)
    return trozos + ([ficha] if ficha else [])


def embeber(textos: list[str], timeout: float = 120) -> list[list[float]]:
    """Calcula los embeddings con Ollama (/api/embed admite una lista de textos)."""
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": MODELO_EMBEDDINGS, "input": textos, "keep_alive": "30m"},
                      timeout=timeout)
    r.raise_for_status()
    return r.json()["embeddings"]


def crear_coleccion(cliente, nombre: str = COLECCION):
    """Crea (o recrea) la colección con distancia coseno."""
    try:
        cliente.delete_collection(nombre)
    except Exception:
        pass
    return cliente.create_collection(name=nombre, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def main() -> None:
    import chromadb

    if not RUTA_DOCUMENTOS.is_file():
        raise SystemExit(f"No existe {RUTA_DOCUMENTOS}. Ejecuta antes: python -m ocr.ocr")
    documentos = json.loads(RUTA_DOCUMENTOS.read_text(encoding="utf-8"))

    fragmentos = []
    for doc in documentos:
        if not doc.get("texto"):
            print(f"  (se omite {doc['documento']}: sin texto)")
            continue
        trozos = fragmentos_documento(doc)
        print(f"  {doc['documento']}: {len(trozos)} fragmentos")
        fragmentos.extend(trozos)

    t0 = time.perf_counter()
    vectores = embeber([f["texto"] for f in fragmentos])
    print(f"Embeddings ({MODELO_EMBEDDINGS}, dim {len(vectores[0])}): {len(vectores)} en "
          f"{time.perf_counter() - t0:.1f} s")

    from chromadb.config import Settings
    # Mismos ajustes que webhook/rag.py: Chroma no admite dos clientes con ajustes distintos
    # sobre la misma carpeta en un proceso (el webhook construye el índice si falta).
    cliente = chromadb.PersistentClient(path=str(RUTA_CHROMA), settings=Settings(anonymized_telemetry=False))
    coleccion = crear_coleccion(cliente)
    coleccion.add(
        ids=[f"frag-{i:03d}" for i in range(len(fragmentos))],
        documents=[f["texto"] for f in fragmentos],
        embeddings=vectores,
        metadatas=[{"documento": f["documento"], "tipo": f["tipo"], "seccion": f.get("seccion", "")}
                   for f in fragmentos],
    )
    RUTA_FRAGMENTOS.write_text(json.dumps(fragmentos, ensure_ascii=False, indent=2), encoding="utf-8")
    tamanos = [len(f["texto"]) for f in fragmentos]
    print(f"Colección '{COLECCION}' en {RUTA_CHROMA}: {coleccion.count()} fragmentos "
          f"(tamaño medio {sum(tamanos) // len(tamanos)} caracteres, máx. {max(tamanos)})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
