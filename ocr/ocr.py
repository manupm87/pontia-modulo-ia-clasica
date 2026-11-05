"""
OCR offline de los documentos de Seguros Alonso + extracción y validación con expresiones regulares.

Se ejecuta UNA SOLA VEZ (no dentro del webhook): reconocer una página tarda
segundos y Dialogflow corta la llamada al webhook a los 5 s. El resultado se
guarda en data/documentos.json y el webhook solo lee ese fichero.

Pipeline por documento:
  1. Preprocesado con OpenCV (escala de grises, enderezado, eliminación de ruido,
     mejora de contraste y, solo para Tesseract, ampliación + binarización).
  2. Motor OCR:
       - "vlm": modelo de visión qwen2.5vl:7b servido por Ollama en local. Lee la
         página entera y devuelve Markdown (tablas incluidas). Es el motor por
         defecto porque entiende tablas, dos columnas y letra manuscrita.
       - "tesseract": OCR clásico (si está instalado). Rápido, pero sufre con
         la letra manuscrita y pierde la estructura de las tablas.
       - "auto": usa el VLM si Ollama responde; si no, Tesseract.
  3. Regex sobre el texto reconocido: fechas, importes, pólizas, matrículas, DNI,
     teléfonos y números de siniestro, VALIDANDO cada candidato (fecha real,
     letra de control del DNI, etc.).
  4. Métrica de calidad: CER/WER contra el texto de referencia (ground_truth.json).

Uso (desde la raíz del repo):
    python -m ocr.ocr                 # motor VLM (por defecto)
    python -m ocr.ocr --motor auto
    python -m ocr.ocr --motor tesseract
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import requests

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DOCUMENTOS = RAIZ / "documentos"
CARPETA_DATOS = RAIZ / "data"
CARPETA_PREPROCESADO = CARPETA_DATOS / "preprocesado"
RUTA_SALIDA = CARPETA_DATOS / "documentos.json"
RUTA_REFERENCIA = CARPETA_DOCUMENTOS / "ground_truth.json"
EXTENSIONES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# 127.0.0.1 y no "localhost": en Windows "localhost" intenta antes IPv6 (::1) y cada conexión nueva
# pierde ~2 s hasta caer a IPv4. Medido: 2,1 s con localhost frente a 0,07 s con 127.0.0.1.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODELO_VLM = os.getenv("OLLAMA_VLM_MODEL", "qwen2.5vl:7b")


# ===========================================================================
# 1. Preprocesado con OpenCV
# ===========================================================================

def estimar_inclinacion(gris: np.ndarray) -> float:
    """
    Estima cuántos grados está girada la página.

    Método principal: transformada de Hough probabilística sobre los bordes. En
    un documento hay muchas líneas casi horizontales (renglones, rejilla de las
    tablas, subrayados); la mediana de sus ángulos es la inclinación del papel
    y la mediana ignora las pocas líneas que no son horizontales.

    Respaldo: minAreaRect sobre los píxeles de tinta (el rectángulo mínimo que
    envuelve el texto queda girado lo mismo que la página).
    """
    bordes = cv2.Canny(gris, 50, 150, apertureSize=3)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 1800, threshold=150,
                             minLineLength=gris.shape[1] // 6, maxLineGap=15)
    angulos = []
    if lineas is not None:
        for x1, y1, x2, y2 in lineas.reshape(-1, 4):  # (N,1,4) o (N,4) según la versión
            angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angulo) < 10:  # solo líneas casi horizontales
                angulos.append(angulo)
    if len(angulos) >= 5:
        return float(np.median(angulos))

    # Respaldo: minAreaRect sobre la tinta
    tinta = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(tinta > 0))[:, ::-1].astype(np.float32)
    if len(coords) < 100:
        return 0.0
    (_, _), (_, _), angulo = cv2.minAreaRect(coords)
    if angulo > 45:  # OpenCV >= 4.5 devuelve ángulos en [0, 90)
        angulo -= 90
    return float(angulo)


def enderezar(gris: np.ndarray, angulo: float) -> np.ndarray:
    """Gira la imagen para dejar los renglones horizontales (rellenando el borde con blanco)."""
    if abs(angulo) < 0.1:
        return gris
    h, w = gris.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    return cv2.warpAffine(gris, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def preprocesar(ruta: Path, para: str = "vlm") -> tuple[np.ndarray, dict]:
    """
    Prepara la imagen para el motor indicado. Devuelve (imagen, info).

    Pasos y POR QUÉ:
      - Escala de grises: el color no aporta nada para leer texto y triplica
        los datos. (La tinta azul del manuscrito sigue siendo más oscura que el papel.)
      - Enderezado: Tesseract segmenta por renglones horizontales; con 1-2 grados
        de giro los renglones largos se "cruzan" y se mezclan palabras de dos
        líneas. Al VLM también le ayuda a alinear las celdas de las tablas.
      - Eliminación de ruido (Non-Local Means): borra el grano del escáner y
        las motas de polvo, que el OCR confunde con puntos, comas o tildes.
      - CLAHE (ecualización de histograma adaptativa por zonas): el papel tiene
        iluminación irregular; CLAHE sube el contraste localmente sin quemar las
        zonas claras, lo que rescata el bolígrafo azul, más débil que el tóner negro.
      - Solo Tesseract:
          * Ampliación x1.5: Tesseract rinde mejor con letras de ~30 px de alto;
            a 170 ppp la letra del cuerpo mide ~20 px.
          * Umbral adaptativo (binarización): Tesseract está entrenado con texto
            negro sobre blanco; un umbral local resiste las sombras del papel
            mejor que un umbral global.
        Al VLM NO se le binariza: el modelo de visión aprovecha los grises
        (antialiasing, trazo del manuscrito) y la binarización rompe trazos finos.
    """
    color = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"No se puede leer la imagen {ruta}")
    gris = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    angulo = estimar_inclinacion(gris)
    gris = enderezar(gris, angulo)
    gris = cv2.fastNlMeansDenoising(gris, None, h=12, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)

    info = {"inclinacion_corregida": round(angulo, 2)}
    if para == "tesseract":
        gris = cv2.resize(gris, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gris = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize=31, C=15)
    return gris, info


# ===========================================================================
# 2. Motores OCR
# ===========================================================================

PROMPT_VLM = """Eres un sistema de OCR. Transcribe FIELMENTE todo el texto de este documento escaneado, en español.
Reglas:
- Respeta el orden de lectura. Si hay dos columnas, transcribe primero la columna izquierda completa y después la derecha.
- Usa Markdown: '#' para títulos, listas con '-' o '1.', y las TABLAS como tablas Markdown con todas sus filas y columnas.
- En las tablas conserva los símbolos tal cual (✓ y ✗) y los importes con su formato (por ejemplo 1.234,56 €).
- Casillas de formulario: marcada = [X], vacía = [ ].
- En los formularios escribe cada campo como 'Etiqueta: valor', incluido el texto manuscrito.
- No resumas, no traduzcas, no corrijas faltas, no añadas comentarios ni explicaciones.
- Si una palabra es ilegible, escribe [ilegible]."""


def ollama_disponible(modelo: str = MODELO_VLM) -> bool:
    """True si Ollama responde y tiene descargado el modelo."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        nombres = {m["name"] for m in r.json().get("models", [])}
        return modelo in nombres or f"{modelo}:latest" in nombres
    except Exception:
        return False


def ocr_vlm(imagen: np.ndarray) -> str:
    """Transcribe la página con el modelo de visión (temperatura 0 = salida determinista)."""
    ok, png = cv2.imencode(".png", imagen)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen")
    cuerpo = {
        "model": MODELO_VLM,
        "stream": False,
        "keep_alive": "10m",
        "messages": [{
            "role": "user",
            "content": PROMPT_VLM,
            "images": [base64.b64encode(png.tobytes()).decode("ascii")],
        }],
        "options": {"temperature": 0, "num_ctx": 12288, "num_predict": 4096, "seed": 42},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=cuerpo, timeout=600)
    r.raise_for_status()
    texto = r.json()["message"]["content"]
    # Algunos modelos envuelven la respuesta en ```markdown ... ```
    texto = re.sub(r"^```(?:markdown|md)?\s*\n|\n?```\s*$", "", texto.strip())
    return texto.strip()


def localizar_tesseract() -> bool:
    """Autodetección como en el código del curso: PATH o ruta típica de Windows."""
    try:
        import pytesseract
    except ImportError:
        return False
    if shutil.which("tesseract"):
        return True
    ruta_windows = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if ruta_windows.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(ruta_windows)
        return True
    return False


# Modo de segmentación de página (psm) de Tesseract según el tipo de documento:
#   3 = automático (detecta columnas y bloques) -> texto corrido y folleto a 2 columnas
#   4 = una columna de texto de tamaños variables -> formularios
#   6 = un bloque uniforme -> tabla (mantiene cada fila en una línea)
PSM_POR_DOCUMENTO = {"01": 3, "02": 6, "03": 3, "04": 4, "05": 4}


def ocr_tesseract(imagen: np.ndarray, nombre: str) -> str:
    import pytesseract
    psm = PSM_POR_DOCUMENTO.get(nombre[:2], 3)
    return pytesseract.image_to_string(imagen, lang="spa", config=f"--psm {psm}")


# ===========================================================================
# 3. Extracción y validación con expresiones regulares
# ===========================================================================

# Patrones "estrictos" sobre el texto tal cual (idea del curso, ampliada).
PATRONES = {
    # 14/03/2026, 2-11-26, 03.11.2026
    "fecha": r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})\b",
    # 612,40 € / 50.000.000 € / 300 euros / 285,00 EUR
    "importe": r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{1,2}))?\s?(?:€|EUR\b|euros?\b)",
    # Póliza de Seguros Alonso: SA-123456
    "poliza": r"\bSA-\d{6}\b",
    # Matrícula española actual: 4 dígitos + 3 consonantes (sin vocales, Ñ ni Q)
    "matricula": r"\b\d{4}[\s-]?[B-DF-HJ-NP-TV-Z]{3}\b",
    # DNI (8 dígitos + letra) o NIE (X/Y/Z + 7 dígitos + letra)
    "dni": r"\b(?:\d{8}|[XYZ]\d{7})[\s-]?[A-Z]\b",
    # Teléfono español: 9 cifras empezando por 6, 7, 8 o 9, agrupadas 3-3-3, 3-2-2-2 o seguidas
    "telefono": r"(?<!\d)(?:\+34\s?)?[6-9]\d{2}(?:[\s.]?\d{3}[\s.]?\d{3}|[\s.]\d{2}[\s.]\d{2}[\s.]\d{2})(?!\d)",
    # Número de siniestro: SIN-2026-004512
    "siniestro": r"\bSIN-\d{4}-\d{6}\b",
    # Código de cita de peritaje: PER-12345
    "cita": r"\bPER-\d{5}\b",
}

# Patrones "laxos": admiten en las posiciones numéricas los caracteres que el OCR
# confunde con dígitos (O, o, l, I, |, S, B...). Sirven para encontrar CANDIDATOS
# que después se corrigen y se vuelven a validar.
_D = r"[\dOoQDlI|SsBZz]"  # "dígito o algo que el OCR suele confundir con un dígito"
PATRONES_LAXOS = {
    "poliza": rf"\b[S5][A4]\s?-\s?{_D}{{6}}\b",
    "siniestro": rf"\b[S5][I1l]N\s?-\s?{_D}{{4}}\s?-\s?{_D}{{6}}\b",
    "dni": rf"(?<![\w-]){_D}{{8}}[\s-]?[A-Z](?!\w)",
    "matricula": rf"(?<![\w-]){_D}{{4}}[\s-]?[B-DF-HJ-NP-TV-Z0-9]{{3}}(?!\w)",
    "fecha": rf"(?<![\w/]){_D}{{1,2}}/{_D}{{1,2}}/{_D}{{4}}(?![\w/])",
}

# Confusiones típicas del OCR al leer una cifra
A_DIGITO = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "l": "1", "I": "1", "|": "1",
                          "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})
# ... y al leer una letra de la matrícula (que solo admite consonantes)
A_LETRA = str.maketrans({"8": "B", "5": "S", "2": "Z", "0": "D", "6": "G", "1": "J"})

LETRAS_DNI = "TRWAGMYFPDXBNJZSQVHLCKE"


def letra_dni(numero: int) -> str:
    """Letra de control del DNI: resto de dividir entre 23, indexado en LETRAS_DNI."""
    return LETRAS_DNI[numero % 23]


def validar_dni(valor: str) -> tuple[bool, str]:
    """Valida un DNI o NIE comprobando la letra de control. Devuelve (válido, normalizado)."""
    v = re.sub(r"[\s-]", "", valor.upper())
    if not re.fullmatch(r"(\d{8}|[XYZ]\d{7})[A-Z]", v):
        return False, v
    cuerpo = v[:-1].replace("X", "0").replace("Y", "1").replace("Z", "2")  # NIE -> número
    return letra_dni(int(cuerpo)) == v[-1], v


def validar_fecha(dia: str, mes: str, anio: str) -> str | None:
    """Devuelve la fecha en ISO (AAAA-MM-DD) si existe en el calendario; si no, None."""
    a = int(anio)
    if len(anio) == 2:
        a += 2000
    try:
        return date(a, int(mes), int(dia)).isoformat()
    except ValueError:
        return None


def importe_a_float(entero: str, decimales: str | None) -> float:
    """'50.000.000' + None -> 50000000.0 ; '612' + '40' -> 612.4 (formato español)."""
    return float(entero.replace(".", "") + "." + (decimales or "0"))


def normalizar_telefono(valor: str) -> str:
    """Deja solo las 9 cifras nacionales agrupadas 3-3-3: '900 123 456'."""
    cifras = re.sub(r"\D", "", valor)
    if cifras.startswith("34") and len(cifras) == 11:
        cifras = cifras[2:]
    return f"{cifras[:3]} {cifras[3:6]} {cifras[6:]}"


def normalizar_ocr_token(token: str) -> str:
    """
    Corrige confusiones del OCR (O->0, l->1, S->5, B->8...) SOLO dentro de un token
    candidato y solo en las posiciones que deben ser cifras.

    El código del curso aplicaba estas sustituciones a TODO el texto, lo que
    destroza las palabras normales ("SEGUROS" -> "5EGUR05"). Aquí se aplican
    únicamente al trozo que ya sabemos que debería ser numérico.
    """
    return token.translate(A_DIGITO)


def _anadir(lista: list, valor) -> None:
    if valor not in lista:
        lista.append(valor)


def extraer_campos(texto: str) -> tuple[dict, list[str]]:
    """
    Busca y VALIDA los datos estructurados del texto reconocido.

    Devuelve (campos, avisos):
      campos = {"fechas": ["2026-03-14", ...], "importes": [612.4, ...], "polizas": [...],
                "matriculas": ["1234BCD"], "dnis": ["12345678Z"], "telefonos": ["900 123 456"],
                "siniestros": [...], "citas": [...]}
      avisos = candidatos descartados o corregidos, para revisión humana.
    """
    campos = {k: [] for k in ("fechas", "importes", "polizas", "matriculas", "dnis",
                              "telefonos", "siniestros", "citas")}
    avisos: list[str] = []

    # --- Fechas: el patrón casa con 31/02/2026, así que hay que comprobar el calendario
    for m in re.finditer(PATRONES["fecha"], texto):
        iso = validar_fecha(*m.groups())
        if iso:
            _anadir(campos["fechas"], iso)
        else:
            avisos.append(f"Fecha descartada (no existe en el calendario): {m.group(0)}")

    # --- Importes en euros, normalizados a float
    for m in re.finditer(PATRONES["importe"], texto, re.IGNORECASE):
        _anadir(campos["importes"], importe_a_float(m.group(1), m.group(2)))

    # --- Identificadores con formato fijo
    for m in re.finditer(PATRONES["poliza"], texto):
        _anadir(campos["polizas"], m.group(0))
    for m in re.finditer(PATRONES["siniestro"], texto):
        _anadir(campos["siniestros"], m.group(0))
    for m in re.finditer(PATRONES["cita"], texto):
        _anadir(campos["citas"], m.group(0))
    for m in re.finditer(PATRONES["matricula"], texto):
        _anadir(campos["matriculas"], re.sub(r"[\s-]", "", m.group(0)))
    for m in re.finditer(PATRONES["telefono"], texto):
        _anadir(campos["telefonos"], normalizar_telefono(m.group(0)))

    # --- DNI con letra de control
    for m in re.finditer(PATRONES["dni"], texto):
        valido, v = validar_dni(m.group(0))
        if valido:
            _anadir(campos["dnis"], v)
        else:
            esperado = letra_dni(int(v[:-1].replace("X", "0").replace("Y", "1").replace("Z", "2")))
            avisos.append(f"DNI descartado (letra de control incorrecta, debería ser {esperado}): {m.group(0)}")

    # --- Segunda pasada: candidatos con posibles confusiones de OCR
    _corregir_candidatos(texto, campos, avisos)
    return campos, avisos


def _corregir_candidatos(texto: str, campos: dict, avisos: list[str]) -> None:
    """Busca con los patrones laxos, corrige el token y lo acepta solo si pasa la validación."""
    for m in re.finditer(PATRONES_LAXOS["poliza"], texto):
        token = m.group(0)
        corregido = "SA-" + normalizar_ocr_token(re.split(r"-\s?", token, maxsplit=1)[1].strip())
        if re.fullmatch(PATRONES["poliza"], corregido) and corregido not in campos["polizas"]:
            campos["polizas"].append(corregido)
            avisos.append(f"Póliza corregida (confusión OCR): '{token}' -> '{corregido}'")

    for m in re.finditer(PATRONES_LAXOS["siniestro"], texto):
        token = m.group(0)
        partes = re.split(r"\s?-\s?", token)
        corregido = f"SIN-{normalizar_ocr_token(partes[1])}-{normalizar_ocr_token(partes[2])}"
        if re.fullmatch(PATRONES["siniestro"], corregido) and corregido not in campos["siniestros"]:
            campos["siniestros"].append(corregido)
            avisos.append(f"Nº de siniestro corregido (confusión OCR): '{token}' -> '{corregido}'")

    for m in re.finditer(PATRONES_LAXOS["dni"], texto):
        token = m.group(0)
        compacto = re.sub(r"[\s-]", "", token)
        corregido = normalizar_ocr_token(compacto[:-1]) + compacto[-1]
        if corregido == compacto:
            continue  # sin confusiones: ya se trató en la pasada estricta
        valido, v = validar_dni(corregido)
        if valido and v not in campos["dnis"]:
            campos["dnis"].append(v)
            avisos.append(f"DNI corregido (confusión OCR) y validado con la letra: '{token}' -> '{v}'")

    for m in re.finditer(PATRONES_LAXOS["matricula"], texto):
        token = m.group(0)
        compacto = re.sub(r"[\s-]", "", token)
        corregido = normalizar_ocr_token(compacto[:4]) + compacto[4:].translate(A_LETRA)
        if corregido != compacto and re.fullmatch(r"\d{4}[B-DF-HJ-NP-TV-Z]{3}", corregido) \
                and corregido not in campos["matriculas"]:
            campos["matriculas"].append(corregido)
            avisos.append(f"Matrícula corregida (confusión OCR): '{token}' -> '{corregido}'")

    for m in re.finditer(PATRONES_LAXOS["fecha"], texto):
        token = m.group(0)
        corregido = normalizar_ocr_token(token)
        if corregido == token:
            continue
        iso = validar_fecha(*corregido.split("/"))
        if iso and iso not in campos["fechas"]:
            campos["fechas"].append(iso)
            avisos.append(f"Fecha corregida (confusión OCR): '{token}' -> '{corregido}'")


# ===========================================================================
# 4. Métrica de calidad del OCR (CER / WER)
# ===========================================================================

def normalizar_para_metrica(texto: str) -> str:
    """
    Normaliza antes de comparar con la referencia, para medir errores de LECTURA
    y no diferencias de formato: quita la sintaxis Markdown (|, #, **, ---),
    unifica variantes de símbolos (✔ -> ✓), pasa a minúsculas y colapsa espacios.
    """
    t = unicodedata.normalize("NFC", texto)
    t = t.replace("✔", "✓").replace("✘", "✗").replace("❌", "✗").replace("☑", "[x]").replace("☐", "[ ]")
    t = re.sub(r"^\s*\|?\s*:?-{3,}.*$", " ", t, flags=re.MULTILINE)  # separadores de tabla
    t = re.sub(r"[|#*_>`]", " ", t)
    t = re.sub(r"^\s*[-•]\s+", " ", t, flags=re.MULTILINE)  # viñetas
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def distancia_edicion(a, b) -> int:
    """Levenshtein. Usa rapidfuzz si está instalado (C++); si no, programación dinámica."""
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        previa = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            actual = [i]
            for j, cb in enumerate(b, 1):
                actual.append(min(previa[j] + 1, actual[j - 1] + 1, previa[j - 1] + (ca != cb)))
            previa = actual
        return previa[-1]


def calcular_metricas(reconocido: str, referencia: str) -> dict:
    """
    CER = errores de carácter / caracteres de la referencia.
    WER = errores de palabra / palabras de la referencia.
    Ojo: si el OCR lee en otro orden (columnas, celdas), CER/WER lo penalizan
    aunque las palabras sean correctas; por eso se añade también el % de
    palabras de la referencia que aparecen en el texto (independiente del orden).
    """
    r, h = normalizar_para_metrica(referencia), normalizar_para_metrica(reconocido)
    pr, ph = r.split(), h.split()
    cer = distancia_edicion(r, h) / max(1, len(r))
    wer = distancia_edicion(pr, ph) / max(1, len(pr))
    vocab = set(ph)
    cobertura = sum(1 for p in pr if p in vocab) / max(1, len(pr))
    return {"cer": round(cer, 4), "wer": round(wer, 4), "palabras_encontradas": round(cobertura, 4)}


# ===========================================================================
# Orquestación
# ===========================================================================

def elegir_motor(motor: str) -> str:
    if motor == "auto":
        if ollama_disponible():
            return "vlm"
        if localizar_tesseract():
            return "tesseract"
        raise SystemExit("No hay motor OCR disponible: ni Ollama con qwen2.5vl ni Tesseract.")
    if motor == "tesseract" and not localizar_tesseract():
        raise SystemExit("Tesseract no está instalado (o no está en el PATH).")
    if motor == "vlm" and not ollama_disponible():
        raise SystemExit(f"Ollama no responde en {OLLAMA_URL} o falta el modelo {MODELO_VLM}.")
    return motor


def procesar_documento(ruta: Path, motor: str, referencia: str | None) -> dict:
    """Procesa un documento. Nunca lanza: si algo falla, lo deja anotado en 'avisos'."""
    resultado = {"documento": ruta.name, "motor": motor, "texto": "", "caracteres": 0,
                 "segundos": 0.0, "campos": {}, "avisos": []}
    t0 = time.perf_counter()
    try:
        imagen, info = preprocesar(ruta, para=motor)
        CARPETA_PREPROCESADO.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(CARPETA_PREPROCESADO / ruta.name), imagen)
        texto = ocr_vlm(imagen) if motor == "vlm" else ocr_tesseract(imagen, ruta.name)
        resultado.update(texto=texto, caracteres=len(texto), preprocesado=info)
        resultado["campos"], resultado["avisos"] = extraer_campos(texto)
        if referencia:
            resultado["metricas"] = calcular_metricas(texto, referencia)
    except Exception as e:  # un documento roto no debe tumbar el resto
        resultado["avisos"].append(f"ERROR de OCR: {type(e).__name__}: {e}")
    resultado["segundos"] = round(time.perf_counter() - t0, 2)
    return resultado


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OCR offline de los documentos de Seguros Alonso")
    parser.add_argument("--motor", choices=["vlm", "tesseract", "auto"], default="vlm")
    parser.add_argument("--carpeta", type=Path, default=CARPETA_DOCUMENTOS)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    args = parser.parse_args(argv)

    motor = elegir_motor(args.motor)
    referencias = {}
    if RUTA_REFERENCIA.is_file():
        referencias = json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8"))

    rutas = sorted(p for p in args.carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {args.carpeta}. Ejecuta antes: python -m ocr.generar_documentos")

    print(f"Motor OCR: {motor} ({MODELO_VLM if motor == 'vlm' else 'tesseract spa'})")
    resultados = []
    for ruta in rutas:
        print(f"\n== {ruta.name}")
        res = procesar_documento(ruta, motor, referencias.get(ruta.name))
        resultados.append(res)
        met = res.get("metricas")
        calidad = (f"CER {met['cer']:.1%} · WER {met['wer']:.1%} · palabras encontradas "
                   f"{met['palabras_encontradas']:.1%}") if met else "sin referencia"
        print(f"   {res['caracteres']} caracteres en {res['segundos']} s · {calidad}")
        for campo, valores in res["campos"].items():
            if valores:
                print(f"   {campo}: {valores}")
        for aviso in res["avisos"]:
            print(f"   AVISO: {aviso}")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    con_metricas = [r["metricas"] for r in resultados if r.get("metricas")]
    if con_metricas:
        print(f"\nCER medio: {np.mean([m['cer'] for m in con_metricas]):.1%} · "
              f"WER medio: {np.mean([m['wer'] for m in con_metricas]):.1%}")
    print(f"Guardado en {args.salida}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
