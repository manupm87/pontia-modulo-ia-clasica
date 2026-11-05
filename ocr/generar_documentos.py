"""
Genera los 5 documentos "escaneados" de la aseguradora ficticia Seguros Alonso, S.A.

Por qué generamos los documentos en vez de fotografiar papeles reales:
  - Los datos son inventados (no hay datos personales reales ni marcas reales).
  - Conocemos el texto exacto de cada documento (la "verdad de referencia" o
    ground truth), así que podemos MEDIR la calidad del OCR (CER/WER) en vez de
    fiarnos de la vista.
  - Podemos fabricar a propósito los casos difíciles que pide el enunciado como
    extra: tablas con rejilla, maquetación a dos columnas, formularios con
    casillas y texto manuscrito.

Cada documento se dibuja con PIL a ~170 ppp (tamaño A4) y después se "escanea":
rotación ligera, fondo de papel grisáceo e irregular, desenfoque, ruido
gaussiano, motas de polvo y artefactos JPEG.

Salida:
  documentos/01_condiciones_generales_auto.png
  documentos/02_tabla_coberturas.png
  documentos/03_guia_peritaje_y_asistencia.png
  documentos/04_condiciones_particulares_SA-104233.png
  documentos/05_parte_siniestro_manuscrito.png
  documentos/ground_truth.json   {nombre_png: texto_de_referencia}

Uso (desde la raíz del repo):
    python -m ocr.generar_documentos
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_SALIDA = RAIZ / "documentos"

# A4 a ~170 ppp: 8,27 x 11,69 pulgadas -> 1406 x 1988 píxeles
ANCHO, ALTO = 1406, 1988
MARGEN = 80

FUENTES_WINDOWS = Path("C:/Windows/Fonts")
TINTA_NEGRA = (25, 25, 28)
TINTA_AZUL = (28, 48, 150)       # bolígrafo azul
TINTA_GRIS = (90, 90, 95)
COLOR_MARCA = (20, 70, 110)      # azul petróleo corporativo (inventado)


# ---------------------------------------------------------------------------
# Fuentes
# ---------------------------------------------------------------------------

_cache_fuentes: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def fuente(nombre: str, tamano: int) -> ImageFont.FreeTypeFont:
    """Carga una fuente TrueType de Windows (con caché). Si no existe, usa la de PIL."""
    clave = (nombre, tamano)
    if clave not in _cache_fuentes:
        ruta = FUENTES_WINDOWS / nombre
        try:
            _cache_fuentes[clave] = ImageFont.truetype(str(ruta), tamano)
        except OSError:
            _cache_fuentes[clave] = ImageFont.load_default(tamano)
    return _cache_fuentes[clave]


def normal(t):
    return fuente("arial.ttf", t)


def negrita(t):
    return fuente("arialbd.ttf", t)


def cursiva(t):
    return fuente("ariali.ttf", t)


def serif(t):
    return fuente("times.ttf", t)


def serif_negrita(t):
    return fuente("timesbd.ttf", t)


def maquina(t):
    """Fuente de 'impresora matricial' para los datos que rellena el sistema."""
    return fuente("cour.ttf", t)


def simbolos(t):
    """Segoe UI Symbol trae los glifos ✓ y ✗."""
    return fuente("seguisym.ttf", t)


# Fuentes de aspecto manuscrito que trae Windows
MANUSCRITAS = ["Inkfree.ttf", "segoepr.ttf", "segoesc.ttf"]


# ---------------------------------------------------------------------------
# Lienzo con "verdad de referencia"
# ---------------------------------------------------------------------------

class Documento:
    """Lienzo PIL que, además de dibujar, acumula el texto de referencia en orden de lectura."""

    def __init__(self, nombre: str, semilla: int, alto: int = ALTO):
        self.nombre = nombre
        self.img = Image.new("RGB", (ANCHO, alto), "white")
        self.d = ImageDraw.Draw(self.img)
        self.gt: list[str] = []
        self.rng = random.Random(semilla)

    # -- texto impreso -------------------------------------------------------

    def texto(self, xy, cadena, f, color=TINTA_NEGRA, gt=True, anchor="la"):
        self.d.text(xy, cadena, font=f, fill=color, anchor=anchor)
        if gt:
            self.gt.append(cadena)

    def parrafo(self, x, y, ancho, cadena, f, interlineado=1.32, color=TINTA_NEGRA,
                sangria_colgante=0, gt=True) -> int:
        """Escribe un párrafo ajustado a `ancho` píxeles. Devuelve la nueva coordenada y."""
        lineas = partir_lineas(cadena, f, ancho, sangria_colgante)
        alto_linea = int(f.size * interlineado)
        for i, linea in enumerate(lineas):
            dx = 0 if i == 0 else sangria_colgante
            self.d.text((x + dx, y), linea, font=f, fill=color)
            y += alto_linea
        if gt:
            self.gt.append(cadena)
        return y

    # -- manuscrito ----------------------------------------------------------

    def manuscrito(self, x, y, cadena, tamano=30, fuente_mano="Inkfree.ttf",
                   color=TINTA_AZUL, temblor=3, gt_extra=None) -> int:
        """
        Dibuja texto "a mano": cada palabra con su propia inclinación, línea base
        desplazada al azar e intensidad de tinta variable. Devuelve la x final.
        """
        f = fuente(fuente_mano, tamano)
        cursor = x
        for palabra in cadena.split(" "):
            if not palabra:
                cursor += int(tamano * 0.35)
                continue
            ancho = int(f.getlength(palabra)) + 12
            alto = int(tamano * 1.9)
            capa = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
            dc = ImageDraw.Draw(capa)
            opacidad = self.rng.randint(190, 255)
            dc.text((4, int(tamano * 0.35)), palabra, font=f, fill=(*color, opacidad))
            capa = capa.rotate(self.rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True)
            dy = self.rng.randint(-temblor, temblor)
            self.img.paste(capa, (cursor, y - int(tamano * 0.35) + dy), capa)
            cursor += int(f.getlength(palabra)) + int(tamano * self.rng.uniform(0.28, 0.42))
        if gt_extra is not False:
            self.gt.append(cadena if gt_extra is None else gt_extra)
        return cursor

    # -- elementos de formulario ---------------------------------------------

    def casilla(self, x, y, marcada: bool, lado=22):
        self.d.rectangle([x, y, x + lado, y + lado], outline=TINTA_NEGRA, width=2)
        if marcada:
            # Aspa a mano: dos trazos ligeramente irregulares en tinta azul
            r = self.rng
            for (x0, y0, x1, y1) in [(x - 3, y - 2, x + lado + 4, y + lado + 3),
                                     (x + lado + 3, y - 4, x - 2, y + lado + 2)]:
                puntos = [(x0 + (x1 - x0) * t + r.uniform(-1.5, 1.5),
                           y0 + (y1 - y0) * t + r.uniform(-1.5, 1.5)) for t in np.linspace(0, 1, 6)]
                self.d.line(puntos, fill=TINTA_AZUL, width=4, joint="curve")

    def opciones(self, x, y, opciones: list[tuple[str, bool]], f, separacion=40, gt_prefijo=""):
        """Fila de casillas con su etiqueta. En la referencia: '[X] Opción' / '[ ] Opción'."""
        partes = []
        for etiqueta, marcada in opciones:
            self.casilla(x, y + 1, marcada)
            self.d.text((x + 32, y), etiqueta, font=f, fill=TINTA_NEGRA)
            x += 32 + int(f.getlength(etiqueta)) + separacion
            partes.append(("[X] " if marcada else "[ ] ") + etiqueta)
        self.gt.append((gt_prefijo + " " if gt_prefijo else "") + " ".join(partes))

    def campo(self, x, y, ancho, etiqueta, valor, f_etiqueta, modo="mano", tamano_valor=30,
              fuente_mano="Inkfree.ttf", color=TINTA_AZUL):
        """Etiqueta impresa + línea punteada + valor manuscrito o mecanografiado."""
        self.d.text((x, y), etiqueta, font=f_etiqueta, fill=TINTA_NEGRA)
        x_valor = x + int(f_etiqueta.getlength(etiqueta)) + 10
        # línea de puntos donde se escribe
        base = y + f_etiqueta.size + 2
        for px in range(x_valor, x + ancho, 7):
            self.d.point((px, base), fill=TINTA_GRIS)
            self.d.point((px + 1, base), fill=TINTA_GRIS)
        if modo == "mano":
            self.manuscrito(x_valor + 6, y - int(tamano_valor * 0.30), valor, tamano=tamano_valor,
                            fuente_mano=fuente_mano, color=color, gt_extra=False)
        else:
            self.d.text((x_valor + 6, y - 1), valor, font=maquina(f_etiqueta.size + 2), fill=TINTA_NEGRA)
        self.gt.append(f"{etiqueta} {valor}")

    def firma(self, x, y, ancho=230, alto=60, color=TINTA_AZUL):
        """Garabato de firma con curvas aleatorias (no contiene texto: no va a la referencia)."""
        r = self.rng
        puntos = []
        for i in range(60):
            t = i / 59
            px = x + t * ancho + 18 * math.sin(t * 19 + r.uniform(0, 0.4))
            py = y + alto / 2 + (alto / 2.3) * math.sin(t * 11 + r.uniform(0, 0.5)) * (1 - 0.5 * t)
            puntos.append((px, py))
        self.d.line(puntos, fill=color, width=3, joint="curve")
        self.d.line([(x + 10, y + alto - 4), (x + ancho * 0.9, y + alto - 12)], fill=color, width=2)

    def sello(self, cx, cy, lineas: list[str], color=(170, 40, 45), angulo=-8):
        """Sello de tinta (rectángulo redondeado con texto), girado y algo transparente."""
        f = negrita(22)
        ancho = max(int(f.getlength(l)) for l in lineas) + 50
        alto = 34 * len(lineas) + 30
        capa = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
        dc = ImageDraw.Draw(capa)
        dc.rounded_rectangle([3, 3, ancho - 4, alto - 4], radius=14, outline=(*color, 200), width=4)
        for i, l in enumerate(lineas):
            dc.text((ancho // 2, 20 + 34 * i), l, font=f, fill=(*color, 185), anchor="ma")
        capa = capa.rotate(angulo, expand=True, resample=Image.BICUBIC)
        self.img.paste(capa, (cx - capa.width // 2, cy - capa.height // 2), capa)
        self.gt.extend(lineas)

    def cabecera(self, titulo: str, subtitulo: str | None = None) -> int:
        """Logotipo inventado (escudo con 'SA') + razón social + título del documento."""
        x, y = MARGEN, 55
        escudo = [(x, y), (x + 70, y), (x + 70, y + 50), (x + 35, y + 82), (x, y + 50)]
        self.d.polygon(escudo, fill=COLOR_MARCA)
        self.d.text((x + 35, y + 36), "SA", font=negrita(30), fill="white", anchor="mm")
        self.texto((x + 92, y + 4), "SEGUROS ALONSO, S.A.", negrita(36), color=COLOR_MARCA)
        self.texto((x + 92, y + 48), "Seguros de automóvil · C/ Gran Vía 41, 28013 Madrid · CIF A-28999999",
                   normal(17), color=TINTA_GRIS)
        self.d.line([(MARGEN, y + 100), (ANCHO - MARGEN, y + 100)], fill=COLOR_MARCA, width=3)
        y += 122
        self.texto((ANCHO // 2, y), titulo, negrita(30), anchor="ma")
        y += 44
        if subtitulo:
            self.texto((ANCHO // 2, y), subtitulo, normal(19), color=TINTA_GRIS, anchor="ma")
            y += 36
        return y + 10

    def pie(self, cadena: str, alto: int | None = None):
        alto = alto or self.img.height
        self.d.line([(MARGEN, alto - 70), (ANCHO - MARGEN, alto - 70)], fill=TINTA_GRIS, width=1)
        self.texto((ANCHO // 2, alto - 58), cadena, normal(15), color=TINTA_GRIS, anchor="ma")

    def referencia(self) -> str:
        return "\n".join(self.gt)


def partir_lineas(cadena: str, f, ancho: int, sangria_colgante: int = 0) -> list[str]:
    """Ajuste de línea por anchura en píxeles (no por número de caracteres)."""
    palabras = cadena.split()
    lineas, actual = [], ""
    for p in palabras:
        prueba = (actual + " " + p).strip()
        limite = ancho - (sangria_colgante if lineas else 0)
        if f.getlength(prueba) <= limite or not actual:
            actual = prueba
        else:
            lineas.append(actual)
            actual = p
    if actual:
        lineas.append(actual)
    return lineas


# ---------------------------------------------------------------------------
# (a) Condiciones generales
# ---------------------------------------------------------------------------

ARTICULOS = [
    ("Artículo 1. Objeto del seguro.",
     "Seguros Alonso, S.A. (en adelante, el Asegurador) garantiza al Tomador y al Asegurado la cobertura de "
     "los riesgos derivados de la propiedad y circulación del vehículo descrito en las Condiciones Particulares, "
     "dentro de los límites y con las modalidades contratadas. Estas Condiciones Generales se aplican a todas "
     "las pólizas de automóvil (turismos de uso particular) emitidas a partir del 1 de enero de 2026."),
    ("Artículo 2. Modalidades.",
     "El Tomador puede elegir entre cuatro modalidades: Terceros básico, Terceros ampliado, Todo riesgo con "
     "franquicia y Todo riesgo sin franquicia. Las garantías incluidas en cada una y sus precios orientativos "
     "figuran en la Tabla de coberturas, que forma parte inseparable de este contrato."),
    ("Artículo 3. Exclusiones.",
     "No quedan cubiertos los siniestros: a) ocurridos cuando el conductor circule bajo la influencia de bebidas "
     "alcohólicas por encima de la tasa legal, o de drogas o estupefacientes; b) producidos durante la "
     "participación en carreras, apuestas, competiciones o pruebas de velocidad; c) causados por un conductor "
     "sin permiso de conducir válido; d) causados intencionadamente por el Asegurado; e) derivados de guerra, "
     "terrorismo o fenómenos de la naturaleza extraordinarios, que cubre el Consorcio de Compensación de "
     "Seguros; f) ocurridos durante el transporte de mercancías peligrosas o el uso del vehículo para alquiler "
     "o VTC no declarado en la póliza."),
    ("Artículo 4. Franquicias.",
     "En la modalidad Todo riesgo con franquicia, el Asegurado asume los primeros 300 € de cada siniestro de "
     "daños propios. La franquicia no se aplica a las garantías de lunas, robo, incendio ni responsabilidad "
     "civil. En la modalidad Todo riesgo sin franquicia el Asegurador paga el importe íntegro de la reparación."),
    ("Artículo 5. Comunicación del siniestro.",
     "El Tomador o el Asegurado deben comunicar el siniestro en un plazo máximo de 7 días desde que lo "
     "conocieron, llamando a Atención al Cliente (910 555 222), a través del asistente de voz o en la web. "
     "Deberán aportar el parte amistoso de accidente si lo hubiera y toda la información sobre las "
     "circunstancias y consecuencias del siniestro."),
    ("Artículo 6. Peritaje.",
     "Para valorar los daños propios, un perito designado por el Asegurador revisará el vehículo en un taller "
     "concertado. El perito acude en un plazo máximo de 72 horas hábiles desde la solicitud de la cita y el "
     "peritaje no tiene ningún coste para el Asegurado. Si el Asegurado no está conforme con la valoración, "
     "puede solicitar un peritaje contradictorio, cuyos honorarios correrán de su cuenta."),
    ("Artículo 7. Vehículo de sustitución.",
     "En las modalidades de Todo riesgo, mientras el vehículo asegurado permanezca en el taller por un "
     "siniestro cubierto, el Asegurado tendrá derecho a un vehículo de sustitución de categoría similar "
     "durante un máximo de 15 días. En Terceros ampliado, solo en caso de robo y durante un máximo de 5 días."),
    ("Artículo 8. Asistencia en carretera.",
     "El servicio de asistencia funciona 24 horas los 365 días del año en el teléfono 900 123 456. Se presta "
     "desde el kilómetro 0 en Terceros ampliado y en Todo riesgo, y a partir de 25 km del domicilio habitual "
     "en Terceros básico. Los límites de remolque de la grúa se indican en la Guía de peritaje y asistencia."),
    ("Artículo 9. Pago de la prima.",
     "La prima puede pagarse de forma anual, semestral (recargo del 2 %) o trimestral (recargo del 3 %), "
     "mediante domiciliación bancaria. El impago de la primera prima permite al Asegurador resolver el contrato."),
    ("Artículo 10. Duración e indemnización.",
     "El contrato dura un año y se prorroga tácitamente por periodos anuales, salvo oposición del Tomador con "
     "un mes de antelación al vencimiento. El Asegurador abonará la indemnización en un plazo máximo de 40 "
     "días desde la recepción del informe pericial."),
]


def documento_condiciones_generales() -> Documento:
    doc = Documento("01_condiciones_generales_auto.png", semilla=101)
    y = doc.cabecera("CONDICIONES GENERALES DEL SEGURO DE AUTOMÓVIL",
                     "Edición 01/2026 · Referencia CG-AUTO-2026")
    ancho_texto = ANCHO - 2 * MARGEN
    f_titulo, f_cuerpo = serif_negrita(23), serif(22)
    for titulo, cuerpo in ARTICULOS:
        doc.texto((MARGEN, y), titulo, f_titulo)
        y += 30
        y = doc.parrafo(MARGEN, y, ancho_texto, cuerpo, f_cuerpo, interlineado=1.28) + 12
    doc.pie("Seguros Alonso, S.A. · Inscrita en el Registro Mercantil de Madrid · CG-AUTO-2026 · Página 1 de 1")
    return doc


# ---------------------------------------------------------------------------
# (b) Tabla de coberturas
# ---------------------------------------------------------------------------

MODALIDADES = ["Terceros básico", "Terceros ampliado", "Todo riesgo con franquicia", "Todo riesgo sin franquicia"]
SI, NO = "✓", "✗"
FILAS_COBERTURAS = [
    ("Responsabilidad civil obligatoria", [SI, SI, SI, SI]),
    ("Responsabilidad civil voluntaria (hasta 50.000.000 €)", [SI, SI, SI, SI]),
    ("Defensa jurídica", [SI, SI, SI, SI]),
    ("Asistencia en carretera", ["Desde km 25", "Desde km 0", "Desde km 0", "Desde km 0"]),
    ("Lunas", [NO, SI, SI, SI]),
    ("Robo", [NO, SI, SI, SI]),
    ("Incendio", [NO, SI, SI, SI]),
    ("Daños propios", [NO, NO, "✓ (franquicia 300 €)", SI]),
    ("Granizo", [NO, NO, SI, SI]),
    ("Vehículo de sustitución", [NO, "5 días (solo robo)", "Hasta 15 días", "Hasta 15 días"]),
    ("Precio anual orientativo", ["285,00 €", "398,50 €", "589,00 €", "845,90 €"]),
]
RECARGOS = [("Anual", "Sin recargo"), ("Semestral", "Recargo del 2 %"), ("Trimestral", "Recargo del 3 %")]


def dibujar_tabla(doc: Documento, x, y, anchos, cabecera, filas, f, f_cab, f_sim, sombreado_cabecera=True) -> int:
    """Tabla con rejilla completa. Las celdas se ajustan en varias líneas si no caben."""
    def lineas_celda(texto, fuente_celda, ancho):
        return partir_lineas(texto, fuente_celda, ancho - 20)

    def alto_fila(celdas, fuentes):
        return max(len(lineas_celda(c, fu, a)) for c, fu, a in zip(celdas, fuentes, anchos)) * int(f.size * 1.3) + 22

    todas = [cabecera] + filas
    total_ancho = sum(anchos)
    y_ini = y
    for i, celdas in enumerate(todas):
        es_cab = i == 0
        fuentes = [f_cab if es_cab else (negrita(f.size) if j == 0 else f) for j in range(len(celdas))]
        h = alto_fila(celdas, fuentes)
        if es_cab and sombreado_cabecera:
            doc.d.rectangle([x, y, x + total_ancho, y + h], fill=(222, 228, 234))
        elif i % 2 == 0:
            doc.d.rectangle([x, y, x + total_ancho, y + h], fill=(246, 246, 246))
        cx = x
        for j, (celda, ancho) in enumerate(zip(celdas, anchos)):
            fu = fuentes[j]
            # Los símbolos solos (✓/✗) se dibujan con la fuente de símbolos y más grandes
            if celda in (SI, NO):
                doc.d.text((cx + ancho // 2, y + h // 2), celda, font=f_sim, fill=TINTA_NEGRA, anchor="mm")
            else:
                lineas = lineas_celda(celda, fu, ancho)
                alto_txt = len(lineas) * int(f.size * 1.3)
                ly = y + (h - alto_txt) // 2
                for linea in lineas:
                    if linea.startswith(SI):  # "✓ (franquicia 300 €)"
                        doc.d.text((cx + ancho // 2 - fu.getlength(linea[1:]) // 2 - 12, ly), SI,
                                   font=f_sim, fill=TINTA_NEGRA)
                        doc.d.text((cx + ancho // 2 + 12, ly), linea[1:], font=fu, fill=TINTA_NEGRA, anchor="ma")
                    elif j == 0 and not es_cab:
                        doc.d.text((cx + 12, ly), linea, font=fu, fill=TINTA_NEGRA)
                    else:
                        doc.d.text((cx + ancho // 2, ly), linea, font=fu, fill=TINTA_NEGRA, anchor="ma")
                    ly += int(f.size * 1.3)
            cx += ancho
        # rejilla: línea horizontal superior de la fila
        doc.d.line([(x, y), (x + total_ancho, y)], fill=TINTA_NEGRA, width=3 if i <= 1 else 2)
        y += h
    doc.d.line([(x, y), (x + total_ancho, y)], fill=TINTA_NEGRA, width=3)
    # rejilla: líneas verticales
    cx = x
    for ancho in [0] + anchos:
        cx += ancho
        doc.d.line([(cx, y_ini), (cx, y)], fill=TINTA_NEGRA, width=2)
    # referencia en formato tabla Markdown (así la debería transcribir el VLM)
    doc.gt.append("| " + " | ".join(cabecera) + " |")
    doc.gt.append("|" + "---|" * len(cabecera))
    for celdas in filas:
        doc.gt.append("| " + " | ".join(celdas) + " |")
    return y


def documento_tabla_coberturas() -> Documento:
    doc = Documento("02_tabla_coberturas.png", semilla=202)
    y = doc.cabecera("TABLA DE COBERTURAS Y PRECIOS · SEGURO DE AUTOMÓVIL",
                     "Anexo a las Condiciones Generales CG-AUTO-2026 · Vigente desde el 1 de enero de 2026")
    anchos = [366] + [208] * 4
    y = dibujar_tabla(doc, MARGEN, y, anchos, ["Cobertura"] + MODALIDADES,
                      [[c] + v for c, v in FILAS_COBERTURAS], normal(20), negrita(20), simbolos(30))
    y += 22
    # Leyenda: los símbolos con la fuente de símbolos y el texto con Arial
    f_nota = fuente("arial.ttf", 18)
    doc.d.text((MARGEN, y - 4), SI, font=simbolos(22), fill=TINTA_NEGRA)
    doc.d.text((MARGEN + 24, y), "= incluida", font=f_nota, fill=TINTA_NEGRA)
    doc.d.text((MARGEN + 140, y - 4), NO, font=simbolos(22), fill=TINTA_NEGRA)
    doc.d.text((MARGEN + 164, y), "= no incluida.", font=f_nota, fill=TINTA_NEGRA)
    doc.gt.append("✓ = incluida ✗ = no incluida.")
    y += 30
    notas = [
        "(1) Franquicia de 300 € por siniestro, aplicable solo a daños propios.",
        "(2) Asistencia en carretera en Terceros básico: solo a partir de 25 km del domicilio habitual.",
        "(3) Precios orientativos para un turismo de hasta 1.600 cc y conductor mayor de 25 años con más de "
        "2 años de carné. Impuestos y recargo del Consorcio incluidos. La prima definitiva figura en las "
        "Condiciones Particulares.",
    ]
    for n in notas:
        y = doc.parrafo(MARGEN, y, ANCHO - 2 * MARGEN, n, fuente("arial.ttf", 18), interlineado=1.3) + 4
    y += 26
    doc.texto((MARGEN, y), "Recargos según la forma de pago", negrita(22))
    y += 38
    y = dibujar_tabla(doc, MARGEN, y, [300, 300], ["Forma de pago", "Recargo sobre la prima"],
                      [list(r) for r in RECARGOS], normal(20), negrita(20), simbolos(30))
    doc.pie("Seguros Alonso, S.A. · Tabla de coberturas TC-AUTO-2026 · Consulte las exclusiones en el artículo 3 "
            "de las Condiciones Generales")
    return doc


# ---------------------------------------------------------------------------
# (c) Guía de peritaje y asistencia (folleto a dos columnas)
# ---------------------------------------------------------------------------

def documento_guia() -> Documento:
    doc = Documento("03_guia_peritaje_y_asistencia.png", semilla=303)
    y0 = doc.cabecera("GUÍA RÁPIDA DE PERITAJE Y ASISTENCIA", "Guárdala en la guantera de tu vehículo")
    columnas = [(MARGEN, 600), (MARGEN + 640, 606)]
    f_cuerpo, f_titulo = normal(21), negrita(22)

    def banda(x, y, ancho, titulo):
        doc.d.rectangle([x, y, x + ancho, y + 40], fill=COLOR_MARCA)
        doc.texto((x + 14, y + 8), titulo, f_titulo, color="white")
        return y + 54

    def vinetas(x, y, ancho, items, numeradas=False):
        for i, item in enumerate(items, 1):
            marca = f"{i}." if numeradas else "•"
            doc.d.text((x + 4, y), marca, font=f_cuerpo, fill=TINTA_NEGRA)
            lineas = partir_lineas(item, f_cuerpo, ancho - 36)
            for linea in lineas:
                doc.d.text((x + 34, y), linea, font=f_cuerpo, fill=TINTA_NEGRA)
                y += 28
            doc.gt.append(f"{marca} {item}")
            y += 6
        return y

    # --- Columna izquierda ---
    x, ancho = columnas[0]
    y = banda(x, y0, ancho, "¿Cómo pido una cita de peritaje?")
    y = vinetas(x, y, ancho, [
        "Llama a Atención al Cliente (910 555 222) o usa el asistente de voz de Seguros Alonso.",
        "Indica la matrícula, la ciudad, el día y la franja (mañana o tarde) que prefieres.",
        "Recibirás un código de cita con el formato PER-12345. Guárdalo.",
        "Acude al taller concertado a la hora acordada. El perito tarda unos 30 minutos.",
    ], numeradas=True)
    y += 10
    y = banda(x, y, ancho, "¿Qué tengo que llevar?")
    y = vinetas(x, y, ancho, [
        "Permiso de circulación del vehículo.",
        "DNI del conductor.",
        "Parte amistoso de accidente, si se rellenó.",
        "Fotos de los daños (si las tienes).",
        "Número de póliza (por ejemplo, SA-104233).",
    ])
    y += 10
    y = banda(x, y, ancho, "Horario de peritaje")
    for linea in ["Lunes a viernes (no festivos).", "Mañana: de 9:00 a 14:00.", "Tarde: de 16:00 a 19:00."]:
        doc.texto((x + 6, y), linea, f_cuerpo)
        y += 30
    y = doc.parrafo(x + 6, y + 4, ancho - 10,
                    "Las citas se conceden desde el día siguiente a la solicitud y como máximo con 30 días "
                    "de antelación. El peritaje es gratuito.", f_cuerpo, interlineado=1.3)

    # --- Columna derecha ---
    x, ancho = columnas[1]
    y = banda(x, y0, ancho, "Teléfonos útiles")
    doc.texto((x + 6, y), "Asistencia en carretera 24 h:", negrita(22))
    doc.texto((x + 6, y + 30), "900 123 456", negrita(40), color=COLOR_MARCA)
    y += 86
    doc.texto((x + 6, y), "Atención al cliente:", negrita(22))
    doc.texto((x + 6, y + 30), "910 555 222 (L-V de 9:00 a 19:00)", normal(24))
    y += 76
    y = banda(x, y, ancho, "Límites de la grúa")
    y = vinetas(x, y, ancho, [
        "Terceros básico: desde 25 km del domicilio; remolque hasta 50 km.",
        "Terceros ampliado: desde el km 0; remolque hasta 100 km.",
        "Todo riesgo: desde el km 0; remolque hasta 150 km o hasta el taller concertado más cercano.",
    ])
    y += 10
    y = banda(x, y, ancho, "Talleres concertados")
    y = doc.parrafo(x + 6, y, ancho - 10,
                    "Madrid, Barcelona, Valencia, Sevilla, Bilbao, Zaragoza, Málaga, Gijón, Oviedo y A Coruña.",
                    f_cuerpo, interlineado=1.3) + 14
    y = banda(x, y, ancho, "Si tienes un accidente")
    y = vinetas(x, y, ancho, [
        "Señaliza el vehículo y ponte el chaleco reflectante.",
        "Si hay heridos, llama al 112.",
        "Rellena el parte amistoso con el otro conductor.",
        "Comunica el siniestro en un plazo máximo de 7 días.",
    ])
    # separador vertical entre columnas
    doc.d.line([(MARGEN + 620, y0), (MARGEN + 620, ALTO - 120)], fill=(170, 170, 170), width=2)
    doc.pie("Seguros Alonso, S.A. · Guía GPA-2026 · Información resumida: prevalecen las Condiciones Generales")
    return doc


# ---------------------------------------------------------------------------
# (d) Condiciones particulares (formulario relleno, parte a mano)
# ---------------------------------------------------------------------------

def documento_condiciones_particulares() -> Documento:
    doc = Documento("04_condiciones_particulares_SA-104233.png", semilla=404)
    y = doc.cabecera("CONDICIONES PARTICULARES · SEGURO DE AUTOMÓVIL")
    f_et, f_sec = normal(20), negrita(21)
    doc.texto((MARGEN, y), "Nº de póliza:", negrita(24))
    doc.texto((MARGEN + 170, y - 2), "SA-104233", maquina(28))
    doc.texto((ANCHO - MARGEN - 360, y), "Oficina: 0412 Gijón", normal(20))
    y += 52
    izq, der, ancho_izq, ancho_der = MARGEN, MARGEN + 650, 610, ANCHO - 2 * MARGEN - 650

    def seccion(titulo, y):
        doc.d.rectangle([MARGEN, y, ANCHO - MARGEN, y + 34], fill=(225, 230, 236))
        doc.texto((MARGEN + 10, y + 6), titulo, f_sec)
        return y + 52

    y = seccion("1. TOMADOR Y ASEGURADO", y)
    doc.campo(izq, y, ancho_izq, "Nombre y apellidos:", "Ana Ruiz Martín", f_et)
    doc.campo(der, y, ancho_der, "DNI/NIF:", "12345678Z", f_et)
    y += 54
    doc.campo(izq, y, ancho_izq, "Fecha de nacimiento:", "22/07/1988", f_et)
    doc.campo(der, y, ancho_der, "Teléfono:", "612 345 678", f_et)
    y += 54
    doc.campo(izq, y, ANCHO - 2 * MARGEN, "Domicilio:", "C/ Corrida 21, 2º A, 33206 Gijón (Asturias)", f_et)
    y += 66

    y = seccion("2. VEHÍCULO ASEGURADO", y)
    doc.campo(izq, y, ancho_izq, "Marca y modelo:", "Seat León 1.5 TSI", f_et, modo="maquina")
    doc.campo(der, y, ancho_der, "Matrícula:", "1234 BCD", f_et, modo="maquina")
    y += 50
    doc.campo(izq, y, ancho_izq, "Fecha de 1ª matriculación:", "10/05/2021", f_et, modo="maquina")
    doc.campo(der, y, ancho_der, "Uso:", "Particular", f_et, modo="maquina")
    y += 62

    y = seccion("3. CONDUCTORES", y)
    doc.campo(izq, y, ancho_izq, "Conductor habitual:", "Ana Ruiz Martín", f_et)
    doc.campo(der, y, ancho_der, "Permiso expedido el:", "05/09/2007", f_et)
    y += 54
    doc.campo(izq, y, ANCHO - 2 * MARGEN, "Conductores ocasionales:", "Ninguno declarado", f_et)
    y += 66

    y = seccion("4. MODALIDAD Y GARANTÍAS", y)
    doc.opciones(izq, y, [(m, m == "Todo riesgo con franquicia") for m in MODALIDADES[:2]], f_et, separacion=60,
                 gt_prefijo="Modalidad:")
    y += 42
    doc.opciones(izq, y, [(m, m == "Todo riesgo con franquicia") for m in MODALIDADES[2:]], f_et, separacion=60)
    y += 50
    doc.campo(izq, y, ancho_izq, "Franquicia de daños propios:", "300 €", f_et, modo="maquina")
    y += 46
    y = doc.parrafo(izq, y, ANCHO - 2 * MARGEN,
                    "Garantías incluidas: responsabilidad civil obligatoria y voluntaria (hasta 50.000.000 €), "
                    "defensa jurídica, asistencia en carretera desde km 0, lunas, robo, incendio, daños propios, "
                    "granizo y vehículo de sustitución hasta 15 días.", f_et, interlineado=1.35) + 18

    y = seccion("5. DURACIÓN Y PRIMA", y)
    doc.campo(izq, y, ancho_izq, "Fecha de efecto:", "14/03/2026", f_et, modo="maquina")
    doc.campo(der, y, ancho_der, "Vencimiento:", "14/03/2027", f_et, modo="maquina")
    y += 50
    doc.campo(izq, y, ancho_izq, "Prima anual total:", "612,40 €", f_et, modo="maquina")
    y += 50
    doc.opciones(izq, y, [("Anual", True), ("Semestral", False), ("Trimestral", False)], f_et, separacion=60,
                 gt_prefijo="Forma de pago:")
    y += 46
    doc.campo(izq, y, ANCHO - 2 * MARGEN, "Domiciliación bancaria (IBAN):", "ES76 **** **** **** **** 4821",
              f_et, modo="maquina")
    y += 70

    y = doc.parrafo(izq, y, ANCHO - 2 * MARGEN,
                    "El Tomador declara haber recibido las Condiciones Generales (CG-AUTO-2026) y la Tabla de "
                    "coberturas, y acepta expresamente las cláusulas limitativas de los artículos 3 y 4.",
                    cursiva(19), interlineado=1.35) + 20
    doc.texto((izq, y), "En", f_et, gt=False)
    doc.manuscrito(izq + 34, y - 8, "Gijón, a 10 de marzo de 2026", tamano=30, gt_extra="En Gijón, a 10 de marzo de 2026")
    y += 70
    doc.texto((izq + 40, y), "Firma del Tomador", f_et)
    doc.texto((der + 60, y), "Por Seguros Alonso, S.A.", f_et)
    doc.firma(izq + 40, y + 34)
    doc.firma(der + 80, y + 30, color=TINTA_NEGRA)
    doc.texto((der + 60, y + 110), "Departamento de Emisión", cursiva(18))
    doc.pie("Seguros Alonso, S.A. · Condiciones Particulares de la póliza SA-104233 · Ejemplar para el Tomador")
    return doc


# ---------------------------------------------------------------------------
# (e) Parte de siniestro manuscrito
# ---------------------------------------------------------------------------

def documento_parte_siniestro() -> Documento:
    doc = Documento("05_parte_siniestro_manuscrito.png", semilla=505)
    y = doc.cabecera("DECLARACIÓN DE SINIESTRO · AUTOMÓVIL",
                     "Rellene con letra clara y envíelo en un plazo máximo de 7 días")
    f_et, f_sec = normal(20), negrita(21)

    # Recuadro "a rellenar por la aseguradora" (lo rellena un empleado con otra letra y tinta negra)
    bx = ANCHO - MARGEN - 520
    doc.d.rectangle([bx, y, ANCHO - MARGEN, y + 118], outline=TINTA_NEGRA, width=2)
    doc.texto((bx + 12, y + 8), "A rellenar por Seguros Alonso", negrita(18))
    doc.campo(bx + 12, y + 40, 496, "Nº de siniestro:", "SIN-2026-004512", normal(18),
              fuente_mano="segoepr.ttf", tamano_valor=24, color=TINTA_NEGRA)
    doc.campo(bx + 12, y + 80, 496, "Fecha de recepción:", "03/11/2026", normal(18),
              fuente_mano="segoepr.ttf", tamano_valor=24, color=TINTA_NEGRA)
    izq, der = MARGEN, MARGEN + 650
    ancho_izq, ancho_der = 610, ANCHO - 2 * MARGEN - 650

    def seccion(titulo, y):
        doc.d.rectangle([MARGEN, y, ANCHO - MARGEN, y + 34], fill=(225, 230, 236))
        doc.texto((MARGEN + 10, y + 6), titulo, f_sec)
        return y + 54

    y = seccion("A. DATOS DEL ASEGURADO Y DEL VEHÍCULO", y + 140)
    doc.campo(izq, y, ancho_izq, "Nº de póliza:", "SA-104233", f_et)
    doc.campo(der, y, ancho_der, "Matrícula:", "1234 BCD", f_et)
    y += 56
    doc.campo(izq, y, ancho_izq, "Tomador:", "Ana Ruiz Martín", f_et)
    doc.campo(der, y, ancho_der, "Teléfono:", "612 345 678", f_et)
    y += 56
    doc.campo(izq, y, ancho_izq, "Marca y modelo:", "Seat León", f_et)
    y += 68

    y = seccion("B. DATOS DEL SINIESTRO", y)
    doc.campo(izq, y, ancho_izq, "Fecha del siniestro:", "02/11/2026", f_et)
    doc.campo(der, y, ancho_der, "Hora:", "18:40", f_et)
    y += 56
    doc.campo(izq, y, ANCHO - 2 * MARGEN, "Lugar:", "Gijón, parking público de la C/ Marqués de San Esteban", f_et)
    y += 58
    for pregunta, si in [("¿Hubo heridos?", False), ("¿Intervino otro vehículo?", False),
                         ("¿Intervino la policía?", False)]:
        doc.d.text((izq, y), pregunta, font=f_et, fill=TINTA_NEGRA)
        doc.opciones(izq + 300, y, [("Sí", si), ("No", not si)], f_et, separacion=50, gt_prefijo=pregunta)
        y += 42
    y += 20

    y = seccion("C. DESCRIPCIÓN DE LO OCURRIDO", y)
    descripcion = [
        "Golpe en la puerta trasera derecha al aparcar en el",
        "parking. Al maniobrar marcha atrás rocé una columna.",
        "La puerta queda abollada y con un rayón largo.",
        "No hay otros vehículos implicados ni heridos.",
    ]
    for linea in descripcion:
        doc.d.line([(izq, y + 36), (ANCHO - MARGEN, y + 36)], fill=(175, 190, 210), width=1)
        doc.manuscrito(izq + 8, y, linea, tamano=32, temblor=4, gt_extra=False)
        y += 50
    doc.gt.append(" ".join(descripcion))
    y += 18

    y = seccion("D. DAÑOS APRECIADOS", y)
    # Croquis: silueta del coche vista desde arriba con la zona dañada rodeada a mano
    cx, cy = ANCHO - MARGEN - 190, y + 70
    doc.d.rounded_rectangle([cx - 150, cy - 55, cx + 150, cy + 55], radius=40, outline=TINTA_NEGRA, width=3)
    doc.d.line([(cx - 60, cy - 55), (cx - 60, cy + 55)], fill=TINTA_NEGRA, width=2)
    doc.d.line([(cx + 40, cy - 55), (cx + 40, cy + 55)], fill=TINTA_NEGRA, width=2)
    doc.d.ellipse([cx - 95, cy + 22, cx - 10, cy + 82], outline=TINTA_AZUL, width=4)
    doc.d.text((cx + 150, cy + 64), "Frontal →", font=normal(16), fill=TINTA_GRIS, anchor="ra")
    doc.manuscrito(izq + 8, y, "Puerta trasera derecha abollada", tamano=32, temblor=4, gt_extra=False)
    doc.manuscrito(izq + 8, y + 50, "y rayón en la aleta trasera derecha.", tamano=32, temblor=4, gt_extra=False)
    doc.gt.append("Puerta trasera derecha abollada y rayón en la aleta trasera derecha.")
    y += 150

    y = seccion("E. CITA DE PERITAJE", y)
    doc.d.text((izq, y), "¿Solicita cita de peritaje?", font=f_et, fill=TINTA_NEGRA)
    doc.opciones(izq + 300, y, [("Sí", True), ("No", False)], f_et, separacion=50,
                 gt_prefijo="¿Solicita cita de peritaje?")
    y += 50
    doc.campo(izq, y, 560, "Ciudad preferida:", "Gijón", f_et)
    doc.d.text((der, y), "Franja:", font=f_et, fill=TINTA_NEGRA)
    doc.opciones(der + 90, y, [("Mañana", False), ("Tarde", True)], f_et, separacion=40, gt_prefijo="Franja:")
    y += 64

    doc.texto((izq, y), "Declaro que los datos anteriores son ciertos.", cursiva(19))
    y += 44
    doc.texto((izq, y), "En", f_et, gt=False)
    doc.manuscrito(izq + 34, y - 8, "Gijón, a 3 de noviembre de 2026", tamano=30,
                   gt_extra="En Gijón, a 3 de noviembre de 2026")
    doc.texto((der + 60, y), "Firma del asegurado", f_et)
    doc.firma(der + 80, y + 34)
    doc.sello(der - 60, y + 95, ["SEGUROS ALONSO", "RECIBIDO 03/11/2026"], angulo=7)
    doc.pie("Seguros Alonso, S.A. · Declaración de siniestro DS-AUTO · Puede adjuntar fotos y el parte amistoso")
    return doc


# ---------------------------------------------------------------------------
# Simulación de escáner
# ---------------------------------------------------------------------------

def escanear(img: Image.Image, semilla: int) -> Image.Image:
    """
    Degrada una imagen limpia para que parezca un escaneo de oficina:
      1. papel grisáceo con manchas suaves de baja frecuencia (papel reciclado/iluminación),
      2. desenfoque ligero (óptica del escáner),
      3. rotación de 0,5-2 grados (hoja mal colocada), con borde gris oscuro,
      4. ruido gaussiano + motas de polvo,
      5. compresión JPEG de calidad media (artefactos de bloque típicos de los PDF escaneados).
    """
    rng = np.random.default_rng(semilla)
    a = np.asarray(img).astype(np.float32) / 255.0
    h, w = a.shape[:2]

    # 1. Papel: tono base grisáceo-crema con variación lenta + gradiente de iluminación
    base = np.array([236, 233, 224], dtype=np.float32)
    manchas = cv2.resize(rng.normal(0, 1, (6, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    gradiente = np.linspace(1.0, 0.94, w, dtype=np.float32)[None, :]
    papel = base[None, None, :] * (1 + 0.018 * manchas[..., None]) * gradiente[..., None]
    a = a * papel  # multiplicativo: la tinta sigue oscura, el blanco toma el color del papel

    # 2. Desenfoque
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=float(rng.uniform(0.6, 0.95)))

    # 3. Rotación (y un pequeño desplazamiento)
    angulo = float(rng.uniform(0.5, 2.0) * rng.choice([-1, 1]))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    m[:, 2] += rng.uniform(-8, 8, 2)
    a = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(120, 120, 118))

    # 4. Ruido gaussiano (sensor) + motas de polvo
    a = a + rng.normal(0, 7.0, a.shape).astype(np.float32)
    for _ in range(int(rng.integers(60, 130))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 3))
        tono = float(rng.uniform(60, 150))
        cv2.circle(a, (x, y), r, (tono, tono, tono), -1)
    a = np.clip(a, 0, 255).astype(np.uint8)

    # 5. Ida y vuelta por JPEG
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 62))])
    a = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(a)


GENERADORES = [
    documento_condiciones_generales,
    documento_tabla_coberturas,
    documento_guia,
    documento_condiciones_particulares,
    documento_parte_siniestro,
]


def main() -> None:
    CARPETA_SALIDA.mkdir(exist_ok=True)
    referencia = {}
    for i, generar in enumerate(GENERADORES, 1):
        doc = generar()
        escaneada = escanear(doc.img, semilla=1000 + i)
        ruta = CARPETA_SALIDA / doc.nombre
        escaneada.save(ruta, dpi=(170, 170))
        referencia[doc.nombre] = doc.referencia()
        print(f"  {ruta.name}: {escaneada.width}x{escaneada.height}, {len(referencia[doc.nombre])} caracteres de referencia")
    ruta_gt = CARPETA_SALIDA / "ground_truth.json"
    ruta_gt.write_text(json.dumps(referencia, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Referencia guardada en {ruta_gt}")


if __name__ == "__main__":
    main()
