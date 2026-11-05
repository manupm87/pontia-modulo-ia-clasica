"""Pruebas de la extracción y validación con expresiones regulares (ocr/ocr.py)."""

import json
from pathlib import Path

import pytest

from ocr.ocr import (calcular_metricas, extraer_campos, letra_dni, normalizar_ocr_token, validar_dni,
                     validar_fecha)

RAIZ = Path(__file__).resolve().parent.parent


# --- DNI / NIE ----------------------------------------------------------------

@pytest.mark.parametrize("dni", ["12345678Z", "12345678-Z", "12345678 Z", "X1234567L", "00000000T"])
def test_dni_valido(dni):
    assert validar_dni(dni)[0]


@pytest.mark.parametrize("dni", ["12345678A", "87654321Z", "X1234567A", "1234567Z"])
def test_dni_con_letra_incorrecta_o_mal_formado(dni):
    assert not validar_dni(dni)[0]


def test_letra_dni():
    assert letra_dni(12345678) == "Z"   # 12345678 % 23 = 14 -> Z


def test_dni_con_letra_incorrecta_genera_aviso():
    campos, avisos = extraer_campos("DNI/NIF: 12345678A")
    assert campos["dnis"] == []
    assert any("debería ser Z" in a for a in avisos)


# --- Fechas -------------------------------------------------------------------

def test_fechas_se_validan_contra_el_calendario():
    campos, avisos = extraer_campos("Efecto: 14/03/2026. Vencimiento 14-03-2027. Imposible: 31/02/2026.")
    assert campos["fechas"] == ["2026-03-14", "2027-03-14"]
    assert any("31/02/2026" in a for a in avisos)


def test_fecha_con_anio_de_dos_cifras():
    assert validar_fecha("2", "11", "26") == "2026-11-02"
    assert validar_fecha("29", "02", "2027") is None   # 2027 no es bisiesto


# --- Importes, identificadores y teléfonos -------------------------------------

def test_importes_normalizados_a_float():
    texto = "Prima anual total: 612,40 €. RC hasta 50.000.000 €. Franquicia de 300 euros. Precio 285,00 EUR"
    campos, _ = extraer_campos(texto)
    assert campos["importes"] == [612.40, 50_000_000.0, 300.0, 285.0]


def test_identificadores_de_seguros_alonso():
    texto = ("Póliza SA-104233, siniestro SIN-2026-004512, cita PER-48213, matrícula 1234 BCD. "
             "Matrícula imposible con vocales: 1234 ABC.")
    campos, _ = extraer_campos(texto)
    assert campos["polizas"] == ["SA-104233"]
    assert campos["siniestros"] == ["SIN-2026-004512"]
    assert campos["citas"] == ["PER-48213"]
    assert campos["matriculas"] == ["1234BCD"]


def test_telefonos_espanoles_en_distintos_formatos():
    texto = "Asistencia 24 h: 900 123 456. Cliente: 910555222. Móvil: +34 612 34 56 78. Código postal 28013."
    campos, _ = extraer_campos(texto)
    assert campos["telefonos"] == ["900 123 456", "910 555 222", "612 345 678"]


# --- Confusiones típicas de OCR -------------------------------------------------

def test_confusiones_ocr_se_corrigen_solo_en_candidatos():
    texto = "Póliza SA-1O4233 · DNI l2345678Z · Siniestro SIN-2O26-OO4512 · Fecha l4/O3/2026 · Matrícula l234 8CD"
    campos, avisos = extraer_campos(texto)
    assert "SA-104233" in campos["polizas"]
    assert "12345678Z" in campos["dnis"]
    assert "SIN-2026-004512" in campos["siniestros"]
    assert "2026-03-14" in campos["fechas"]
    assert "1234BCD" in campos["matriculas"]
    assert any("confusión OCR" in a for a in avisos)


def test_dni_corregido_que_no_valida_se_descarta():
    # 'l2345678A' -> '12345678A', pero la letra correcta es Z: no se acepta
    campos, _ = extraer_campos("DNI l2345678A")
    assert campos["dnis"] == []


def test_el_texto_normal_no_se_destroza():
    # El código del curso cambiaba S->5 y O->0 en TODO el texto; aquí no se toca nada que no sea candidato
    campos, avisos = extraer_campos("SEGUROS ALONSO, S.A. Condiciones Generales. Oficina de Bilbao.")
    assert all(v == [] for v in campos.values())
    assert avisos == []
    assert normalizar_ocr_token("1O4Z33") == "104233"


# --- Métrica de calidad ---------------------------------------------------------

def test_metricas_ignoran_formato_markdown():
    referencia = "| Cobertura | Lunas |\n|---|---|\n| Terceros | ✗ |"
    reconocido = "Cobertura Lunas\nTerceros ✘"
    m = calcular_metricas(reconocido, referencia)
    assert m["cer"] == 0 and m["wer"] == 0


def test_metricas_detectan_errores():
    m = calcular_metricas("La franquicia es de 3OO euros", "La franquicia es de 300 euros")
    assert 0 < m["cer"] < 0.1
    assert m["wer"] == pytest.approx(1 / 6, abs=1e-3)


# --- Resultado real del pipeline (si ya se ha ejecutado python -m ocr.ocr) -------

@pytest.mark.skipif(not (RAIZ / "data" / "documentos.json").is_file(), reason="falta data/documentos.json")
def test_campos_extraidos_de_los_documentos_reales():
    docs = {d["documento"]: d for d in json.loads((RAIZ / "data" / "documentos.json").read_text(encoding="utf-8"))}
    poliza = docs["04_condiciones_particulares_SA-104233.png"]["campos"]
    assert "12345678Z" in poliza["dnis"]
    assert "1234BCD" in poliza["matriculas"]
    assert {"2026-03-14", "2027-03-14"} <= set(poliza["fechas"])
    assert 612.40 in poliza["importes"]
    parte = docs["05_parte_siniestro_manuscrito.png"]["campos"]
    assert "SIN-2026-004512" in parte["siniestros"]
    assert "2026-11-02" in parte["fechas"]
    guia = docs["03_guia_peritaje_y_asistencia.png"]["campos"]
    assert {"900 123 456", "910 555 222"} <= set(guia["telefonos"])
