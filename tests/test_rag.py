"""
Pruebas de la consulta documental (webhook/rag.py).

- Las pruebas unitarias (controles de alucinación, formato, modo sin embeddings) no
  necesitan Ollama: el LLM se sustituye por una función falsa.
- Las pruebas de extremo a extremo usan Ollama de verdad (bge-m3 + qwen2.5:7b) y
  ChromaDB; se saltan si Ollama no está levantado o falta el índice.
"""

import json
import os
from pathlib import Path

import httpx
import pytest

from webhook import llm, rag
from webhook.rag import MENSAJE_ERROR, MENSAJE_NO_CONSTA, MotorDocumental

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "data" / "documentos.json"
RUTA_CHROMA = RAIZ / "data" / "chroma"


# ---------------------------------------------------------------------------
# Utilidades de control de alucinaciones (sin Ollama)
# ---------------------------------------------------------------------------

def test_numeros_respaldados_por_el_contexto():
    contexto = "Asistencia 24 h: 900 123 456. Franquicia de 300 €. RC hasta 50.000.000 €. Precio 398,50 €."
    assert rag.numeros_sin_respaldo("Llama al 900123456, la franquicia es de 300 €.", contexto) == []
    assert rag.numeros_sin_respaldo("Cubre hasta 50000000 euros y cuesta 398,50 €.", contexto) == []
    assert rag.numeros_sin_respaldo("La franquicia es de 500 €.", contexto) == ["500"]


def test_separar_evidencia_y_respuesta():
    ev, resp = rag.separar_salida("EVIDENCIA: Franquicia de 300 €.\nRESPUESTA: La franquicia es de 300 €.")
    assert ev == "Franquicia de 300 €." and resp == "La franquicia es de 300 €."
    ev, resp = rag.separar_salida("Respuesta libre sin formato.")
    assert ev == "" and resp == "Respuesta libre sin formato."


def test_limpiar_para_voz_quita_markdown_y_frases_cortadas():
    assert rag.limpiar_para_voz("**Sí**, cubre:\n- lunas\n- robo.") == "Sí, cubre: lunas robo."
    assert rag.limpiar_para_voz("Primera frase completa. Segunda frase cort") == "Primera frase completa."


# ---------------------------------------------------------------------------
# Motor con el LLM simulado (sin Ollama: modo de búsqueda por palabras clave)
# ---------------------------------------------------------------------------

@pytest.fixture
def motor_palabras(tmp_path):
    if not RUTA_JSON.is_file():
        pytest.skip("falta data/documentos.json (ejecuta python -m ocr.ocr)")
    # Una carpeta de Chroma vacía obliga al modo degradado por palabras clave
    motor = MotorDocumental(ruta_json=RUTA_JSON, ruta_chroma=tmp_path / "chroma_vacio")
    assert motor.modo == "palabras"
    return motor


def test_modo_palabras_recupera_el_fragmento_correcto(motor_palabras):
    r = motor_palabras.recuperar("¿Qué tengo que llevar al peritaje?")
    assert r and r[0]["documento"].startswith("03_")


def test_pregunta_ajena_no_llama_al_llm(motor_palabras, monkeypatch):
    def no_deberia_llamarse(*a, **k):
        raise AssertionError("el LLM no debe llamarse si no hay nada parecido en los documentos")
    monkeypatch.setattr(llm, "generar", no_deberia_llamarse)
    r = motor_palabras.responder("¿Cuál es la capital de Francia?")
    assert not r.respondida and r.texto == MENSAJE_NO_CONSTA


def test_no_consta_del_llm(motor_palabras, monkeypatch):
    monkeypatch.setattr(llm, "generar", lambda *a, **k: "EVIDENCIA: NINGUNA\nRESPUESTA: NO_CONSTA")
    r = motor_palabras.responder("¿Qué tengo que llevar al peritaje?")
    assert not r.respondida and r.texto == MENSAJE_NO_CONSTA and r.fuentes == []


def test_cifra_inventada_se_descarta(motor_palabras, monkeypatch):
    monkeypatch.setattr(llm, "generar", lambda *a, **k:
                        "EVIDENCIA: Permiso de circulación del vehículo.\nRESPUESTA: Lleva el permiso y paga 45 €.")
    r = motor_palabras.responder("¿Qué tengo que llevar al peritaje?")
    assert not r.respondida


def test_evidencia_inventada_se_descarta(motor_palabras, monkeypatch):
    monkeypatch.setattr(llm, "generar", lambda *a, **k:
                        "EVIDENCIA: El perito te invita a un café y te regala una bufanda.\n"
                        "RESPUESTA: El perito te invita a un café.")
    r = motor_palabras.responder("¿Qué tengo que llevar al peritaje?")
    assert not r.respondida


def test_respuesta_valida_con_fuentes(motor_palabras, monkeypatch):
    monkeypatch.setattr(llm, "generar", lambda *a, **k:
                        "EVIDENCIA: Permiso de circulación del vehículo.\n"
                        "RESPUESTA: Lleva el permiso de circulación y el DNI del conductor.")
    r = motor_palabras.responder("¿Qué tengo que llevar al peritaje?")
    assert r.respondida
    assert r.fuentes == ["03_guia_peritaje_y_asistencia.png"]


def test_error_del_llm_no_lanza_excepcion(motor_palabras, monkeypatch):
    def lento(*a, **k):
        raise TimeoutError("demasiado lento")
    monkeypatch.setattr(llm, "generar", lento)
    r = motor_palabras.responder("¿Qué tengo que llevar al peritaje?")
    assert not r.respondida and r.texto == MENSAJE_ERROR

    monkeypatch.setattr(llm, "generar", lambda *a, **k: 1 / 0)
    r = motor_palabras.responder("¿Qué tengo que llevar al peritaje?")
    assert not r.respondida and r.texto == MENSAJE_ERROR


# ---------------------------------------------------------------------------
# Extremo a extremo con Ollama + ChromaDB
# ---------------------------------------------------------------------------

def _ollama_listo() -> bool:
    try:
        url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        modelos = {m["name"] for m in httpx.get(f"{url}/api/tags", timeout=2).json()["models"]}
        return {"bge-m3:latest", llm.modelo_ollama()} <= modelos and RUTA_CHROMA.is_dir()
    except Exception:
        return False


requiere_ollama = pytest.mark.skipif(not _ollama_listo(), reason="Ollama/ChromaDB no disponibles")


@pytest.fixture(scope="module")
def motor():
    os.environ.setdefault("LLM_BACKEND", "ollama")
    m = MotorDocumental()
    assert m.modo == "vectorial"
    llm.precalentar()
    m.responder("¿Cuál es el horario del peritaje?")   # primera llamada de calentamiento
    return m


RESPONDIBLES = [
    # (pregunta, algún texto que debe aparecer en la respuesta, documento fuente esperado)
    ("¿Cuál es la franquicia del todo riesgo?", "300", "01_"),
    ("¿Cuál es el teléfono de asistencia 24 horas?", "900 123 456", "03_"),
    ("¿Cuánto tiempo tengo para comunicar un siniestro?", "7 días", "01_"),
    ("¿Qué tengo que llevar al peritaje?", "DNI", "03_"),
    ("¿Cuánto cuesta el seguro a terceros ampliado?", "398,50", "02_"),
]

NO_RESPONDIBLES = [
    "¿Cubre el seguro de vida?",
    "¿Cuál es la capital de Francia?",
    "¿Cuánto cuesta un seguro de moto?",
]


@requiere_ollama
@pytest.mark.parametrize("pregunta,esperado,fuente", RESPONDIBLES)
def test_preguntas_respondibles(motor, pregunta, esperado, fuente):
    r = motor.responder(pregunta)
    assert r.respondida, r.texto
    assert esperado.lower() in r.texto.lower(), r.texto
    assert any(f.startswith(fuente) for f in r.fuentes), r.fuentes
    assert r.latencia < 4.0
    assert "*" not in r.texto and "\n" not in r.texto   # apta para voz


@requiere_ollama
@pytest.mark.parametrize("pregunta", NO_RESPONDIBLES)
def test_preguntas_no_respondibles(motor, pregunta):
    r = motor.responder(pregunta)
    assert not r.respondida, r.texto
    assert r.texto == MENSAJE_NO_CONSTA
    assert r.latencia < 4.0


@requiere_ollama
def test_recuperar_devuelve_similitudes_ordenadas(motor):
    r = motor.recuperar("¿Cuánto tarda el perito?", k=4)
    assert len(r) == 4 and set(r[0]) == {"documento", "texto", "similitud"}
    sims = [x["similitud"] for x in r]
    assert sims == sorted(sims, reverse=True) and sims[0] > rag.UMBRAL_SIMILITUD
