"""
Backends de modelo de lenguaje para la consulta documental.

  LLM_BACKEND=ollama (por defecto): modelo local servido por Ollama
      (OLLAMA_MODEL, por defecto qwen2.5:7b). Sin coste, sin enviar datos fuera
      y con latencia estable (~0,5-1 s en GPU con el modelo ya cargado).
  LLM_BACKEND=gemini: Gemini 2.5 Flash con la librería google-genai
      (necesita GOOGLE_API_KEY), como en el código del curso.

Interfaz única:
    generar(prompt, sistema="", max_tokens=200, timeout=3.5) -> str
Lanza TimeoutError si el modelo no contesta a tiempo (Dialogflow corta el
webhook a los 5 s, así que es mejor rendirse antes y dar una respuesta educada).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturoTimeout

import httpx

# 127.0.0.1 y no "localhost": en Windows "localhost" intenta antes IPv6 (::1) y cada conexión nueva
# pierde ~2 s hasta caer a IPv4. Medido: 2,1 s con localhost frente a 0,07 s con 127.0.0.1.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "60m")   # mantiene el modelo en la GPU entre preguntas
# Mismo tamaño de contexto en el precalentado y en las preguntas: si cambia, Ollama recarga el modelo
NUM_CTX = 4096
MODELO_GEMINI = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Un cliente HTTP reutilizable: mantiene la conexión abierta (ahorra el handshake en cada pregunta)
_http = httpx.Client(timeout=httpx.Timeout(10.0, connect=1.0))
# Hilos para poder imponer un tiempo máximo TOTAL a cualquier backend
_ejecutor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")
_gemini = None
_cerrojo_gemini = threading.Lock()


def backend_actual() -> str:
    return os.getenv("LLM_BACKEND", "ollama").strip().lower()


def modelo_ollama() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _generar_ollama(prompt: str, sistema: str, max_tokens: int, timeout: float) -> str:
    mensajes = ([{"role": "system", "content": sistema}] if sistema else []) + \
               [{"role": "user", "content": prompt}]
    r = _http.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": modelo_ollama(),
            "messages": mensajes,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": NUM_CTX},
        },
        timeout=httpx.Timeout(timeout, connect=1.0),
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _cliente_gemini():
    global _gemini
    with _cerrojo_gemini:
        if _gemini is None:
            from google import genai
            clave = os.getenv("GOOGLE_API_KEY")
            if not clave:
                raise RuntimeError("Falta GOOGLE_API_KEY para usar LLM_BACKEND=gemini")
            _gemini = genai.Client(api_key=clave)
    return _gemini


def _generar_gemini(prompt: str, sistema: str, max_tokens: int, timeout: float) -> str:
    from google.genai import types
    respuesta = _cliente_gemini().models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=sistema or None,
            temperature=0.1,
            max_output_tokens=max_tokens,
            # gemini-2.5-flash "piensa" por defecto: lo desactivamos para ganar latencia
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            # No dejamos que el modelo llame a funciones Python por su cuenta
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        ),
    )
    return (respuesta.text or "").strip()


# ---------------------------------------------------------------------------
# Interfaz pública
# ---------------------------------------------------------------------------

def generar(prompt: str, sistema: str = "", max_tokens: int = 200, timeout: float = 3.5) -> str:
    """
    Genera una respuesta con el backend configurado en LLM_BACKEND.

    El tiempo máximo se impone de forma TOTAL (no solo por operación de red)
    ejecutando la llamada en un hilo y esperando como mucho `timeout` segundos.
    Lanza TimeoutError si se supera y propaga cualquier otro error del backend.
    """
    backend = backend_actual()
    funcion = {"ollama": _generar_ollama, "gemini": _generar_gemini}.get(backend)
    if funcion is None:
        raise ValueError(f"LLM_BACKEND desconocido: {backend!r} (usa 'ollama' o 'gemini')")
    futuro = _ejecutor.submit(funcion, prompt, sistema, max_tokens, timeout)
    try:
        return futuro.result(timeout=timeout)
    except (_FuturoTimeout, httpx.TimeoutException) as e:
        futuro.cancel()
        raise TimeoutError(f"El LLM ({backend}) no respondió en {timeout:.1f} s") from e


def precalentar() -> None:
    """
    Carga el modelo en memoria al arrancar el webhook. La primera carga de un
    modelo de 7B en la GPU tarda varios segundos; si ocurriera durante la
    primera pregunta, Dialogflow cortaría la llamada.
    Con Gemini no hay nada que cargar.
    """
    if backend_actual() != "ollama":
        return
    _http.post(f"{OLLAMA_URL}/api/generate",
               json={"model": modelo_ollama(), "prompt": "", "keep_alive": KEEP_ALIVE,
                     "options": {"num_ctx": NUM_CTX}},
               timeout=httpx.Timeout(120.0, connect=2.0)).raise_for_status()


if __name__ == "__main__":
    import time
    precalentar()
    t0 = time.perf_counter()
    print(generar("Di 'hola' en una palabra.", max_tokens=10))
    print(f"{time.perf_counter() - t0:.2f} s")
