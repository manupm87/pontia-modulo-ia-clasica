"""
Clonación de voz para el bot de Seguros Alonso.

Dos vías:

- **XTTS-v2 (local, gratis, GPU)**: basta una muestra limpia de ~20-30 s. No hay entrenamiento:
  el modelo extrae un "embedding" de la voz de la muestra y habla con él (zero-shot).
- **ElevenLabs Instant Voice Cloning**: se suben las muestras a la API y devuelve un `voice_id`
  que se pone en `.env` (`ELEVENLABS_VOICE_ID`) junto a `TTS_BACKEND=elevenlabs`.

Uso (desde la raíz del repo):

    python -m voz.clonar_voz grabar                 # graba tu voz -> voz/muestras/referencia.wav
    python -m voz.clonar_voz demo                   # muestra sintética (edge-tts) si aún no hay voz propia
    python -m voz.clonar_voz xtts-probar            # frase de prueba con la voz clonada (XTTS)
    python -m voz.clonar_voz elevenlabs --nombre "Bot Seguros Alonso"   # sube la muestra a ElevenLabs

Consejos para la muestra: habitación silenciosa, micro a un palmo, tono natural y amable (el bot
copiará también la entonación), sin música ni eco.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from voz.tts import DIR_MUESTRAS, FRECUENCIA, a_wav, duracion, sintetizar

# Texto fonéticamente variado (todas las vocales y consonantes del español, incluidas ñ, rr, ll,
# ch, j, z, diptongos y preguntas) y en el tono del asistente, para que la voz clonada suene
# como un agente de atención al cliente. Se lee en unos 30 segundos.
TEXTO_PARA_LEER = (
    "Buenos días, le atiende el asistente virtual de Seguros Alonso. "
    "Ayer cayó una granizada enorme en Zaragoza y abolló el capó de muchos coches. "
    "No se preocupe: un perito revisará con calma la chapa, las lunas, los faros y los neumáticos. "
    "¿Prefiere que le reserve la cita por la mañana, o le viene mejor por la tarde? "
    "El pequeño taller de la carretera de Jaén ya tiene huecos este año. "
    "Recuerde traer la póliza, el parte firmado y una foto de la matrícula. "
    "¡Muchas gracias por confiar en nosotros, y que tenga un feliz día!"
)

FRASE_DE_PRUEBA = (
    "<speak>Hola, soy el asistente de Seguros Alonso. <break time=\"250ms\"/> "
    "Tu cita de peritaje para el vehículo "
    "<say-as interpret-as=\"characters\">1234</say-as><break time=\"200ms\"/>"
    "<say-as interpret-as=\"characters\">BCD</say-as> queda reservada el lunes 9 de noviembre "
    "por la mañana. <emphasis level=\"moderate\">¿Te puedo ayudar en algo más?</emphasis></speak>"
)

REFERENCIA = DIR_MUESTRAS / "referencia.wav"
REFERENCIA_DEMO = DIR_MUESTRAS / "referencia_demo.wav"
EJEMPLO = DIR_MUESTRAS / "ejemplo_voz_clonada.wav"


# ---------------------------------------------------------------------------
# Grabación de la muestra
# ---------------------------------------------------------------------------

def grabar(segundos: int, salida: Path) -> None:
    import sounddevice as sd

    print("=" * 70)
    print("Lee este texto en voz alta, con calma y en el tono con el que quieres que hable el bot:\n")
    print(_envolver(TEXTO_PARA_LEER))
    print("=" * 70)
    input(f"\nPulsa Enter y empieza a leer (se grabarán {segundos} s)...")
    for n in (3, 2, 1):
        print(f"  {n}...", end="", flush=True)
        time.sleep(0.7)
    print("  ¡Graba!")

    audio = sd.rec(int(segundos * FRECUENCIA), samplerate=FRECUENCIA, channels=1, dtype="float32")
    for restante in range(segundos, 0, -1):
        print(f"\r  quedan {restante:2d} s ", end="", flush=True)
        time.sleep(1)
    sd.wait()
    print("\r  grabación terminada.   ")

    audio = audio[:, 0]
    pico = float(np.abs(audio).max())
    if pico < 0.05:
        print("AVISO: el nivel es muy bajo. ¿Está bien elegido el micrófono? (python -m sounddevice)")
    elif pico > 0.99:
        print("AVISO: hay saturación (clipping). Aléjate un poco del micrófono y repite.")
    audio = _recortar(audio)
    if pico > 0:
        audio = audio / pico * 0.9  # normalizar el volumen

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(a_wav(audio))
    print(f"Guardado {salida} ({len(audio) / FRECUENCIA:.1f} s). Escúchalo antes de clonar.")


def _recortar(audio: np.ndarray, umbral: float = 0.02) -> np.ndarray:
    """Quita silencio al principio y al final."""
    activos = np.flatnonzero(np.abs(audio) > umbral)
    if activos.size == 0:
        return audio
    margen = FRECUENCIA // 5
    return audio[max(0, activos[0] - margen): activos[-1] + margen]


def _envolver(texto: str, ancho: int = 68) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(texto, ancho))


# ---------------------------------------------------------------------------
# Muestra de demostración (sin voz propia)
# ---------------------------------------------------------------------------

def crear_demo(voz: str, salida: Path) -> None:
    """Sintetiza el texto de referencia con otra voz de edge-tts para probar la clonación
    sin haber grabado aún una voz humana."""
    wav = sintetizar(TEXTO_PARA_LEER, backend="edge", voz=voz)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(wav)
    print(f"Muestra demo ({voz}) guardada en {salida}: {duracion(wav):.1f} s")


# ---------------------------------------------------------------------------
# XTTS-v2 local
# ---------------------------------------------------------------------------

def probar_xtts(referencia: Path, texto: str, salida: Path) -> None:
    if not referencia.exists():
        sys.exit(f"No existe {referencia}. Graba una muestra (grabar) o crea la demo (demo).")
    print(f"Clonando la voz de {referencia} con XTTS-v2...")
    os.environ["TTS_CACHE"] = "0"  # medir la latencia real
    for intento in (1, 2):  # el 1º incluye calcular el embedding de la voz (y cargar el modelo)
        inicio = time.perf_counter()
        wav = sintetizar(texto, backend="xtts", voz=str(referencia))
        t = time.perf_counter() - inicio
        print(f"  intento {intento}: {t:.2f} s para {duracion(wav):.1f} s de audio "
              f"(RTF {t / duracion(wav):.2f})")
    salida.write_bytes(wav)
    print(f"Ejemplo guardado en {salida}")


# ---------------------------------------------------------------------------
# ElevenLabs Instant Voice Cloning
# ---------------------------------------------------------------------------

def clonar_elevenlabs(muestras: list[Path], nombre: str, probar: bool) -> None:
    """POST /v1/voices/add (multipart) -> voice_id. Requiere ELEVENLABS_API_KEY y un plan que
    incluya Instant Voice Cloning."""
    import requests

    clave = os.environ.get("ELEVENLABS_API_KEY")
    if not clave:
        sys.exit("Define ELEVENLABS_API_KEY (https://elevenlabs.io/app/settings/api-keys).")
    faltan = [m for m in muestras if not m.exists()]
    if faltan:
        sys.exit(f"No existen: {', '.join(map(str, faltan))}")

    ficheros = [("files", (m.name, m.read_bytes(), "audio/wav")) for m in muestras]
    resp = requests.post(
        "https://api.elevenlabs.io/v1/voices/add",
        headers={"xi-api-key": clave},
        data={"name": nombre,
              "description": "Voz del asistente telefónico de Seguros Alonso (español de España)",
              "remove_background_noise": "true",
              "labels": '{"language": "es", "accent": "castellano", "use_case": "asistente"}'},
        files=ficheros, timeout=120,
    )
    if resp.status_code != 200:
        sys.exit(f"ElevenLabs respondió {resp.status_code}: {resp.text[:500]}")
    voice_id = resp.json()["voice_id"]
    print(f"\nVoz clonada creada. Añade a tu .env:\n\n  TTS_BACKEND=elevenlabs\n"
          f"  ELEVENLABS_VOICE_ID={voice_id}\n")
    if probar:
        os.environ["ELEVENLABS_VOICE_ID"] = voice_id
        inicio = time.perf_counter()
        wav = sintetizar(FRASE_DE_PRUEBA, backend="elevenlabs", voz=voice_id)
        destino = DIR_MUESTRAS / "ejemplo_elevenlabs.wav"
        destino.write_bytes(wav)
        print(f"Prueba: {time.perf_counter() - inicio:.2f} s -> {destino}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Clonación de voz para el bot de Seguros Alonso")
    sub = p.add_subparsers(dest="orden", required=True)

    g = sub.add_parser("grabar", help="graba tu voz leyendo un texto")
    g.add_argument("--segundos", type=int, default=30)
    g.add_argument("--salida", type=Path, default=REFERENCIA)

    d = sub.add_parser("demo", help="crea una muestra sintética para probar sin voz propia")
    d.add_argument("--voz", default="es-ES-AlvaroNeural")
    d.add_argument("--salida", type=Path, default=REFERENCIA_DEMO)

    x = sub.add_parser("xtts-probar", help="sintetiza una frase con la voz clonada (XTTS-v2)")
    x.add_argument("--referencia", type=Path, default=None,
                   help="WAV a clonar (por defecto referencia.wav o, si no existe, la demo)")
    x.add_argument("--texto", default=FRASE_DE_PRUEBA)
    x.add_argument("--salida", type=Path, default=EJEMPLO)

    e = sub.add_parser("elevenlabs", help="sube muestras a ElevenLabs (Instant Voice Cloning)")
    e.add_argument("--muestras", type=Path, nargs="+", default=[REFERENCIA])
    e.add_argument("--nombre", default="Asistente Seguros Alonso")
    e.add_argument("--sin-prueba", action="store_true")

    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # tildes en la consola de Windows
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        ejecutar(args)
    except RuntimeError as e:  # backends no configurados: mensaje claro, sin traza
        sys.exit(f"ERROR: {e}")


def ejecutar(args: argparse.Namespace) -> None:
    if args.orden == "grabar":
        grabar(args.segundos, args.salida)
    elif args.orden == "demo":
        crear_demo(args.voz, args.salida)
    elif args.orden == "xtts-probar":
        referencia = args.referencia or (REFERENCIA if REFERENCIA.exists() else REFERENCIA_DEMO)
        probar_xtts(referencia, args.texto, args.salida)
    elif args.orden == "elevenlabs":
        clonar_elevenlabs(args.muestras, args.nombre, not args.sin_prueba)


if __name__ == "__main__":
    main()
