# Diseño técnico — Asistente de voz "Seguros Alonso"

Contrato compartido entre los módulos. Si cambias un nombre aquí, cámbialo en todos los módulos.

## Caso de uso

Aseguradora ficticia **Seguros Alonso**. El asistente de voz:

1. **Flujo de negocio multi-turno**: reserva de **cita de peritaje** (un perito revisa los daños del coche
   en un taller concertado). Recoge matrícula, fecha, franja y ciudad, valida y pide confirmación sí/no.
2. **Consulta documental**: responde preguntas sobre 5 documentos escaneados (condiciones generales,
   tabla de coberturas, guía de peritaje/asistencia, condiciones particulares de una póliza y un parte de
   siniestro manuscrito) procesados offline con OCR. Si la respuesta no está en los documentos, lo dice.

## Estructura del repositorio

```
agente_dialogflow/      export del agente Dialogflow ES (JSON, formato de export oficial)
  agent.json, package.json, entities/, intents/
scripts/empaquetar_agente.py  -> dist/SegurosAlonso.zip (para "Import/Restore" en la consola)
dialogflow_local/
  emulador.py           emulador local de Dialogflow ES que lee agente_dialogflow/ y llama al webhook real
  cliente.py            ClienteDialogflow: backend "local" (emulador) o "google" (API real)
webhook/
  main.py               FastAPI: POST /webhook, GET /
  negocio.py            pólizas simuladas, validación regex, reglas de la cita
  respuestas.py         construcción de respuestas (texto, SSML, payload Telegram, contextos, eventos)
  rag.py                MotorDocumental: recuperación (ChromaDB) + LLM + control de alucinaciones
  llm.py                backends LLM: ollama (por defecto) y gemini
ocr/
  generar_documentos.py genera los 5 documentos "escaneados" (PNG) en documentos/
  ocr.py                OCR offline (preprocesado OpenCV + VLM Ollama / Tesseract) + regex -> data/documentos.json
  indexar.py            trocea y crea embeddings (Ollama bge-m3) en ChromaDB -> data/chroma/
voz/
  ssml.py               utilidades SSML (construir, convertir a texto, segmentar)
  tts.py                TTS: edge (por defecto), elevenlabs, xtts (voz clonada)
  stt.py                STT: faster-whisper
  clonar_voz.py         clonación de voz (ElevenLabs IVC y XTTS-v2 local)
cliente_voz.py          micrófono -> Dialogflow (local o Google) -> altavoz
telegram_bot.py         puente Telegram (texto + notas de voz + botones)
demo/                   servidor y página de demo + demo guiada que genera audio/vídeo
tests/                  pytest
```

Python: `.venv\Scripts\python.exe` (3.11). Ejecutar siempre desde la raíz del repo.
Imports absolutos desde la raíz (`from webhook.negocio import ...`, `from voz.tts import ...`).

## Agente Dialogflow ES

- Idioma `es`, zona horaria `Europe/Madrid`.
- **Contexto principal**: `ctx_solicitud_en_curso` (lifespan 10).
- **Follow-up**: `solicitar_cita-followup` (lifespan 2), creado por `solicitar_cita`.

### Entidades

| Entidad | Tipo | Detalle |
| --- | --- | --- |
| `@matricula` | regex | `[0-9]{4}\s?[B-DF-HJ-NP-TV-Z]{3}` (matrícula española actual) |
| `@franja` | custom | `manana` (sinónimos: por la mañana, de mañana, a primera hora, temprano, antes de comer) / `tarde` (por la tarde, de tarde, después de comer, a última hora) |
| `@tipo_dano` | custom | `chapa` (golpe, abolladura, chapa, rayón, carrocería), `lunas` (luna, parabrisas, cristal), `mecanica` (motor, avería, mecánica), `granizo` (granizo, pedrisco) |
| `@sys.date`, `@sys.geo-city` | sistema | fecha y ciudad |

### Intenciones (6 significativas + bienvenida + fallback)

| Intención | Contextos entrada | Contextos salida | Webhook | Propósito |
| --- | --- | --- | --- | --- |
| `Default Welcome Intent` | – | – | sí | Saludo (evento `WELCOME`) |
| `solicitar_cita` | – | `ctx_solicitud_en_curso`(10), `solicitar_cita-followup`(2) | sí + slot filling | Recoge `matricula`*, `fecha`*, `franja`*, `ciudad`*, `tipo_dano` (opcional). Evento `SOLICITAR_CITA` |
| `solicitar_cita - si` | `ctx_solicitud_en_curso`, `solicitar_cita-followup` | `ctx_solicitud_en_curso`(0) | sí | Confirma y crea la cita |
| `solicitar_cita - no` | `ctx_solicitud_en_curso`, `solicitar_cita-followup` | `ctx_solicitud_en_curso`(3) | sí | No confirma; invita a corregir un dato o deja la solicitud cancelada |
| `corregir_dato` | `ctx_solicitud_en_curso` | `ctx_solicitud_en_curso`(10) | sí | Corrige un dato a mitad de flujo ("mejor el jueves", "la matrícula es 5678 FGH") → el webhook reenvía el evento `SOLICITAR_CITA` con los parámetros fusionados |
| `consulta_documental` | – | – | sí | Preguntas sobre coberturas/condiciones → RAG. Si hay solicitud en curso, responde y **retoma** el flujo (gestión de interrupciones) |
| `transferir_agente` | – | `ctx_solicitud_en_curso`(0) | sí | Transferencia a agente humano con ticket y resumen |
| `Default Fallback Intent` | – | – | sí | Intenta responder con los documentos; si no, lo dice |

Parámetros de `solicitar_cita` (todos con `value: $<nombre>`). Los requeridos tienen
`defaultValue: "#SOLICITAR_CITA.<nombre>"` para que el evento de corrección los rellene.
Prompts: matrícula "¿Cuál es la matrícula del vehículo?", fecha "¿Qué día te viene bien para la cita?",
franja "¿Prefieres por la mañana o por la tarde?", ciudad "¿En qué ciudad quieres que te vea el perito?".

`corregir_dato` tiene los mismos parámetros, todos opcionales y sin default.

### Canal de voz y SSML

El cliente que habla por voz envía `queryParams.payload = {"canal": "voz"}` (llega al webhook en
`originalDetectIntentRequest.payload.canal`). Para canal voz, `fulfillmentText` es **SSML**
(`<speak>…</speak>`), que el TTS nativo de Dialogflow interpreta. Para el resto de canales,
`fulfillmentText` es texto plano. Siempre se incluye además en `fulfillmentMessages`:

- `{"text": {"text": [texto_plano]}}`
- `{"payload": {"ssml": "<speak>…</speak>", "texto": "...", "botones": ["Sí", "No"]}}` (genérico)
- `{"platform": "TELEGRAM", "payload": {"telegram": {"text": "...", "reply_markup": {...}}}}`

## Interfaces Python

### dialogflow_local/cliente.py
```python
@dataclass
class Turno:
    query_text: str            # lo que entendió (transcripción si fue audio)
    intent: str                # displayName ("" si nada)
    confianza: float
    parametros: dict
    contextos: list[dict]      # [{"name": "ctx_solicitud_en_curso", "lifespan": 9, "parameters": {...}}] (nombre corto)
    texto: str                 # respuesta en texto plano (sin etiquetas SSML)
    ssml: str | None           # SSML si la respuesta lo trae
    mensajes: list[dict]       # fulfillmentMessages crudos
    audio: bytes | None        # WAV de respuesta si lo generó Dialogflow (backend google)
    latencias: dict[str, float]  # segundos: {"dialogflow": .., "webhook": .., "total": ..}

def crear_cliente(backend: str | None = None) -> "ClienteDialogflow"   # env DIALOGFLOW_BACKEND = local|google
class ClienteDialogflow:
    def texto(self, session_id: str, texto: str, payload: dict | None = None) -> Turno
    def evento(self, session_id: str, nombre: str, parametros: dict | None = None, payload: dict | None = None) -> Turno
```
El emulador llama al webhook por HTTP (`WEBHOOK_URL`, por defecto `http://127.0.0.1:8000/webhook`)
o en proceso si se le pasa una app ASGI/callable (para tests).

### webhook/rag.py
```python
@dataclass
class RespuestaDocumental:
    texto: str                 # respuesta para el usuario (2-3 frases, apta para voz)
    respondida: bool           # False si no consta en los documentos
    fuentes: list[str]         # nombres de documento usados
    latencia: float

class MotorDocumental:
    def __init__(self, ruta_json="data/documentos.json", ruta_chroma="data/chroma"): ...
    def recuperar(self, pregunta: str, k: int = 4) -> list[dict]   # [{"documento","texto","similitud"}]
    def responder(self, pregunta: str) -> RespuestaDocumental      # nunca lanza excepción
```

### webhook/llm.py
```python
def generar(prompt: str, sistema: str = "", max_tokens: int = 200, timeout: float = 3.5) -> str
# LLM_BACKEND=ollama|gemini ; OLLAMA_MODEL (por defecto qwen2.5:7b) ; GOOGLE_API_KEY para gemini
```

### voz/
```python
# ssml.py
def a_ssml(texto: str) -> str                      # envuelve en <speak>, escapa XML
def ssml_a_texto(ssml: str) -> str                 # quita etiquetas, para mostrar
def segmentar(ssml: str) -> list[dict]             # [{"tipo":"texto","texto":..,"rate":"+0%","pitch":"+0Hz"} | {"tipo":"pausa","ms":300}]
# tts.py
def sintetizar(texto_o_ssml: str, backend: str | None = None, voz: str | None = None) -> bytes   # WAV mono 16-bit (24 kHz)
# TTS_BACKEND=edge|elevenlabs|xtts ; respeta <break>, <prosody>, <emphasis>, <say-as> segmentando
# stt.py
def transcribir(audio: bytes, idioma: str = "es") -> str   # acepta wav/ogg/webm/mp3 (ffmpeg); WHISPER_MODEL por defecto "small"
```

## Datos de negocio (webhook/negocio.py)

| Matrícula | Titular | Vehículo | Póliza | Modalidad | Peritaje daños propios |
| --- | --- | --- | --- | --- | --- |
| 1234BCD | Ana Ruiz | Seat León | SA-104233 | Todo riesgo con franquicia 300 € | sí |
| 5678FGH | Luis Gómez | Volkswagen Golf | SA-208871 | Terceros ampliado | no (solo lunas, incendio y robo) |
| 9012JKL | María López | Toyota Corolla | SA-311540 | Todo riesgo sin franquicia | sí |

Reglas de la cita: fecha futura (desde mañana) y como máximo 30 días; de lunes a viernes;
ciudad con taller concertado (Madrid, Barcelona, Valencia, Sevilla, Bilbao, Zaragoza, Málaga, Gijón,
Oviedo, A Coruña); franja mañana 9:00–14:00 o tarde 16:00–19:00. Código de cita `PER-` + 5 dígitos.
