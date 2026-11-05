# Seguros Alonso · Clara, asistente de voz

Evaluación final de **Herramientas de IA Clásica** (Máster en IA, Cloud Computing y DevOps).
https://github.com/manupm87/pontia-modulo-ia-clasica/

## El caso de uso

**Seguros Alonso** es una aseguradora de coches ficticia. Su asistente telefónico, **Clara**, atiende por voz
a los clientes que han tenido un golpe y necesitan que **un perito revise el coche**: identifica la póliza
por la matrícula (validada con expresión regular), recoge el día, la franja y la ciudad, comprueba que la
modalidad cubre el daño, que el día es laborable y que hay taller concertado, resume y pide confirmación,
y crea la cita con perito, hora y dirección. Además responde dudas sobre **cinco documentos escaneados**
(condiciones generales, tabla de coberturas y precios, guía de peritaje y asistencia, condiciones particulares
de una póliza y un parte de siniestro **manuscrito**) que se procesaron offline con OCR. Si lo que se pregunta
no está en los documentos, lo dice en lugar de inventarlo. Se puede interrumpir el flujo con una pregunta,
corregir un dato a mitad ("mejor por la tarde") o pedir hablar con una persona.

## Arquitectura

![Arquitectura](docs/arquitectura.svg)

| Pieza | Dónde | Qué hace |
| --- | --- | --- |
| Agente Dialogflow ES | `agente_dialogflow/` → `dist/SegurosAlonso.zip` | Intenciones, entidades, contexto y follow-ups |
| Webhook FastAPI | `webhook/` | Lógica de negocio, regex, eventos, interrupciones, RAG, SSML, Telegram |
| OCR offline | `ocr/` | Genera/lee los documentos, preprocesa, OCR, regex, índice vectorial |
| Voz | `voz/`, `cliente_voz.py` | STT (Dialogflow nativo o Whisper), TTS con SSML (nativo, edge, ElevenLabs, XTTS clonada) |
| Emulador de Dialogflow | `dialogflow_local/` | Ejecuta el mismo export del agente en local para probar sin cuenta de Google |
| Telegram | `telegram_bot.py` | Texto, notas de voz y botones |
| Demo | `demo/` | Página web con micrófono y panel interno, demo guiada que genera vídeo |

## Intenciones

Mapa: un único contexto de flujo (`ctx_solicitud_en_curso`) más el follow-up de confirmación.

| Intención | Contexto entrada → salida | Propósito |
| --- | --- | --- |
| `Default Welcome Intent` | – | Saludo de Clara (evento `WELCOME`), con botones en Telegram |
| `solicitar_cita` | – → `ctx_solicitud_en_curso`, `solicitar_cita-followup` | Abre el flujo y recoge **matrícula**, **fecha**, **franja** y **ciudad** (obligatorias) y el tipo de daño (opcional). Webhook en cada turno de *slot filling*. Evento `SOLICITAR_CITA` para reentrar con datos corregidos |
| `solicitar_cita - si` | follow-up → cierra contextos | Confirmación: el webhook revalida y crea la cita |
| `solicitar_cita - no` | follow-up → `ctx_solicitud_en_curso` (3) | No confirma: deja corregir un dato o la solicitud queda cancelada |
| `corregir_dato` | `ctx_solicitud_en_curso` | **Extra**: corrección a mitad de flujo ("mejor el jueves", "la matrícula es 5678 FGH"). El webhook fusiona el dato y relanza `SOLICITAR_CITA` |
| `consulta_documental` | – (retoma el flujo si estaba abierto) | Preguntas sobre los documentos → RAG con control de alucinaciones |
| `transferir_agente` | – → cierra contextos | **Extra**: transferencia a agente humano con ticket y resumen de lo recogido |
| `Default Fallback Intent` | – | Antes de rendirse intenta responder con los documentos |

Entidades: `@matricula` (**regex** `[0-9]{4}\s?[B-DF-HJ-NP-TV-Z]{3}`), `@franja` y `@tipo_dano` (propias, con
sinónimos), `@sys.date` y `@sys.geo-city` (sistema).

## Resultados medidos

**OCR** (qwen2.5vl:7b, 4-9 s por página, una sola vez):

| Documento | Tipo | CER | Palabras encontradas | Campos validados con regex |
| --- | --- | --- | --- | --- |
| 01 condiciones generales | texto corrido | 2,5 % | 97,9 % | teléfonos, importes |
| 02 tabla de coberturas | tabla con rejilla | 0,5 % | 100 % | primas anuales (285; 398,50; 589; 845,90 €) |
| 03 guía de peritaje | folleto a 2 columnas | 0,5 % | 98,6 % | teléfonos, póliza, código de cita |
| 04 condiciones particulares | formulario + manuscrito | 10,2 % | 95,1 % | DNI 12345678Z (letra verificada), matrícula, póliza, 5 fechas, prima 612,40 € |
| 05 parte de siniestro | manuscrito + sello | 32,8 % | 84,9 % | SIN-2026-004512, fechas, matrícula, póliza, teléfono |

El CER alto de los formularios se debe sobre todo al orden: el VLM reorganiza los formularios como tablas
o listas y omite cabecera y pie. El manuscrito se lee bien palabra a palabra, como muestra la columna de
palabras encontradas.

**Recuperación.** 42 fragmentos (~380 caracteres) con embeddings **bge-m3** en ChromaDB (coseno). Cada fila de la
tabla es un fragmento autodescriptivo ("Granizo · Terceros básico: ✗ · …"), y cada documento añade un fragmento con
sus campos validados. Umbral de similitud **0,40**, calibrado con 22 preguntas con respuesta y 14 sin respuesta:
corta sin gastar LLM las que no tienen nada que ver (0,30-0,38). Las preguntas del dominio sin respuesta en los
documentos (moto, vida, hogar: 0,52-0,63) se solapan con las que sí la tienen, y las paran el `NO_CONSTA` y las
comprobaciones de evidencia y cifras. Las 36 preguntas de calibración salen bien; respuesta media 0,30 s.

**Voz y latencia** (demo guiada, 11 turnos, TTS sin caché, RTX 5090):

| Etapa | Tiempo típico |
| --- | --- |
| STT Whisper `small` en GPU (3 s de audio) | 0,07-0,15 s |
| Dialogflow (emulador) + webhook | < 0,01 s; con consulta documental 0,2-0,4 s |
| TTS neuronal con SSML (2 frases) | 0,7-1,4 s |
| **Turno completo** | **media 1,29 s · máximo 2,45 s** (objetivo < 5 s) |

El informe completo de cada turno está en `demo/salida/informe.md` y el vídeo en `demo/salida/demo.mp4`
(2 min 19 s).

**Tests:** 112 pruebas con pytest (negocio, webhook, emulador, regex del OCR, RAG contra Ollama, voz, Telegram).


## Decisiones técnicas

**Validación en el webhook, no solo en Dialogflow.** La entidad regex hace que Dialogflow reconozca la
matrícula, pero el webhook vuelve a aplicar la regex tras normalizar (`1234-bcd` → `1234BCD`), busca la póliza y
aplica las reglas del negocio: la modalidad *terceros ampliado* solo permite peritaje por lunas, no hay peritos
en fin de semana, máximo 30 días de antelación y solo ciudades con taller concertado. Cada error se explica
y se pregunta solo por el dato erróneo, sin empezar de cero. Si hay un error se cierra el follow-up para que un
"sí" no confirme una cita inválida, y en la confirmación se revalida todo antes de crear la cita.

**Corrección de datos mediante eventos.** `corregir_dato` solo se activa con el contexto del flujo abierto.
El webhook fusiona el dato nuevo con los parámetros guardados en el contexto y responde con
`followupEventInput: SOLICITAR_CITA`. Los parámetros de `solicitar_cita` tienen como valor por defecto
`#SOLICITAR_CITA.<parametro>`, así que la intención se rellena sola, el webhook revalida y se vuelve a pedir
confirmación con el resumen actualizado.

**Interrupciones.** Con *webhook for slot filling* el webhook ve cada frase de la recogida de datos. Si es una
pregunta (regex de inicio interrogativo, porque el STT no siempre pone "¿"), responde con los documentos y
repite la pregunta pendiente. En la fase de confirmación, `consulta_documental` responde y, si el contexto sigue
vivo, retoma el flujo leyendo el resumen otra vez y renovando los contextos.

**Voz natural con SSML.** El cliente de voz manda `{"canal": "voz"}` en el payload. Para ese canal el
`fulfillmentText` es SSML (`<speak>`): pausas entre ideas, matrículas y códigos de cita deletreados
(`<say-as interpret-as="characters">`), énfasis en la pregunta de confirmación y la lista de documentos más lenta.
Para Telegram y chat se manda texto plano. Las respuestas son cortas (2-3 frases) porque se van a escuchar.

**OCR con modelo de visión y preprocesado.** El enunciado admite Tesseract, Azure o un VLM vía Ollama. El motor por
defecto es **qwen2.5vl:7b** en Ollama (local, GPU), porque los documentos son "complejos" a propósito: una tabla
con rejilla y ✓/✗, un folleto a dos columnas, un formulario con campos manuscritos y casillas, y un parte de
siniestro escrito a mano con sello. El VLM devuelve Markdown y conserva las tablas como tablas. Tesseract está
soportado (`--motor tesseract`, se detecta solo; modo de segmentación por documento) para comparar.
Preprocesado con OpenCV antes del OCR, cada paso con su porqué en `ocr/ocr.py`: escala de grises,
**enderezado** (estimación del ángulo con Hough y respaldo con `minAreaRect`; las páginas venían giradas 0,5-2°),
**eliminación de ruido** Non-Local Means (las motas se leen como comas y tildes) y **CLAHE** (iluminación
irregular y tinta azul débil). Solo para Tesseract se amplía ×1,5 y se binariza con umbral adaptativo; al VLM no se
le binariza porque rompe los trazos finos del manuscrito.
**Errores de OCR.** Las regex se aplican primero sobre el texto tal cual. Después, unos patrones "laxos" buscan
candidatos con las confusiones típicas (O/0, l/1, S/5, B/8) y se corrigen **solo esos tokens**, nunca el texto
entero que va al LLM; la corrección se acepta únicamente si el valor resultante pasa la validación (fecha real
de calendario, letra de control del DNI, formato de póliza/matrícula/siniestro) y queda registrada como aviso.
La calidad se mide con CER/WER contra el texto real de cada documento (`documentos/ground_truth.json`).

**Por qué no la Knowledge Base de Dialogflow.** Es una función beta que solo acepta FAQ o artículos en texto
(no imágenes), no deja controlar el prompt ni decir "no lo sé" de forma fiable, y no permite validar los campos
con regex. Con el RAG propio controlamos la recuperación, el umbral, el prompt y las comprobaciones.

**Control de alucinaciones en capas.** (1) Si ningún fragmento supera el umbral de similitud, se responde que
no consta sin llamar al LLM. (2) El prompt obliga a usar solo el CONTEXTO y a contestar `NO_CONSTA` si no está.
(3) Si la respuesta contiene cifras o importes que no aparecen en los fragmentos recuperados, se descarta.
En todos los casos el mensaje es explícito: "No dispongo de esa información en la documentación… ¿Quieres que
te pase con un agente?".

**Si el modelo tarda o falla.** Dialogflow corta el webhook a los 5 s. El OCR va offline y el índice se carga al
arrancar, el LLM se precalienta, la llamada tiene timeout (3,5 s) y `try/except`, y el manejador entero está
protegido: ante cualquier error se responde algo útil ("he tenido un problema técnico, ¿te paso con un agente?").

**Emulador local de Dialogflow.** Para probar todo sin cuenta de Google Cloud y escribir tests de extremo a
extremo, `dialogflow_local/` carga el mismo export del agente y reproduce lo que usa este proyecto: coincidencia de
frases de entrenamiento, entidades regex/propias/fecha/ciudad, contextos con vida, *slot filling* con webhook,
follow-ups, eventos y `followupEventInput`, con la petición al webhook en el formato real de Dialogflow ES v2.
Con `DIALOGFLOW_BACKEND=google` todo funciona igual contra el agente real.

## Mejoras opcionales implementadas

| Mejora del enunciado | Cómo está resuelta |
| --- | --- |
| Integración con Telegram con respuestas enriquecidas | `telegram_bot.py`: texto, **notas de voz** (Whisper → Dialogflow → nota de voz con TTS) y botones *inline* (Sí/No, accesos rápidos). El webhook también emite el payload `TELEGRAM` para la integración nativa de Dialogflow |
| STT/TTS de terceros y clonación de voz | Whisper (faster-whisper en GPU); TTS neuronal edge, **ElevenLabs** (incluida la clonación instantánea `POST /v1/voices/add`) y **XTTS-v2 local** con voz clonada a partir de 30 s grabados con `python -m voz.clonar_voz grabar` (XTTS va en un entorno aparte, `.venv-xtts`, y la primera vez pide aceptar la licencia no comercial CPML de Coqui al descargar el modelo). El SSML se respeta en todos: se trocea en segmentos con su velocidad/tono y las pausas se insertan como silencio |
| Interrupciones y corrección a mitad de flujo con eventos | `corregir_dato` + `followupEventInput: SOLICITAR_CITA`; preguntas en mitad del *slot filling* o de la confirmación se responden y el flujo se retoma |
| Base de datos vectorial | ChromaDB + embeddings bge-m3 (Ollama), con recuperación por palabras como respaldo |
| Documentos complejos | Tabla con rejilla, folleto a 2 columnas, formulario con campos manuscritos y casillas, parte de siniestro manuscrito con sello, todos "escaneados" (giro, ruido, desenfoque, JPEG) |
| Transferencia a agente humano | `transferir_agente`: ticket, resumen de lo recogido, `liveAgentHandoff` en el payload y teléfono de atención |
| Además | Emulador local de Dialogflow ES para probar sin nube, demo web con panel interno (intención, parámetros, contextos, SSML, latencias) y demo guiada que genera el vídeo |


## Cómo ejecutarlo

### 1. Entorno

```bash
uv venv --python 3.11 .venv
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

```bash
copy .env.example .env
```

Modelos locales (LLM, embeddings y visión para el OCR):

```bash
ollama pull qwen2.5:7b
```

```bash
ollama pull bge-m3
```

```bash
ollama pull qwen2.5vl:7b
```

### 2. OCR offline (una sola vez)

```bash
.venv\Scripts\python -m ocr.generar_documentos
```

```bash
.venv\Scripts\python -m ocr.ocr --motor auto
```

```bash
.venv\Scripts\python -m ocr.indexar
```

### 3. Webhook

```bash
.venv\Scripts\uvicorn webhook.main:app --port 8000
```

Para Dialogflow real hay que exponerlo a internet (VS Code → PORTS → Public, o `devtunnel host -p 8000`) y
poner la URL terminada en `/webhook` en **Fulfillment**. Comprobación: abrir la URL raíz debe devolver
`{"estado": "ok", ...}`.

### 4. Agente de Dialogflow

```bash
.venv\Scripts\python scripts\empaquetar_agente.py
```

En la consola de Dialogflow ES: ⚙️ → *Export and Import* → **Restore from ZIP** con `dist/SegurosAlonso.zip`.
Después: cambiar la URL de Fulfillment y pulsar **Train**.

### 5. Hablar con Clara

Demo web con micrófono, panel interno y demo guiada (`http://localhost:8080`):

```bash
.venv\Scripts\uvicorn demo.servidor:app --port 8080
```

Cliente de voz por terminal (Enter para hablar). Con Dialogflow real y su STT/TTS nativos:

```bash
.venv\Scripts\python cliente_voz.py --backend google --stt dialogflow --tts dialogflow
```

Todo en local (emulador + Whisper + TTS neuronal):

```bash
.venv\Scripts\python cliente_voz.py --backend local --stt whisper --tts edge
```

Demo guiada que genera `demo/salida/demo.mp4`, el audio y un informe de latencias:

```bash
.venv\Scripts\python -m demo.demo_guiada
```

Bot de Telegram (con `TELEGRAM_BOT_TOKEN` en `.env`):

```bash
.venv\Scripts\python telegram_bot.py
```

### 6. Tests

```bash
.venv\Scripts\python -m pytest -q
```
