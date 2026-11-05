"""
Canal de voz del asistente de Seguros Alonso.

- `voz.ssml`       construir SSML natural (pausas, énfasis, matrículas, fechas) y segmentarlo
- `voz.tts`        texto/SSML -> WAV (edge-tts, ElevenLabs o XTTS-v2 con voz clonada)
- `voz.stt`        audio -> texto con faster-whisper (+ normalización de matrículas habladas)
- `voz.clonar_voz` CLI para grabar una muestra de voz propia y clonarla

Los módulos pesados (whisper, edge-tts, XTTS) se importan de forma perezosa para que
`import voz` sea instantáneo y el webhook pueda usar `voz.ssml` sin dependencias extra.
"""
