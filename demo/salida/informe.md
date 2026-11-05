# Demo guiada: informe de turnos

| # | Paso | Whisper entendió | Intención | STT | DF+webhook | TTS | Total |
|---|---|---|---|---|---|---|---|
| 1 | Bienvenida (evento WELCOME) | (evento WELCOME) | `Default Welcome Intent` | 0.00 s | 0.00 s | 1.14 s | **1.15 s** |
| 2 | Intención solicitar_cita: arranca el flujo con contexto | Hola, quiero pedir cita con un périto para que revise un golpe en mi coche. | `solicitar_cita` | 1.51 s | 0.00 s | 0.94 s | **2.45 s** |
| 3 | Entidad regex @matricula; el webhook la valida y personaliza la siguiente pregunta | La matrícula es 1234 BCD. | `solicitar_cita` | 0.09 s | 0.00 s | 1.03 s | **1.12 s** |
| 4 | Interrupción en mitad de la recogida de datos: responde con los documentos y retoma | ¿Tengo que pagar algo por el peritaje? | `solicitar_cita` | 0.10 s | 0.24 s | 0.96 s | **1.29 s** |
| 5 | @sys.date + entidad propia @franja en una sola frase | El lunes por la mañana. | `solicitar_cita` | 0.08 s | 0.00 s | 0.78 s | **0.86 s** |
| 6 | @sys.geo-city: con todos los datos el webhook valida reglas y pide confirmación | En Gijón. | `solicitar_cita` | 0.07 s | 0.00 s | 1.00 s | **1.07 s** |
| 7 | Corrección a mitad de flujo: corregir_dato relanza el evento SOLICITAR_CITA | Mejor por la tarde. | `solicitar_cita` | 0.07 s | 0.00 s | 1.03 s | **1.11 s** |
| 8 | Follow-up de confirmación sí: el webhook crea la cita | Sí, y confirmala. | `solicitar_cita - si` | 0.14 s | 0.00 s | 1.12 s | **1.26 s** |
| 9 | Consulta documental sobre la tabla escaneada (RAG con ChromaDB) | ¿Cuánto cuesta al año el todo riesgo sin franquicia? | `consulta_documental` | 0.12 s | 0.38 s | 0.96 s | **1.46 s** |
| 10 | Control de alucinaciones: no está en los documentos y lo dice | ¿Me cubre el seguro de vida de mi mujer? | `consulta_documental` | 0.10 s | 0.18 s | 0.92 s | **1.20 s** |
| 11 | Transferencia a agente humano con ticket | Vale, quiero hablar con una persona. | `transferir_agente` | 0.08 s | 0.00 s | 1.12 s | **1.21 s** |

Latencia media por turno: **1.29 s** · máxima: **2.45 s** (objetivo < 5 s)

## Respuestas

1. **Default Welcome Intent** {} → ¡Hola! Soy Clara, la asistente virtual de Seguros Alonso. Puedo pedirte cita con un perito para revisar los daños de tu coche, o resolver dudas sobre tus coberturas y condiciones. ¿En qué te ayudo?  
   contextos: —
2. **solicitar_cita** {'tipo_dano': 'chapa'} → ¿Cuál es la matrícula del vehículo?  
   contextos: solicitar_cita_dialog_context(2), solicitar_cita_dialog_params_matricula(1)
3. **solicitar_cita** {'matricula': '1234 BCD', 'tipo_dano': 'chapa'} → Gracias, Ana. He encontrado tu Seat León. ¿Qué día te viene bien para la cita?  
   contextos: solicitar_cita_dialog_context(2), solicitar_cita_dialog_params_fecha(1)
4. **solicitar_cita** {'matricula': '1234 BCD', 'tipo_dano': 'chapa'} → No, el peritaje es gratuito para ti. Seguimos con tu cita. ¿Qué día te viene bien para la cita?  
   contextos: solicitar_cita_dialog_context(2), solicitar_cita_dialog_params_fecha(1)
5. **solicitar_cita** {'matricula': '1234 BCD', 'fecha': '2026-11-09T12:00:00+01:00', 'franja': 'manana', 'tipo_dano': 'chapa'} → Perfecto. ¿En qué ciudad quieres que te vea el perito?  
   contextos: solicitar_cita_dialog_context(2), solicitar_cita_dialog_params_ciudad(1)
6. **solicitar_cita** {'matricula': '1234 BCD', 'fecha': '2026-11-09T12:00:00+01:00', 'franja': 'manana', 'ciudad': 'Gijón', 'tipo_dano': 'chapa'} → Ana, te resumo la solicitud: cita de peritaje para tu Seat León, matrícula 1234 BCD el lunes 9 de noviembre por la mañana, en Talleres El Musel, Gijón. Motivo: daños de chapa. ¿Confirmo la cita?  
   contextos: ctx_solicitud_en_curso(10), solicitar_cita-followup(2)
7. **solicitar_cita** {'matricula': '1234BCD', 'fecha': '2026-11-09T12:00:00+01:00', 'franja': 'tarde', 'ciudad': 'Gijón', 'tipo_dano': 'chapa'} → Ana, te resumo la solicitud: cita de peritaje para tu Seat León, matrícula 1234 BCD el lunes 9 de noviembre por la tarde, en Talleres El Musel, Gijón. Motivo: daños de chapa. ¿Confirmo la cita?  
   contextos: ctx_solicitud_en_curso(10), solicitar_cita-followup(2)
8. **solicitar_cita - si** {} → ¡Hecho! Tu cita es la PER-81094. Te atenderá Elena Prieto, del equipo de peritos, el lunes 9 de noviembre a las 18:30 en Talleres El Musel, avenida de la Costa 64, Gijón. Lleva el permiso de circulación, tu DNI y, si lo tienes, el parte amistoso. ¿Te ayudo con algo más?  
   contextos: —
9. **consulta_documental** {} → El todo riesgo sin franquicia cuesta 845,90 € al año.  
   contextos: —
10. **consulta_documental** {} → No dispongo de esa información en la documentación de Seguros Alonso. ¿Quieres que te pase con un agente?  
   contextos: —
11. **transferir_agente** {} → Claro, te paso con una persona del equipo de atención al cliente. Tu número de ticket es el TCK-514449. Si se corta, puedes llamar al 910 555 222, de lunes a viernes de 9 a 14 y de 16 a 19 horas.  
   contextos: —