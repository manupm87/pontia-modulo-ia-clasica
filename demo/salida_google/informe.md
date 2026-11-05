# Demo guiada: informe de turnos

| # | Paso | STT entendió | Intención | STT | DF+webhook | TTS | Total |
|---|---|---|---|---|---|---|---|
| 1 | Bienvenida (evento WELCOME) | (evento WELCOME) | `Default Welcome Intent` | 0.00 s | 1.29 s | 0.00 s | **1.29 s** |
| 2 | Intención solicitar_cita: arranca el flujo con contexto | hola quiero pedir cita con un perito para que revise un golpe en mi coche | `solicitar_cita` | 0.00 s | 1.80 s | 0.00 s | **1.80 s** |
| 3 | Entidad regex @matricula; el webhook la valida y personaliza la siguiente pregunta | la matrícula es 1234 be ce de | `solicitar_cita` | 0.00 s | 1.70 s | 0.00 s | **1.70 s** |
| 4 | Interrupción en mitad de la recogida de datos: responde con los documentos y retoma | tengo que pagar algo por el peritaje | `consulta_documental` | 0.00 s | 1.88 s | 0.00 s | **1.88 s** |
| 5 | @sys.date + entidad propia @franja en una sola frase | el lunes por la mañana | `solicitar_cita` | 0.00 s | 1.39 s | 0.00 s | **1.39 s** |
| 6 | @sys.geo-city: con todos los datos el webhook valida reglas y pide confirmación | en Gijón | `solicitar_cita` | 0.00 s | 1.35 s | 0.00 s | **1.35 s** |
| 7 | Corrección a mitad de flujo: corregir_dato relanza el evento SOLICITAR_CITA | mejor por la tarde | `solicitar_cita` | 0.00 s | 1.41 s | 0.00 s | **1.41 s** |
| 8 | Follow-up de confirmación sí: el webhook crea la cita | sí y confirma la | `solicitar_cita - si` | 0.00 s | 1.01 s | 0.00 s | **1.01 s** |
| 9 | Consulta documental sobre la tabla escaneada (RAG con ChromaDB) | cuánto cuesta el año el todo riesgo sin franquicia | `Default Fallback Intent` | 0.00 s | 2.18 s | 0.00 s | **2.18 s** |
| 10 | Control de alucinaciones: no está en los documentos y lo dice | me cubre el seguro de be de de mi mujer | `consulta_documental` | 0.00 s | 1.64 s | 0.00 s | **1.64 s** |
| 11 | Transferencia a agente humano con ticket | vale quiero hablar con una persona | `transferir_agente` | 0.00 s | 1.47 s | 0.00 s | **1.47 s** |

Latencia media por turno: **1.56 s** · máxima: **2.18 s** (objetivo < 5 s)

## Respuestas

1. **Default Welcome Intent** {} → ¡Hola! Soy Clara, la asistente virtual de Seguros Alonso. Puedo pedirte cita con un perito para revisar los daños de tu coche, o resolver dudas sobre tus coberturas y condiciones. ¿En qué te ayudo?  
   contextos: —
2. **solicitar_cita** {'tipo_dano': 'chapa'} → ¿Cuál es la matrícula del vehículo?  
   contextos: ctx_solicitud_en_curso(10), 65200a1a-3bb3-449e-8374-510eeaecdfee_id_dialog_context(2), solicitar_cita_dialog_context(2), solicitar_cita_dialog_params_matricula(1), __system_counters__(1)
3. **solicitar_cita** {'tipo_dano': 'chapa', 'matricula': '1234 BCD'} → Gracias, Ana. He encontrado tu Seat León. ¿Qué día te viene bien para la cita?  
   contextos: ctx_solicitud_en_curso(10), 65200a1a-3bb3-449e-8374-510eeaecdfee_id_dialog_context(2), solicitar_cita_dialog_context(2), solicitar_cita_dialog_params_fecha(1), solicitar_cita(1), __system_counters__(1)
4. **consulta_documental** {} → No, el peritaje es gratuito para ti. Seguimos con tu cita. ¿Qué día te viene bien para la cita?  
   contextos: ctx_solicitud_en_curso(10)
5. **solicitar_cita** {'franja': 'manana', 'tipo_dano': 'chapa', 'fecha': '2026-11-09T12:00:00+01:00', 'matricula': '1234 BCD'} → Perfecto. ¿En qué ciudad quieres que te vea el perito?  
   contextos: ctx_solicitud_en_curso(10), 65200a1a-3bb3-449e-8374-510eeaecdfee_id_dialog_context(2), solicitar_cita_dialog_context(2), solicitar_cita_dialog_params_ciudad(1), solicitar_cita(1), __system_counters__(1)
6. **solicitar_cita** {'fecha': '2026-11-09T12:00:00+01:00', 'franja': 'manana', 'tipo_dano': 'chapa', 'matricula': '1234 BCD', 'ciudad': 'Gijón'} → Ana, te resumo la solicitud: cita de peritaje para tu Seat León, matrícula 1234 BCD, el lunes 9 de noviembre por la mañana, en Talleres El Musel, Gijón. Motivo: daños de chapa. ¿Confirmo la cita?  
   contextos: ctx_solicitud_en_curso(10), solicitar_cita-followup(2)
7. **solicitar_cita** {'tipo_dano': 'chapa', 'fecha': '2026-11-09T12:00:00+01:00', 'franja': 'tarde', 'matricula': '1234BCD', 'ciudad': 'Gijón'} → Ana, te resumo la solicitud: cita de peritaje para tu Seat León, matrícula 1234 BCD, el lunes 9 de noviembre por la tarde, en Talleres El Musel, Gijón. Motivo: daños de chapa. ¿Confirmo la cita?  
   contextos: ctx_solicitud_en_curso(10), solicitar_cita-followup(2)
8. **solicitar_cita - si** {} → ¡Hecho! Tu cita es la PER-81094. Te atenderá Elena Prieto, del equipo de peritos, el lunes 9 de noviembre a las 18:30 en Talleres El Musel, avenida de la Costa 64, Gijón. Lleva el permiso de circulación, tu DNI y, si lo tienes, el parte amistoso. ¿Te ayudo con algo más?  
   contextos: —
9. **Default Fallback Intent** {} → El todo riesgo sin franquicia cuesta 845,90 € al año.  
   contextos: __system_counters__(1)
10. **consulta_documental** {} → No dispongo de esa información en la documentación de Seguros Alonso. ¿Quieres que te pase con un agente?  
   contextos: __system_counters__(1)
11. **transferir_agente** {} → Claro, te paso con una persona del equipo de atención al cliente. Tu número de ticket es el TCK-789839. Si se corta, puedes llamar al 910 555 222, de lunes a viernes de 9 a 14 y de 16 a 19 horas.  
   contextos: —