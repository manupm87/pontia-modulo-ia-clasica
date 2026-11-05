"""
Guion de la demo guiada: una conversación completa que recorre todos los requisitos.

Cada paso es lo que "dice" el usuario. En la demo se sintetiza con una voz distinta a la del
bot y se pasa por el STT, así que la conversación es de voz de extremo a extremo.
"""

GUION: list[dict] = [
    {"evento": "WELCOME", "nota": "Bienvenida (evento WELCOME)"},
    {"dice": "Hola, quiero pedir cita con un perito para que revise un golpe en mi coche.",
     "nota": "Intención solicitar_cita: arranca el flujo con contexto"},
    {"dice": "La matrícula es 1234 BCD.",
     "nota": "Entidad regex @matricula; el webhook la valida y personaliza la siguiente pregunta"},
    {"dice": "¿Tengo que pagar algo por el peritaje?",
     "nota": "Interrupción en mitad de la recogida de datos: responde con los documentos y retoma"},
    {"dice": "El lunes por la mañana.",
     "nota": "@sys.date + entidad propia @franja en una sola frase"},
    {"dice": "En Gijón.",
     "nota": "@sys.geo-city: con todos los datos el webhook valida reglas y pide confirmación"},
    {"dice": "Mejor por la tarde.",
     "nota": "Corrección a mitad de flujo: corregir_dato relanza el evento SOLICITAR_CITA"},
    {"dice": "Sí, confírmala.",
     "nota": "Follow-up de confirmación sí: el webhook crea la cita"},
    {"dice": "¿Cuánto cuesta al año el todo riesgo sin franquicia?",
     "nota": "Consulta documental sobre la tabla escaneada (RAG con ChromaDB)"},
    {"dice": "¿Me cubre el seguro de vida de mi mujer?",
     "nota": "Control de alucinaciones: no está en los documentos y lo dice"},
    {"dice": "Vale, quiero hablar con una persona.",
     "nota": "Transferencia a agente humano con ticket"},
]
