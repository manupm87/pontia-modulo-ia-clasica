"""Dialogflow ES en local (emulador) y cliente común para el backend local o la API de Google.

Uso típico::

    from dialogflow_local.cliente import crear_cliente
    cliente = crear_cliente()            # DIALOGFLOW_BACKEND=local|google
    turno = cliente.evento("sesion-1", "WELCOME")
    turno = cliente.texto("sesion-1", "quiero cita con el perito")
"""
