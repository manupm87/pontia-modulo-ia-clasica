"""
Lógica de negocio de Seguros Alonso: pólizas simuladas y reglas de la cita de peritaje.

En un sistema real esto sería una consulta a la base de datos de pólizas y a la agenda
de peritos. Para el prototipo bastan diccionarios: lo importante es que la respuesta
la decide nuestro código (validaciones, reglas, datos de la póliza) y no una plantilla
fija de Dialogflow.

Todo lo de este módulo es puro (sin red ni Dialogflow), así que se prueba con pytest.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Expresiones regulares (requisito: validar al menos un parámetro con regex)
# ---------------------------------------------------------------------------

# Matrícula española actual (desde 2000): 4 dígitos + 3 consonantes.
# Sin vocales (para evitar palabras malsonantes) ni Ñ ni Q (se confunden con N y O).
# Ojo: la clase [B-DF-HJ-NP-TV-Z] del ejemplo de clase deja pasar la Q (P-T incluye Q);
# la entidad de Dialogflow es permisiva y aquí aplicamos la regla exacta.
PATRON_MATRICULA = re.compile(r"^(\d{4})([BCDFGHJKLMNPRSTVWXYZ]{3})$")

# Fecha que manda Dialogflow en @sys.date: "2026-11-09T12:00:00+01:00".
# Aceptamos también "2026-11-09" (la manda el emulador o un evento escrito a mano).
PATRON_FECHA_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")

# ---------------------------------------------------------------------------
# Datos simulados
# ---------------------------------------------------------------------------

POLIZAS: dict[str, dict] = {
    "1234BCD": {
        "titular": "Ana Ruiz",
        "vehiculo": "Seat León",
        "poliza": "SA-104233",
        "modalidad": "todo riesgo con franquicia",
        "franquicia": 300,
        "danos_propios": True,
        "coberturas_sin_danos_propios": [],
    },
    "5678FGH": {
        "titular": "Luis Gómez",
        "vehiculo": "Volkswagen Golf",
        "poliza": "SA-208871",
        "modalidad": "terceros ampliado",
        "franquicia": 0,
        "danos_propios": False,
        # Terceros ampliado sí cubre lunas, incendio y robo: para esos daños sí hay peritaje
        "coberturas_sin_danos_propios": ["lunas"],
    },
    "9012JKL": {
        "titular": "María López",
        "vehiculo": "Toyota Corolla",
        "poliza": "SA-311540",
        "modalidad": "todo riesgo sin franquicia",
        "franquicia": 0,
        "danos_propios": True,
        "coberturas_sin_danos_propios": [],
    },
}

# Talleres concertados donde puede citarse el perito (clave normalizada sin tildes)
TALLERES: dict[str, dict] = {
    "madrid": {"ciudad": "Madrid", "taller": "Talleres Castellana", "direccion": "calle de Bravo Murillo 210"},
    "barcelona": {"ciudad": "Barcelona", "taller": "Autoservei Diagonal", "direccion": "avinguda Diagonal 455"},
    "valencia": {"ciudad": "Valencia", "taller": "Taller Turia", "direccion": "avenida del Puerto 88"},
    "sevilla": {"ciudad": "Sevilla", "taller": "Talleres Giralda", "direccion": "avenida de la Buhaira 15"},
    "bilbao": {"ciudad": "Bilbao", "taller": "Nervión Motor", "direccion": "calle Autonomía 42"},
    "zaragoza": {"ciudad": "Zaragoza", "taller": "Ebro Car", "direccion": "avenida de Cataluña 120"},
    "malaga": {"ciudad": "Málaga", "taller": "Costa Motor", "direccion": "avenida de Andalucía 30"},
    "gijon": {"ciudad": "Gijón", "taller": "Talleres El Musel", "direccion": "avenida de la Costa 64"},
    "oviedo": {"ciudad": "Oviedo", "taller": "Naranco Automoción", "direccion": "calle Uría 25"},
    "a coruna": {"ciudad": "A Coruña", "taller": "Talleres Orzán", "direccion": "ronda de Outeiro 101"},
}

FRANJAS = {
    "manana": {"nombre": "por la mañana", "horario": "entre las 9 y las 14 horas", "hora_inicio": 9},
    "tarde": {"nombre": "por la tarde", "horario": "entre las 16 y las 19 horas", "hora_inicio": 16},
}

TIPOS_DANO = {
    "chapa": "daños de chapa",
    "lunas": "rotura de lunas",
    "mecanica": "daños mecánicos",
    "granizo": "daños por granizo",
}

PERITOS = ["Javier Ortega", "Lucía Fernández", "Marta Ibáñez", "Pablo Sanz", "Elena Prieto"]

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

DIAS_MAXIMOS_ANTELACION = 30

# "Base de datos" de citas creadas durante la ejecución
CITAS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Normalización y validación de cada parámetro
# ---------------------------------------------------------------------------

def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def normalizar_matricula(valor: str | None) -> str:
    """'1234 bcd' / '1234-BCD' -> '1234BCD'."""
    return re.sub(r"[\s\-.]", "", str(valor or "")).upper()


def matricula_legible(matricula: str) -> str:
    """'1234BCD' -> '1234 BCD' (para mostrar)."""
    m = PATRON_MATRICULA.match(matricula)
    return f"{m.group(1)} {m.group(2)}" if m else matricula


def validar_matricula(valor: str | None) -> tuple[bool, str]:
    """Devuelve (es_valida, matricula_normalizada) aplicando la regex oficial."""
    normalizada = normalizar_matricula(valor)
    return bool(PATRON_MATRICULA.match(normalizada)), normalizada


def parsear_fecha(valor) -> date | None:
    """Convierte el valor de @sys.date (str ISO, o dict con date_time) en date."""
    if not valor:
        return None
    if isinstance(valor, dict):  # algunas variantes de @sys.date-time
        valor = valor.get("date_time") or valor.get("startDate") or ""
    m = PATRON_FECHA_ISO.match(str(valor))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def fecha_en_texto(fecha: date) -> str:
    """date(2026, 11, 9) -> 'el lunes 9 de noviembre'."""
    return f"el {DIAS_SEMANA[fecha.weekday()]} {fecha.day} de {MESES[fecha.month - 1]}"


def siguiente_dia_laborable(fecha: date) -> date:
    while fecha.weekday() >= 5:
        fecha += timedelta(days=1)
    return fecha


def normalizar_franja(valor: str | None) -> str | None:
    v = sin_tildes(str(valor or "")).lower().strip()
    if not v:
        return None
    if "tarde" in v:
        return "tarde"
    if "manana" in v or "temprano" in v or "primera" in v:
        return "manana"
    return None


def normalizar_ciudad(valor) -> str:
    if isinstance(valor, dict):  # @sys.location devuelve un objeto
        valor = valor.get("city") or valor.get("admin-area") or ""
    return sin_tildes(str(valor or "")).lower().strip()


def normalizar_tipo_dano(valor: str | None) -> str | None:
    v = sin_tildes(str(valor or "")).lower().strip()
    return v if v in TIPOS_DANO else None


# ---------------------------------------------------------------------------
# Evaluación de la solicitud completa
# ---------------------------------------------------------------------------

@dataclass
class Problema:
    campo: str          # parámetro que hay que corregir (matricula, fecha, franja, ciudad, tipo_dano)
    mensaje: str        # explicación para el usuario (texto plano, apto para voz)


@dataclass
class Evaluacion:
    matricula: str = ""
    poliza: dict | None = None
    fecha: date | None = None
    franja: str | None = None
    ciudad: str = ""
    taller: dict | None = None
    tipo_dano: str | None = None
    problemas: list[Problema] = field(default_factory=list)

    @property
    def valida(self) -> bool:
        return not self.problemas


def evaluar_solicitud(parametros: dict, hoy: date | None = None) -> Evaluacion:
    """
    Aplica todas las reglas de negocio sobre los parámetros recogidos.

    El orden de las comprobaciones es el orden en que se lo diríamos a una persona:
    primero quién eres (matrícula -> póliza), luego si está cubierto, y al final la agenda.
    """
    hoy = hoy or date.today()
    ev = Evaluacion()

    # 1) Matrícula: regex + existencia de póliza
    valida, ev.matricula = validar_matricula(parametros.get("matricula"))
    if not valida:
        ev.problemas.append(Problema(
            "matricula",
            f"La matrícula {ev.matricula or 'indicada'} no tiene un formato válido. "
            "Debe ser de cuatro números y tres letras, por ejemplo 1234 BCD.",
        ))
    else:
        ev.poliza = POLIZAS.get(ev.matricula)
        if ev.poliza is None:
            ev.problemas.append(Problema(
                "matricula",
                f"No encuentro ninguna póliza con la matrícula {matricula_legible(ev.matricula)}. "
                "Puede que la haya entendido mal.",
            ))

    # 2) Cobertura: el peritaje de daños propios solo entra en todo riesgo
    ev.tipo_dano = normalizar_tipo_dano(parametros.get("tipo_dano"))
    if ev.poliza and not ev.poliza["danos_propios"]:
        cubiertos = ev.poliza["coberturas_sin_danos_propios"]
        if ev.tipo_dano not in cubiertos:
            ev.problemas.append(Problema(
                "tipo_dano",
                f"{ev.poliza['titular']}, tu póliza es de {ev.poliza['modalidad']} y no cubre daños "
                "propios, así que solo puedo citar al perito si se trata de una rotura de lunas.",
            ))

    # 3) Fecha: futura, laborable y con antelación máxima
    ev.fecha = parsear_fecha(parametros.get("fecha"))
    if ev.fecha is None:
        ev.problemas.append(Problema("fecha", "No he entendido la fecha de la cita."))
    elif ev.fecha <= hoy:
        sugerida = siguiente_dia_laborable(hoy + timedelta(days=1))
        ev.problemas.append(Problema(
            "fecha",
            f"La cita tiene que ser a partir de mañana. El primer hueco es {fecha_en_texto(sugerida)}.",
        ))
    elif ev.fecha.weekday() >= 5:
        sugerida = siguiente_dia_laborable(ev.fecha)
        ev.problemas.append(Problema(
            "fecha",
            f"Los peritos no trabajan el fin de semana. ¿Te viene bien {fecha_en_texto(sugerida)}?",
        ))
    elif (ev.fecha - hoy).days > DIAS_MAXIMOS_ANTELACION:
        ev.problemas.append(Problema(
            "fecha",
            f"Solo puedo dar cita con un máximo de {DIAS_MAXIMOS_ANTELACION} días de antelación.",
        ))

    # 4) Franja horaria
    ev.franja = normalizar_franja(parametros.get("franja"))
    if ev.franja is None:
        ev.problemas.append(Problema("franja", "No he entendido si prefieres mañana o tarde."))

    # 5) Ciudad con taller concertado
    clave_ciudad = normalizar_ciudad(parametros.get("ciudad"))
    ev.taller = TALLERES.get(clave_ciudad)
    ev.ciudad = ev.taller["ciudad"] if ev.taller else str(parametros.get("ciudad") or "")
    if ev.taller is None:
        ciudades = ", ".join(t["ciudad"] for t in list(TALLERES.values())[:-1])
        ev.problemas.append(Problema(
            "ciudad",
            f"No tenemos taller concertado en {ev.ciudad or 'esa ciudad'}. "
            f"Puedo darte cita en {ciudades} o A Coruña.",
        ))

    return ev


def _determinista(*partes: str) -> int:
    """Número estable a partir de los datos (misma solicitud -> mismo código)."""
    return int(hashlib.sha1("|".join(partes).encode()).hexdigest(), 16)


def crear_cita(ev: Evaluacion) -> dict:
    """Registra la cita y devuelve sus datos. Requiere una evaluación válida."""
    if not ev.valida:
        raise ValueError("No se puede crear una cita con problemas pendientes")
    assert ev.fecha and ev.franja and ev.poliza and ev.taller  # garantizado por ev.valida
    semilla = _determinista(ev.matricula, ev.fecha.isoformat(), ev.franja, ev.ciudad)
    codigo = f"PER-{semilla % 100000:05d}"
    franja = FRANJAS[ev.franja]
    cita = {
        "codigo": codigo,
        "matricula": ev.matricula,
        "poliza": ev.poliza["poliza"],
        "titular": ev.poliza["titular"],
        "vehiculo": ev.poliza["vehiculo"],
        "fecha": ev.fecha.isoformat(),
        "fecha_texto": fecha_en_texto(ev.fecha),
        "franja": ev.franja,
        "hora": f"{franja['hora_inicio'] + semilla % 4}:{'00' if semilla % 2 else '30'}",
        "ciudad": ev.ciudad,
        "taller": ev.taller["taller"],
        "direccion": ev.taller["direccion"],
        "perito": PERITOS[semilla % len(PERITOS)],
        "tipo_dano": ev.tipo_dano,
        "creada": datetime.now().isoformat(timespec="seconds"),
    }
    CITAS[codigo] = cita
    return cita


def crear_ticket_agente(datos: dict) -> str:
    """Código de ticket para la transferencia a un agente humano."""
    return f"TCK-{_determinista(str(sorted(datos.items())), datetime.now().isoformat()) % 1000000:06d}"
