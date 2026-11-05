"""Empaqueta el agente Dialogflow ES en ``dist/SegurosAlonso.zip``.

El ZIP tiene ``agent.json``, ``package.json``, ``entities/`` e ``intents/`` en la RAÍZ, que es
lo que espera la consola de Dialogflow ES (Configuración del agente > Export and Import >
"Restore from ZIP" o "Import from ZIP").

Antes de empaquetar valida el export:
  * todos los .json se pueden leer;
  * cada entidad tiene su fichero ``_entries_<idioma>.json`` y las regex compilan;
  * cada ``_usersays_`` tiene su intención, los follow-ups apuntan a un padre que existe y las
    anotaciones usan entidades definidas (o de sistema).

Uso (desde la raíz del repo)::

    .venv\\Scripts\\python.exe scripts\\empaquetar_agente.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ORIGEN = RAIZ / "agente_dialogflow"
DESTINO = RAIZ / "dist" / "SegurosAlonso.zip"


def validar(origen: Path) -> list[str]:
    """Devuelve la lista de errores encontrados (vacía si el export es válido)."""
    errores: list[str] = []
    datos: dict[Path, object] = {}
    for fichero in sorted(origen.rglob("*.json")):
        try:
            datos[fichero] = json.loads(fichero.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errores.append(f"{fichero.relative_to(origen)}: JSON inválido ({exc})")
    if errores:
        return errores

    agente = datos.get(origen / "agent.json")
    if not isinstance(agente, dict):
        return ["Falta agent.json"]
    if not (origen / "package.json").exists():
        errores.append("Falta package.json")
    idioma = agente.get("language", "es")

    # Entidades
    entidades = set()
    for fichero in sorted((origen / "entities").glob("*.json")):
        if "_entries_" in fichero.stem:
            continue
        ent = datos[fichero]
        entidades.add(ent["name"])
        entradas = fichero.with_name(f"{fichero.stem}_entries_{idioma}.json")
        if not entradas.exists():
            errores.append(f"Entidad {ent['name']}: falta {entradas.name}")
            continue
        if ent.get("isRegexp"):
            for entrada in datos[entradas]:
                try:
                    re.compile(entrada["value"])
                except re.error as exc:
                    errores.append(f"Entidad {ent['name']}: regex inválida {entrada['value']!r} ({exc})")

    # Intenciones
    ids, padres = {}, []
    for fichero in sorted((origen / "intents").glob("*.json")):
        if "_usersays_" in fichero.stem:
            nombre = fichero.stem.split("_usersays_")[0]
            if not (fichero.parent / f"{nombre}.json").exists():
                errores.append(f"{fichero.name}: no existe la intención {nombre}")
            for frase in datos[fichero]:
                for parte in frase.get("data", []):
                    meta = parte.get("meta", "")
                    if meta and not meta.startswith("@sys.") and meta[1:] not in entidades:
                        errores.append(f"{fichero.name}: entidad desconocida {meta}")
            continue
        intent = datos[fichero]
        ids[intent["id"]] = intent["name"]
        if intent.get("parentId"):
            padres.append((intent["name"], intent["parentId"]))
        for p in intent.get("responses", [{}])[0].get("parameters", []):
            tipo = p.get("dataType", "")
            if not tipo.startswith("@sys.") and tipo[1:] not in entidades:
                errores.append(f"{intent['name']}: parámetro {p['name']} con entidad desconocida {tipo}")
    for nombre, padre in padres:
        if padre not in ids:
            errores.append(f"{nombre}: parentId {padre} no existe")
    return errores


def empaquetar(origen: Path = ORIGEN, destino: Path = DESTINO) -> Path:
    """Crea el ZIP con los ficheros del agente en la raíz."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fichero in sorted(origen.rglob("*.json")):
            zf.write(fichero, fichero.relative_to(origen).as_posix())
    return destino


def main() -> int:
    errores = validar(ORIGEN)
    if errores:
        print("El agente tiene errores:")
        for e in errores:
            print("  -", e)
        return 1
    zip_creado = empaquetar()
    with zipfile.ZipFile(zip_creado) as zf:
        nombres = zf.namelist()
    n_intents = sum(1 for n in nombres if n.startswith("intents/") and "_usersays_" not in n)
    n_entidades = sum(1 for n in nombres if n.startswith("entities/") and "_entries_" not in n)
    print(f"OK: {zip_creado.relative_to(RAIZ)} ({len(nombres)} ficheros, {n_intents} intenciones, "
          f"{n_entidades} entidades)")
    print("Impórtalo en la consola de Dialogflow ES: Settings > Export and Import > Restore from ZIP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
