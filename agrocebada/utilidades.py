"""Lectura/escritura de archivos y huellas para la trazabilidad."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def guardar_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def guardar_csv(frame, path, index=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding="utf-8-sig", float_format="%.8f")


def hash_archivo(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_codigo(raiz):
    """Huella del código: el punto de entrada y todos los módulos del paquete."""
    h = hashlib.sha256()
    for ruta in sorted([Path(raiz) / "predicción.py", *(Path(raiz) / "agrocebada").glob("*.py")]):
        h.update(ruta.name.encode("utf-8"))
        h.update(ruta.read_bytes())
    return h.hexdigest()
