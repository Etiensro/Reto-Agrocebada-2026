#!/usr/bin/env python3
"""Punto de entrada del modelo de rendimiento de cebada (toneladas por hectárea).

Ejecución completa:      .venv/bin/python predicción.py
Preparar sin entrenar:   .venv/bin/python predicción.py --solo-preparar
Búsqueda breve:          .venv/bin/python predicción.py --modo rapido
Ver todas las opciones:  .venv/bin/python predicción.py --help

El código está dividido en el paquete agrocebada/ (datos, modelos, selección, evaluación,
preguntas guía, intervalos, reporte y línea de comandos). Este archivo solo lo invoca y
reexporta sus nombres principales para quien los importe desde aquí.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agrocebada.cli import main  # noqa: E402
from agrocebada.constantes import ANIOS_HISTORICOS, ID, METRICAS_FENOLOGICAS, TARGET  # noqa: E402,F401
from agrocebada.datos import metricas_fenologicas, preparar_datos  # noqa: E402,F401
from agrocebada.evaluacion import bootstrap_metricas, dividir_parcelas, evaluar_cv, metricas  # noqa: E402,F401
from agrocebada.intervalos import (intervalos_por_zona, tabla_intervalos, variables_seleccionadas,  # noqa: E402,F401
                                   zonas_intervalo)
from agrocebada.modelos import GeoResidualRegressor, crear_pipeline, generar_candidatos  # noqa: E402,F401
from agrocebada.preguntas_guia import analisis_preguntas_guia  # noqa: E402,F401
from agrocebada.seleccion import diagnostico_espacial, importancia_grupos, seleccionar_modelo  # noqa: E402,F401

__all__ = ["main", "ANIOS_HISTORICOS", "ID", "METRICAS_FENOLOGICAS", "TARGET", "metricas_fenologicas", "preparar_datos",
           "bootstrap_metricas", "dividir_parcelas", "evaluar_cv", "metricas", "intervalos_por_zona", "tabla_intervalos",
           "variables_seleccionadas", "zonas_intervalo", "GeoResidualRegressor", "crear_pipeline", "generar_candidatos",
           "analisis_preguntas_guia", "diagnostico_espacial", "importancia_grupos", "seleccionar_modelo"]

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
