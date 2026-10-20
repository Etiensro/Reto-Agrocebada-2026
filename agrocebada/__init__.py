"""Modelo reproducible de rendimiento de cebada, en toneladas por hectárea.

Ejecución: .venv/bin/python predicción.py
Preparar sin entrenar: .venv/bin/python predicción.py --solo-preparar
Búsqueda breve: .venv/bin/python predicción.py --modo rapido

La división 80/20 se hace sobre las 138 parcelas CON etiqueta (110/28).
Selección, imputación, escalado e hiperparámetros usan solamente entrenamiento.
La prueba se evalúa después de congelar la selección. Al final se reajusta el
modelo elegido con las 138 conocidas para estimar las 59 del concurso.
Datos utilizados: abril-octubre 2025 (más historial satelital 2022-2024). Es una
estimación al cierre del ciclo, no un pronóstico anterior a esas observaciones.

Además de los modelos generales, se evalúa un modelo geográfico-residual: la
vecindad en kilómetros captura el efecto de zona y una regresión Ridge corrige
el residuo con superficie y métricas fenológicas elegidas dentro de cada fold.
Después de la prueba se responden las preguntas guía de las bases (índice,
variable climática y combinación de variables) con validación cruzada.

Estructura del paquete:
  constantes      nombres de columnas y parámetros fijos
  utilidades      lectura/escritura de archivos y huella del código
  datos           una fila por parcela: satélite, fenología, clima, topografía y ubicación
  evaluacion      métricas, partición 80/20 y validación cruzada
  modelos         modelo geográfico-residual y catálogo de candidatos
  seleccion       selección con validación interna, diagnóstico espacial, importancia
  preguntas_guia  respuestas a las tres preguntas guía de las bases
  intervalos      intervalos de predicción por zona
  reporte         figuras y reporte técnico automático
  cli             línea de comandos que orquesta todo lo anterior
"""
import os
from pathlib import Path

# Limitar paralelismo evita multiplicar los hilos entre CV y estimadores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "salidas_prediccion" / ".matplotlib"))

try:
    import catboost, joblib, numpy, pandas, scipy, sklearn, threadpoolctl  # noqa: F401,E401
except ImportError as exc:
    raise SystemExit(f"Falta una dependencia: {exc}. Ejecuta .venv/bin/pip install -r requirements_prediccion.txt") from exc
