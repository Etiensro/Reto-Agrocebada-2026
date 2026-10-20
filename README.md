# AgroCebada 2026 


## Descripción

Este repositorio contiene la solución completa del equipo ECOBITS para el Reto AgroCebada 2026, convocado por FIRA. El objetivo fue desarrollar un **modelo predictivo de rendimiento de cebada (t/ha)** para 59 parcelas de Hidalgo, Puebla y Tlaxcala, sustentado exclusivamente en datos de percepción remota, clima y topografía de acceso público, y presentarlo en una **aplicación web interactiva de acceso público**.

**Panel interactivo:** 
https://etiensro.github.io/Reto-Agrocebada-2026/dashboard/panel_mapa_satelital_autonomo.html

---

## Resultados Principales

| Métrica | Valor |
|---|---|
| **R² · Validación cruzada** (138 parcelas, 5×3 pliegues) | **0.69** |
| **RMSE · Validación cruzada** | **0.475 t/ha** |
| **MAE · Prueba independiente** (28 parcelas) | **0.376 t/ha** |
| **MAPE · Prueba independiente** | **11.3 %** |
| Parcelas con error relativo ≤ 20 % (prueba) | 92.9 % |
| Modelo seleccionado | `GeoResidualRegressor` (vecindad k=30 + Ridge α=80) |
| Variables del modelo final | Ubicación, superficie y verdor basal EVI/NDVI 2022–2024 |

---

## Estructura del Repositorio

```
AgroCebada_Reto2026_Paquete_Equipo/
│
├──  dashboard/                          # Aplicación web interactiva
│   ├── panel_mapa_satelital_autonomo.html # ← LA APP (archivo único, sin servidor)
│   ├── index.html                         # Versión modular con mapa simple
│   ├── app.js / estilos.css / datos.js    # Código y datos del panel
│   ├── modelo.js                          # GeoResidualRegressor en JavaScript
│   └── LEEME.md                           # Guía del panel
│
├──  agrocebada/                         # Paquete Python del modelo
│   ├── datos.py                           # Construcción de variables
│   ├── modelos.py                         # GeoResidualRegressor y candidatos
│   ├── evaluacion.py                      # Métricas y validación cruzada
│   ├── seleccion.py                       # Selección y diagnóstico espacial
│   ├── preguntas_guia.py                  # Respuestas a las 3 preguntas guía
│   ├── intervalos.py                      # Intervalos conformales por municipio
│   ├── reporte.py                         # Figuras y reporte técnico
│   └── cli.py                             # Orquestador de línea de comandos
│
├──  salidas_prediccion/                 # Salidas del modelo (pre-generadas)
│   ├── predicciones_59.csv                # ← ENTREGA: estimaciones de las 59 parcelas
│   ├── predicciones_59_intervalos.csv     # Con intervalos conformales 80% y 90%
│   ├── resultados_para_mapa.csv           # 197 parcelas para el panel
│   ├── reporte.md                         # Resumen técnico con tablas y figuras
│   ├── metricas.json                      # Todas las métricas de evaluación
│   ├── precision_prueba.png               # Gráfica real vs. predicho
│   ├── preguntas_guia.png                 # Respuestas a las 3 preguntas guía
│   ├── modelo_final.joblib                # Modelo entrenado con las 138 parcelas
│   ├── manifiesto.json                    # Trazabilidad: versiones, hashes, partición
│   ├── seleccion_congelada.json           # Hiperparámetros congelados pre-prueba
│
├──  salidas_extractor/                 # Salidas de extracción geoespacial
│   ├── clima_mensual.csv                  # Clima por parcela y mes (CHIRPS/CHIRTS)
│   ├── topografia_parcelas.csv            # Elevación y pendiente (CEM 4.0 INEGI)
│   └── parcelas_datos.csv                 # Una columna por capa extraída
│
├── predicción.py                          # Punto de entrada: entrena y predice
├── extraccion.py                          # Extrae clima y topografía de los TIFF
├── tests_prediccion.py                    # 29 pruebas del modelo
├── tests_dashboard.py                     # 15 pruebas del panel web
├── requirements_prediccion.txt            # Dependencias Python (Python 3.12)
├── LEEME_prediccion.md                    # Guía del modelo
├── LEEME_extractor.md                     # Guía del extractor
└── README.md                              # Este archivo
```

##  Reproducir el Modelo desde Cero

### Prerrequisitos

- Python 3.10 o posterior
- El dataset oficial del reto descomprimido: `DATASET_RETO_AGRO_2026/`

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/[usuario]/[repositorio].git
cd [repositorio]

# 2. Crear entorno virtual
python -m venv .venv

# Windows:
.venv\Scripts\pip install -r requirements_prediccion.txt

# macOS / Linux:
.venv/bin/pip install -r requirements_prediccion.txt
```

### Ejecución Completa (~10 minutos)

```bash
# Coloca DATASET_RETO_AGRO_2026/ junto a predicción.py, luego:

# Windows:
.venv\Scripts\python predicción.py

# macOS / Linux:
.venv/bin/python predicción.py
```

Opciones útiles:

```bash
# Búsqueda rápida de modelos (pruebas, ~2 min)
.venv/bin/python predicción.py --modo rapido

# Predecir con el modelo ya guardado (sin reentrenar, ~30 seg)
.venv/bin/python predicción.py --modelo salidas_prediccion/modelo_final.joblib

# Omitir el análisis de preguntas guía
.venv/bin/python predicción.py --sin-preguntas-guia
```

### Pruebas Automáticas

```bash
# Modelo (29 tests)
.venv/bin/python -m unittest tests_prediccion.py

# Panel web (15 tests)
.venv/bin/python -m unittest tests_dashboard.py

# Todo junto
.venv/bin/python -m unittest tests_prediccion.py tests_dashboard.py
```
