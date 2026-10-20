# Modelo de predicción AgroCebada

`predicción.py` estima el rendimiento de cebada (t/ha) del ciclo abril–octubre de 2025 para
las 59 parcelas sin etiqueta. Usa los CSV satelitales del reto y las salidas de `extraccion.py`
(clima y topografía). Si faltan esas salidas, ejecuta primero el extractor automáticamente.

## Ejecución

```bash
.venv/bin/pip install -r requirements_prediccion.txt
.venv/bin/python predicción.py
```

La ejecución completa tarda unos 10 minutos. Opciones útiles:

```bash
# Búsqueda breve de modelos (pruebas rápidas)
.venv/bin/python predicción.py --modo rapido

# Omitir el análisis de las preguntas guía
.venv/bin/python predicción.py --sin-preguntas-guia

# Predecir con el modelo guardado, sin volver a entrenar
.venv/bin/python predicción.py --modelo salidas_prediccion/modelo_final.joblib

# Pruebas automatizadas
.venv/bin/python -m unittest tests_prediccion.py
```

En Windows, usa `.venv\Scripts\python.exe`.

## Estructura del código

`predicción.py` es solo el punto de entrada. El código vive en el paquete `agrocebada/`:

| Módulo | Qué contiene |
| --- | --- |
| `datos.py` | Una fila por parcela: satélite, fenología, historial 2022-2024, clima, topografía y ubicación. |
| `modelos.py` | El modelo geográfico-residual y el catálogo de candidatos comparados. |
| `evaluacion.py` | Métricas, partición 80/20 y validación cruzada. |
| `seleccion.py` | Selección con validación interna, diagnóstico espacial e importancia de variables. |
| `preguntas_guia.py` | Respuestas a las tres preguntas guía de las bases. |
| `intervalos.py` | Intervalos de predicción por zona. |
| `reporte.py` | Figuras y reporte técnico automático. |
| `cli.py` | Línea de comandos que orquesta todo. |
| `utilidades.py`, `constantes.py` | Archivos, huellas del código y nombres compartidos. |

## Protocolo

1. Se construye una fila por parcela con variables que no usan rendimientos: medianas
   mensuales y fenología de cada índice (Sentinel-2, Planet, Landsat), historial Sentinel-2
   2022–2024, clima, topografía, superficie, coordenadas y municipio.
2. Las 138 parcelas con etiqueta se dividen 80/20 (110/28) con semilla 42.
3. Con las 110 se comparan unos 100 candidatos mediante validación cruzada 5×3. Toda
   imputación, escalado y selección de variables ocurre dentro de cada fold.
4. La configuración se congela (`seleccion_congelada.json`) y solo entonces se evalúa en las 28.
5. El mismo algoritmo se reajusta con las 138 para predecir las 59.
6. Con la configuración fija se calculan la validación cruzada sobre 138, los intervalos de
   predicción y las respuestas a las preguntas guía.

## Salidas principales (`salidas_prediccion/`)

| Archivo | Uso |
| --- | --- |
| `predicciones_59.csv` | Entrega: `ID_POLIGONO`, `RENDIMIENTO_T_HA` de las 59 parcelas. |
| `predicciones_59_intervalos.csv` | Mismas predicciones con límites al 80 % y 90 %. |
| `resultados_para_mapa.csv` | Las 197 parcelas, para la aplicación: real, estimado (fuera de fold en las etiquetadas), error, intervalos y producción. |
| `metricas.json` | Prueba 20 %, referencia, bootstrap, CV sobre 138, intervalos y variables del modelo final. |
| `reporte.md` | Resumen técnico con tablas y figuras. |
| `preguntas_guia.json`, `pregunta1_indices.csv`, `pregunta2_clima.csv`, `pregunta3_combinaciones.csv`, `preguntas_guia.png` | Respuestas a las preguntas guía de las bases. |
| `comparacion_cv.csv` | Todos los candidatos con su RMSE de validación cruzada. |
| `importancia_variables.csv` | Permutación por índice o familia de variables. |
| `modelo_final.joblib` | Modelo con las 138, columnas esperadas e intervalos. |
| `manifiesto.json`, `calidad_datos.json`, `particion.csv` | Trazabilidad: versiones, hashes, partición y tratamiento de datos. |
| `registro_asistencia_IA.md` | Declaración del uso de IA y prompts, exigida por las bases. |

## Panel web

`dashboard/` contiene el panel interactivo (mapa, tabla, ejecución del modelo, riesgo, desempeño y
preguntas guía). Se abre con doble clic en `dashboard/index.html`; la versión con fondo satelital
y capas de clima y relieve es `dashboard/panel_mapa_satelital.html`. Instrucciones, archivos y cómo
publicarlo están en [dashboard/LEEME.md](dashboard/LEEME.md). Después de reentrenar:

```bash
.venv/bin/python dashboard/construir_datos.py
.venv/bin/python -m unittest tests_dashboard.py
```

## Usar el modelo desde Python

El modelo contiene una clase propia (`GeoResidualRegressor`). Para cargarlo, la carpeta del
proyecto debe estar en `sys.path`:

```python
import sys, joblib, pandas as pd
sys.path.insert(0, "/ruta/a/AGRO cebada")
paquete = joblib.load("salidas_prediccion/modelo_final.joblib")
X = pd.read_csv("salidas_prediccion/variables_por_parcela.csv", index_col="ID_POLIGONO", encoding="utf-8-sig")
pred = paquete["modelo"].predict(X[paquete["columnas"]])
```

Para mostrar resultados basta `resultados_para_mapa.csv`; no hace falta cargar el modelo.
