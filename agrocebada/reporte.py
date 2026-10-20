"""Figuras y texto del reporte técnico automático."""
from __future__ import annotations

import numpy as np
import pandas as pd


def figuras_prueba(y, pred, baseline, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4.5), constrained_layout=True)
    lo, hi = min(min(y),min(pred))-.2, max(max(y),max(pred))+.2
    axes[0].scatter(y, pred, c="#227f89", s=45, edgecolors="white", linewidth=.6)
    axes[0].plot([lo,hi],[lo,hi], color="#a14e36", linestyle="--", label="Predicción perfecta")
    axes[0].set(xlabel="Rendimiento real (t/ha)", ylabel="Predicción (t/ha)", title=f"Prueba independiente: {len(y)} parcelas", xlim=(lo,hi), ylim=(lo,hi))
    axes[0].legend(frameon=False)
    axes[1].scatter(pred, np.asarray(pred)-np.asarray(y), c="#227f89", s=45)
    axes[1].axhline(0, color="#a14e36", linestyle="--")
    axes[1].set(xlabel="Predicción (t/ha)", ylabel="Predicción − real (t/ha)", title="Errores en parcelas reservadas")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(salida / "precision_prueba.png", dpi=170)
    plt.close(fig)


def figura_preguntas(guia, salida):
    """Tres paneles, uno por pregunta guía; una sola serie por panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, p1, p2, p3 = guia
    azul, gris, tinta, tinta2 = "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True,
                             gridspec_kw={"height_ratios": [10, 5, 10]})
    a = p1.head(10).iloc[::-1]
    valores = a["APORTE_SOBRE_UBICACION_T_HA"].to_numpy()
    claro = a["APORTE_CLARO"].astype(bool).to_numpy()
    axes[0].barh(a["VARIABLE"], valores, color=np.where(claro, azul, gris), height=.55)
    axes[0].errorbar(valores, np.arange(len(a)), xerr=2 * a["APORTE_EE_T_HA"].to_numpy(), fmt="none",
                     ecolor=tinta2, elinewidth=1, capsize=3)
    axes[0].axvline(0, color=tinta2, linewidth=.8)
    axes[0].set_title("1. ¿Qué índice es más importante?", loc="left", color=tinta, fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Reducción del RMSE sobre ubicación + superficie (t/ha), barras: ±2 errores estándar\n"
                       "azul: aporte claro; gris: no distinguible de cero", color=tinta2)
    b = p2.iloc[::-1]
    axes[1].barh(b["VARIABLE"], b["R2_SOLO"].clip(lower=0), color=azul, height=.55)
    for y_pos, v in enumerate(b["R2_SOLO"]):
        axes[1].text(max(v, 0) + .01, y_pos, f"R² {v:.2f}", va="center", color=tinta2, fontsize=9)
    axes[1].set_xlim(0, max(.1, float(b["R2_SOLO"].max()) * 1.25))
    axes[1].set_title("2. ¿Qué variable climática tiene mayor peso?", loc="left", color=tinta, fontsize=12, fontweight="bold")
    axes[1].set_xlabel("R² de validación cruzada con solo esa variable (sin ubicación)", color=tinta2)
    c = p3.head(10).iloc[::-1]
    colores = np.where(c["EMPATE_CON_MEJOR"].astype(bool), azul, gris)
    axes[2].scatter(c["RMSE_MEJOR_T_HA"], c["COMBINACION"], s=64, color=colores, edgecolors="white", linewidths=1.5, zorder=3)
    for y_pos, v in enumerate(c["RMSE_MEJOR_T_HA"]):
        axes[2].annotate(f"{v:.3f}", (v, y_pos), xytext=(8, 0), textcoords="offset points",
                         va="center", color=tinta2, fontsize=9)
    rango = float(c["RMSE_MEJOR_T_HA"].max() - c["RMSE_MEJOR_T_HA"].min()) or .01
    axes[2].set_xlim(float(c["RMSE_MEJOR_T_HA"].min()) - .1 * rango, float(c["RMSE_MEJOR_T_HA"].max()) + .25 * rango)
    axes[2].set_title("3. ¿Qué combinación de variables funciona mejor?", loc="left", color=tinta, fontsize=12, fontweight="bold")
    axes[2].set_xlabel("RMSE de validación cruzada (t/ha, menor es mejor; eje recortado)\n"
                       "azul: empatada con la mejor (diferencia < 2 errores estándar)", color=tinta2)
    for ax in axes:
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(gris)
        ax.tick_params(colors=tinta2, labelsize=9)
        for label in ax.get_yticklabels():
            label.set_color(tinta)
    fig.savefig(salida / "preguntas_guia.png", dpi=170)
    plt.close(fig)


def _tabla_md(frame, columnas, formatos):
    filas = ["| " + " | ".join(columnas.values()) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(columnas))) + "|"]
    for _, fila in frame.iterrows():
        celdas = []
        for c in columnas:
            v = fila[c]
            if isinstance(v, (bool, np.bool_)):
                celdas.append("sí" if v else "no")
            else:
                celdas.append(formatos.get(c, "{}").format(v) if pd.notna(v) else "—")
        filas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def seccion_preguntas(guia):
    if guia is None:
        return "Análisis omitido con --sin-preguntas-guia.\n"
    resumen, p1, p2, p3 = guia
    ref = resumen["referencia_ubicacion_superficie"]
    i1, i1s = resumen["pregunta1_indice_mayor_aporte"], resumen["pregunta1_indice_mayor_poder_solo"]
    i1h, c2, c3 = (resumen["pregunta1_mejor_familia_incluyendo_historial"],
                   resumen["pregunta2_variable_climatica_mayor_peso"], resumen["pregunta3_mejor_combinacion"])
    empatadas = resumen["pregunta3_empatadas_con_la_mejor"] or [c3["COMBINACION"]]
    bloques = sorted({b for c in p3["COMBINACION"] for b in c.split(" + ")})
    conteo = {b: sum(b in c.split(" + ") for c in empatadas) for b in bloques}
    presencia = ", ".join(f"{b} {n}/{len(empatadas)}" for b, n in sorted(conteo.items(), key=lambda t: -t[1]))
    f = {"RMSE_SOLO_T_HA": "{:.3f}", "R2_SOLO": "{:.3f}", "APORTE_SOBRE_UBICACION_T_HA": "{:+.4f}",
         "R_INTRA_MUNICIPIO": "{:+.2f}", "RMSE_MEJOR_T_HA": "{:.3f}", "RMSE_RIDGE_T_HA": "{:.3f}",
         "RMSE_EXTRATREES_T_HA": "{:.3f}", "APORTE_EE_T_HA": "{:.4f}"}
    return f"""Validación cruzada 5×4 sobre las 138 parcelas conocidas, ejecutada después de la prueba;
no modifica el modelo. Cada índice o variable se mide de dos formas:
**solo** (Ridge con sus métricas, sin ubicación) y **aporte sobre ubicación**: cuánto baja el RMSE
al añadir su mejor métrica, elegida dentro de cada fold, a la vecindad geográfica + superficie
(referencia: RMSE {ref['RMSE_t_ha']:.3f} t/ha, R² {ref['R2']:.3f}).

![Respuestas a las preguntas guía](preguntas_guia.png)

### 1. ¿Qué índice es más importante?

- Mayor poder por sí solo: **{i1s['VARIABLE']}** (RMSE {i1s['RMSE_SOLO_T_HA']:.3f} t/ha, R² {i1s['R2_SOLO']:.3f}).
  Sin ubicación, los índices distinguen sobre todo zonas productivas.
- Mayor aporte sobre la ubicación, incluyendo el historial: **{i1h['VARIABLE']}**
  ({i1h['APORTE_SOBRE_UBICACION_T_HA']:+.4f} ± {2 * i1h['APORTE_EE_T_HA']:.4f} t/ha).
- Mayor aporte entre los índices de la temporada 2025: **{i1['VARIABLE']}**
  ({i1['APORTE_SOBRE_UBICACION_T_HA']:+.4f} ± {2 * i1['APORTE_EE_T_HA']:.4f} t/ha);
  métrica más asociada dentro del municipio: `{i1['METRICA_MAS_ASOCIADA_INTRA_MUNICIPIO']}`.
- Aportes mayores que dos errores estándar: {", ".join(resumen["pregunta1_aportes_claros"]) or "ninguno"}.
  Los demás no se distinguen de cero con 138 parcelas.

{_tabla_md(p1.head(10), {"VARIABLE": "Índice", "RMSE_SOLO_T_HA": "RMSE solo", "R2_SOLO": "R² solo",
                         "APORTE_SOBRE_UBICACION_T_HA": "Aporte (t/ha)", "APORTE_EE_T_HA": "Error estándar",
                         "METRICA_MAS_ASOCIADA_INTRA_MUNICIPIO": "Métrica más asociada",
                         "R_INTRA_MUNICIPIO": "r intra-municipio"}, f)}

Tabla completa: pregunta1_indices.csv.

### 2. ¿Qué variable climática tiene mayor peso?

**{c2['VARIABLE']}**: R² {c2['R2_SOLO']:.3f} por sí sola (RMSE {c2['RMSE_SOLO_T_HA']:.3f} t/ha).
Su aporte una vez conocida la ubicación es {c2['APORTE_SOBRE_UBICACION_T_HA']:+.4f} t/ha: el clima
(celdas de ~5 km) distingue zonas, pero casi no varía entre parcelas vecinas.

{_tabla_md(p2, {"VARIABLE": "Variable", "RMSE_SOLO_T_HA": "RMSE solo", "R2_SOLO": "R² solo",
                "APORTE_SOBRE_UBICACION_T_HA": "Aporte (t/ha)"}, f)}

### 3. ¿Qué combinación de variables funciona mejor?

Se probaron las {len(p3)} combinaciones de bloques con Ridge y ExtraTrees.
La de menor RMSE fue **{c3['COMBINACION']}** ({c3['RMSE_MEJOR_T_HA']:.3f} t/ha, {c3['ALGORITMO_MEJOR'].title()}).
{len(empatadas)} combinaciones están estadísticamente empatadas con ella (diferencia pareada
< 2 errores estándar); la más simple con menor RMSE es
**{min(empatadas, key=lambda c: (c.count("+"), empatadas.index(c)))}**.
Presencia de cada bloque entre las empatadas: {presencia}.
El modelo geográfico-residual, que combina vecindad, superficie y métricas fenológicas
elegidas en cada fold, se compara con estos resultados en comparacion_cv.csv.

{_tabla_md(p3.head(10), {"COMBINACION": "Combinación", "RMSE_RIDGE_T_HA": "Ridge",
                         "RMSE_EXTRATREES_T_HA": "ExtraTrees", "RMSE_MEJOR_T_HA": "Mejor",
                         "EMPATE_CON_MEJOR": "Empate"}, f)}

Tabla completa: pregunta3_combinaciones.csv. Con 138 parcelas, las diferencias menores a dos
errores estándar (columnas EE_* de los CSV) no deben interpretarse como superioridad.
"""


def escribir_reporte(salida, resultado, seleccion, calidad, importance, guia=None):
    m, b = resultado["prueba"], resultado["baseline_prueba"]
    spatial = resultado["diagnostico_espacial_entrenamiento"]
    cv138 = resultado.get("validacion_cruzada_138", {})
    rmse_espacial = (f"{spatial['RMSE_t_ha']:.4f}" if spatial.get("RMSE_t_ha") is not None
                     else "no disponible")
    intervalos = resultado.get("intervalos_prediccion_t_ha", {})
    tabla_int = "\n".join(f"| {z} | {v['n']} | ±{v['80']:.2f} | ±{v['90']:.2f} |" for z, v in intervalos.items())
    elegidas = resultado.get("variables_modelo_final", {})
    texto_elegidas = "\n".join(
        f"- {k}: " + ", ".join(f"`{v}` ({c:+.3f})" for v, c in d["coeficientes_estandarizados"].items())
        for k, d in elegidas.items()) or "- El modelo elegido no contiene componente geográfico-residual."
    text = f"""# Predicción del rendimiento de cebada

Se estimó el rendimiento, en toneladas por hectárea, del ciclo abril–octubre de 2025.
El programa integra las geometrías y la extracción existente con índices de Sentinel-2,
Landsat y Planet. Usa observaciones con nubosidad máxima {calidad.get('cloud_max',20)}%.
Las estadísticas topográficas de cobertura insuficiente se tratan como ausentes.

## Variables construidas

- Medianas mensuales de cada índice (abril–octubre) y sus resúmenes, como en la versión previa.
- **Fenología por parcela** para todos los índices de los tres sensores: curva suavizada
  (mediana móvil de 3 fechas) con pico, percentil 90, nivel base de abril–mayo, amplitud,
  día del pico, inicio y fin de temporada (50 % de la amplitud), duración e integrales.
- **Historial Sentinel-2 2022–2024** (NDVI y EVI: pico, base, integral y su media):
  describe la productividad habitual de cada parcela antes del ciclo evaluado.
- Coordenadas en km, indicadores de municipio (≥5 parcelas), superficie, clima y topografía.
  Ninguna de estas variables usa rendimientos; se calculan igual para las 197 parcelas.

## Modelo geográfico-residual

La exploración descriptiva mostró que el rendimiento depende sobre todo de la zona (municipio)
y que parcelas a menos de 0.5 km rinden parecido; el clima (~5 km) apenas cambia dentro de una
zona. Esa exploración (medias por municipio, semivariograma) usó las 138 parcelas etiquetadas;
las comparaciones de variables y modelos que guiaron este diseño usaron solo las 110 de
desarrollo. La prueba de 28 ya había sido consultada por la versión anterior del programa, así
que no es una prueba completamente nueva. Por eso se añadió un candidato que:
1) estima el efecto de zona con los vecinos más cercanos en km (ponderados por distancia);
2) corrige el residuo fuera de fold con Ridge usando la superficie y las métricas
   fenológicas más correlacionadas con ese residuo, elegidas dentro de cada fold.
Compite en igualdad con el resto de candidatos; la validación cruzada decide.

Variables del modelo final (coeficiente estandarizado del residuo):

{texto_elegidas}

## Diseño de evaluación

- {resultado['n_entrenamiento']} parcelas para desarrollo y {resultado['n_prueba']} para prueba independiente.
- {seleccion['folds']} folds × {seleccion['repeticiones']} repeticiones exclusivamente en desarrollo.
- Imputación, eliminación de constantes, selección de variables y escalado se ajustan dentro de cada fold.
- Se selecciona por RMSE, comparando modelos y promedios de familias distintas.
- La configuración se guarda en seleccion_congelada.json antes de consultar la prueba.
- La media del rendimiento de entrenamiento es la referencia mínima.
- Después de la prueba, el mismo algoritmo se reajusta con las 138 parcelas conocidas para las 59 ocultas.
- El modelo reajustado no tiene una segunda evaluación independiente; las métricas corresponden al modelo entrenado con el 80%.

Modelo seleccionado: **{seleccion['configuracion']['nombre']}**.
RMSE de selección interna: **{seleccion['cv_rmse']:.4f} t/ha**.
Esta cifra puede ser optimista por selección entre candidatos; la prueba independiente es la evaluación principal.

## Resultado de la prueba reservada

| Métrica | Modelo elegido | Media de entrenamiento |
|---|---:|---:|
| RMSE (t/ha) | {m['RMSE_t_ha']:.4f} | {b['RMSE_t_ha']:.4f} |
| MAE (t/ha) | {m['MAE_t_ha']:.4f} | {b['MAE_t_ha']:.4f} |
| R² | {m['R2']:.4f} | {b['R2']:.4f} |
| Error porcentual absoluto medio (MAPE) | {m['MAPE_pct']:.2f}% | {b['MAPE_pct']:.2f}% |
| Predicciones con error relativo ≤10% | {m['dentro_10_pct']:.2f}% | {b['dentro_10_pct']:.2f}% |
| Predicciones con error relativo ≤20% | {m['dentro_20_pct']:.2f}% | {b['dentro_20_pct']:.2f}% |

Validación cruzada complementaria (configuración fija, 5×3, 138 parcelas):
RMSE {cv138.get('RMSE_t_ha', float('nan')):.4f} t/ha, MAE {cv138.get('MAE_t_ha', float('nan')):.4f} t/ha,
R² {cv138.get('R2', float('nan')):.4f}. Con 28 parcelas la prueba tiene un intervalo amplio; esta cifra
es más estable, aunque algo optimista porque incluye las parcelas que eligieron la configuración.

MAE expresa el error absoluto promedio por hectárea. RMSE penaliza más los errores grandes.
R² compara con la variabilidad real; no es un porcentaje de exactitud. MAPE es error relativo,
no debe transformarse en una supuesta «precisión» mediante 100 − MAPE.

![Valores reales y predichos](precision_prueba.png)

Los rangos bootstrap de MAE/RMSE en metricas.json describen variabilidad al remuestrear
estas parcelas; no son intervalos de predicción ni corrigen dependencia espacial.

## Dependencia espacial

La partición aleatoria mide interpolación entre parcelas de las zonas entregadas.
Muchas comparten clima y ubicación próxima. El diagnóstico GroupKFold separa zonas
de 0.05° asignadas por centroide, usando únicamente el conjunto de desarrollo.
Parcelas que cruzan límites pueden compartir parte del clima por la extracción zonal.
No constituye una validación espacial anidada y no se utiliza para reajustar la elección.
RMSE espacial diagnóstico: {rmse_espacial} t/ha.

## Intervalos de predicción

Semiamplitud conformal por zona, a partir de los residuos fuera de fold de las 138 parcelas
(municipios con ≥15 parcelas; el resto se agrupa). Cobertura aproximada, no garantizada:

| Zona | Parcelas | 80 % | 90 % |
|---|---:|---:|---:|
{tabla_int}

Los límites para cada parcela están en predicciones_59_intervalos.csv y resultados_para_mapa.csv.
Para un análisis de crédito, el límite inferior expresa un rendimiento conservador.

## Preguntas guía

{seccion_preguntas(guia)}
## Interpretación

importancia_variables.csv mide cuánto empeora el RMSE de validación al permutar conjuntamente
las variables de un índice o familia. Importancia positiva indica utilidad predictiva.
Se calcula únicamente en folds del conjunto de desarrollo. Es descriptiva, no causal;
la correlación entre índices puede repartir u ocultar importancia.

## Alcance y trazabilidad

El uso de datos hasta octubre permite una estimación al cierre del ciclo, no un pronóstico
anticipado. --mes-final limita las observaciones y obliga a volver a validar el nuevo modelo.
Los años históricos de clima, si se incorporan, aportan contexto y no etiquetas anuales.
El clima tiene resolución aproximada de 5 km y la topografía 120 m; la extracción no añade detalle.
Los 59 rendimientos ocultos se mantienen ausentes y nunca se usan para entrenar o evaluar.
Las predicciones no se recortan al rango observado de rendimientos ni se corrigen mirando la prueba.
El rendimiento futuro en otra campaña o región requiere datos y validación adicionales.

manifiesto.json registra versiones, parámetros, hashes de insumos y partición; calidad_datos.json
registra el tratamiento de datos. particion.csv identifica las parcelas de cada conjunto.
modelo_evaluacion.joblib conserva el modelo de 80%; modelo_final.joblib conserva el de todas las conocidas
y la tabla de intervalos. resultados_para_mapa.csv muestra, para parcelas con etiqueta, la estimación
de validación cruzada (sin ver su propio rendimiento), útil para comparar real y estimado en el mapa.

## Reproducir

```bash
.venv/bin/pip install -r requirements_prediccion.txt
.venv/bin/python predicción.py
```

Para recuperar predicciones del modelo guardado sin volver a buscar hiperparámetros:

```bash
.venv/bin/python predicción.py --modelo salidas_prediccion/modelo_final.joblib
```

No cambiar la semilla o configuración para buscar una prueba más favorable.
Una vez consultada, esta partición deja de ser una prueba nueva para futuras iteraciones.
La convocatoria permite IA y exige declarar dónde se utilizó y anexar los prompts.
El diseño y la implementación de este programa se realizaron con asistencia de Codex y de
Claude (Anthropic); los prompts están en registro_asistencia_IA.md.
Este reporte describe el componente analítico; la entrega del reto requiere además la interfaz,
el reporte técnico con la estructura de las bases y el video demostrativo.
"""
    (salida / "reporte.md").write_text(text, encoding="utf-8")
