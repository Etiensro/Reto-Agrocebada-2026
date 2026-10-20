"""Respuestas a las tres preguntas guía de las bases mediante validación cruzada."""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluacion import evaluar_cv
from .modelos import GeoResidualRegressor
from .utilidades import guardar_csv, guardar_json


SENSORES = {"s2": "Sentinel-2", "planet": "Planet", "landsat": "Landsat"}


def bloques_variables(columnas):
    """Bloques temáticos fijados a priori para la pregunta 3."""
    estacional = re.compile(r"_m\d\d$")
    return {
        "Ubicación": [c for c in columnas if c in ("x_km", "y_km") or c.startswith(("estado_", "municipio_"))],
        "Superficie": [c for c in columnas if c in ("area_ha", "log_area_ha")],
        "Satélite 2025": [c for c in columnas if c.startswith("fen_")],
        "Historial satelital": [c for c in columnas if c.startswith("fenh_")],
        "Clima 2025": [c for c in columnas if c.startswith("clima_") and not estacional.search(c)],
        "Topografía": [c for c in columnas if c.startswith(("elevacion_m", "pendiente_grados"))],
    }


def _cv_rmse(model, X, y, splits, por_fold=False):
    """RMSE global fuera de fold, su dispersión entre repeticiones y R²."""
    cv = evaluar_cv(model, X, y, splits)
    por_rep = [float(np.mean(cv["fold_rmse"][i:i + 5])) for i in range(0, len(splits), 5)]
    out = (cv["cv_rmse"], float(np.std(por_rep)), float(r2_score(y, cv["oof"])))
    return out + (np.asarray(cv["fold_rmse"]),) if por_fold else out


def _ridge_simple():
    return Pipeline([("imputacion", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("escala", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-1, 3.5, 12)))])


def analisis_preguntas_guia(X, y, meta, salida, seed, threads):
    """Responde las preguntas guía con validación cruzada repetida.

    Se ejecuta DESPUÉS de evaluar la prueba y no modifica el modelo elegido.
    P1/P2 miden dos cosas: el poder del índice o variable por sí solo (Ridge solo
    con sus métricas) y su aporte adicional sobre la vecindad geográfica y la
    superficie (modelo geográfico-residual con una variable elegida en el fold).
    P3 compara todas las combinaciones de bloques temáticos con dos algoritmos y
    marca las estadísticamente empatadas con la mejor.
    """
    splits = list(RepeatedKFold(n_splits=5, n_repeats=4, random_state=seed + 7).split(X))
    municipio = meta.loc[X.index, "Municipio"]
    intra_y = y - y.groupby(municipio).transform("mean")
    referencia = GeoResidualRegressor(k_sel=0, n_vecinos=25, semilla=seed)
    ref_rmse, ref_std, ref_r2, ref_folds = _cv_rmse(referencia, X, y, splits, por_fold=True)

    def evaluar_familia(nombre, tipo, columnas):
        solo, solo_std, solo_r2 = _cv_rmse(_ridge_simple(), X[columnas], y, splits)
        con = GeoResidualRegressor(candidatas=tuple(columnas), k_sel=1, n_vecinos=25, semilla=seed)
        con_rmse, con_std, _, con_folds = _cv_rmse(con, X, y, splits, por_fold=True)
        # Comparación pareada: mismos folds para la referencia y el modelo con la variable.
        diferencia = ref_folds - con_folds
        error_estandar = float(np.std(diferencia, ddof=1) / np.sqrt(len(diferencia)))
        bloque = X[columnas]
        r_total = bloque.corrwith(y)
        r_intra = (bloque - bloque.groupby(municipio).transform("mean")).corrwith(intra_y)
        mejor = r_intra.abs().idxmax() if r_intra.notna().any() else None
        return {"VARIABLE": nombre, "TIPO": tipo, "N_COLUMNAS": len(columnas),
                "RMSE_SOLO_T_HA": solo, "R2_SOLO": solo_r2, "DESV_REPETICIONES_SOLO": solo_std,
                "RMSE_CON_UBICACION_T_HA": con_rmse, "DESV_REPETICIONES_CON_UBICACION": con_std,
                "APORTE_SOBRE_UBICACION_T_HA": ref_rmse - con_rmse,
                "APORTE_EE_T_HA": error_estandar,
                "APORTE_CLARO": bool(ref_rmse - con_rmse > 2 * error_estandar),
                "METRICA_MAS_ASOCIADA_INTRA_MUNICIPIO": mejor,
                "R_TOTAL": float(r_total.get(mejor, np.nan)) if mejor else None,
                "R_INTRA_MUNICIPIO": float(r_intra.get(mejor, np.nan)) if mejor else None}

    filas = []
    for c in sorted({"_".join(c.split("_")[1:3]) for c in X.columns if c.startswith("fen_")}):
        sensor, indice = c.split("_")
        columnas = [k for k in X.columns if k.startswith(f"fen_{c}_")]
        filas.append(evaluar_familia(f"{indice.upper()} ({SENSORES[sensor]})", "Índice 2025", columnas))
    for indice in ("ndvi", "evi"):
        columnas = [k for k in X.columns if k.startswith(f"fenh_s2_{indice}_")]
        if columnas:
            filas.append(evaluar_familia(f"{indice.upper()} (Sentinel-2, historial 2022-2024)",
                                         "Historial", columnas))
    p1 = pd.DataFrame(filas).sort_values("APORTE_SOBRE_UBICACION_T_HA", ascending=False)
    guardar_csv(p1, salida / "pregunta1_indices.csv")

    nombres_clima = {"precipitacion_mm": "Precipitación", "tmin_promedio_c": "Temperatura mínima",
                     "tmax_promedio_c": "Temperatura máxima",
                     "temperatura_media_aprox_c": "Temperatura media",
                     "amplitud_termica_promedio_c": "Amplitud térmica"}
    filas = []
    for clave, nombre in nombres_clima.items():
        columnas = [f"clima_{clave}_{e}" for e in ("temprano", "medio", "tardio", "ciclo")
                    if f"clima_{clave}_{e}" in X.columns]
        if columnas:
            filas.append(evaluar_familia(nombre, "Clima 2025", columnas))
    p2 = pd.DataFrame(filas).sort_values("RMSE_SOLO_T_HA")
    guardar_csv(p2, salida / "pregunta2_clima.csv")

    bloques = {k: v for k, v in bloques_variables(X.columns).items() if v}
    nombres = list(bloques)
    filas, folds_mejor = [], {}
    for n in range(1, len(nombres) + 1):
        for combo in itertools.combinations(nombres, n):
            columnas = [c for b in combo for c in bloques[b]]
            fila = {"COMBINACION": " + ".join(combo), "N_BLOQUES": n, "N_COLUMNAS": len(columnas)}
            arboles = Pipeline([("imputacion", SimpleImputer(strategy="median", keep_empty_features=True)),
                                ("modelo", ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5,
                                                               max_features=0.5, n_jobs=threads,
                                                               random_state=seed))])
            por_algoritmo = {}
            for etiqueta, modelo in (("RIDGE", _ridge_simple()), ("EXTRATREES", arboles)):
                rmse, desv, r2, folds = _cv_rmse(modelo, X[columnas], y, splits, por_fold=True)
                fila.update({f"RMSE_{etiqueta}_T_HA": rmse, f"R2_{etiqueta}": r2})
                por_algoritmo[etiqueta] = (rmse, folds)
            etiqueta = min(por_algoritmo, key=lambda k: por_algoritmo[k][0])
            fila.update({"RMSE_MEJOR_T_HA": por_algoritmo[etiqueta][0], "ALGORITMO_MEJOR": etiqueta})
            folds_mejor[fila["COMBINACION"]] = por_algoritmo[etiqueta][1]
            filas.append(fila)
    p3 = pd.DataFrame(filas).sort_values("RMSE_MEJOR_T_HA").reset_index(drop=True)
    # Diferencia pareada (mismos folds) contra la mejor: empate si < 2 errores estándar.
    lider = folds_mejor[p3.loc[0, "COMBINACION"]]
    diferencias = [folds_mejor[c] - lider for c in p3["COMBINACION"]]
    p3["DIFERENCIA_CON_MEJOR_T_HA"] = p3["RMSE_MEJOR_T_HA"] - p3.loc[0, "RMSE_MEJOR_T_HA"]
    p3["EE_DIFERENCIA_T_HA"] = [float(np.std(d, ddof=1) / np.sqrt(len(d))) for d in diferencias]
    p3["EMPATE_CON_MEJOR"] = p3["DIFERENCIA_CON_MEJOR_T_HA"] <= 2 * p3["EE_DIFERENCIA_T_HA"]
    guardar_csv(p3, salida / "pregunta3_combinaciones.csv")

    indices = p1[p1["TIPO"] == "Índice 2025"]
    resumen = {
        "nota": ("Validación cruzada 5x4 sobre las 138 parcelas conocidas, posterior a la prueba; "
                 "no interviene en la selección. Aporte positivo = reduce el RMSE."),
        "referencia_ubicacion_superficie": {"RMSE_t_ha": ref_rmse, "R2": ref_r2},
        "pregunta1_indice_mayor_aporte": indices.iloc[0][["VARIABLE", "APORTE_SOBRE_UBICACION_T_HA", "APORTE_EE_T_HA",
                                                          "APORTE_CLARO", "METRICA_MAS_ASOCIADA_INTRA_MUNICIPIO"]].to_dict(),
        "pregunta1_aportes_claros": p1.loc[p1["APORTE_CLARO"], "VARIABLE"].tolist(),
        "pregunta1_indice_mayor_poder_solo": indices.sort_values("RMSE_SOLO_T_HA").iloc[0][
            ["VARIABLE", "RMSE_SOLO_T_HA", "R2_SOLO"]].to_dict(),
        "pregunta1_mejor_familia_incluyendo_historial": p1.iloc[0][["VARIABLE", "APORTE_SOBRE_UBICACION_T_HA",
                                                                    "APORTE_EE_T_HA", "APORTE_CLARO"]].to_dict(),
        "pregunta2_variable_climatica_mayor_peso": p2.iloc[0][["VARIABLE", "RMSE_SOLO_T_HA", "R2_SOLO",
                                                               "APORTE_SOBRE_UBICACION_T_HA"]].to_dict(),
        "pregunta3_mejor_combinacion": p3.iloc[0][["COMBINACION", "RMSE_MEJOR_T_HA", "ALGORITMO_MEJOR",
                                                   "RMSE_RIDGE_T_HA", "RMSE_EXTRATREES_T_HA"]].to_dict(),
        "pregunta3_empatadas_con_la_mejor": p3.loc[p3["EMPATE_CON_MEJOR"], "COMBINACION"].tolist(),
    }
    guardar_json(salida / "preguntas_guia.json", resumen)
    return resumen, p1, p2, p3
