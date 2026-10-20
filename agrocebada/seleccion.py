"""Selección del modelo con validación interna, diagnóstico espacial e importancia de variables."""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, RepeatedKFold

from .constantes import ID
from .evaluacion import evaluar_cv
from .modelos import crear_pipeline, generar_candidatos
from .utilidades import guardar_csv, guardar_json


def seleccionar_modelo(X, y, grupos, salida, modo, seed, threads):
    """No recibe X_test ni y_test: imposible utilizarlos para seleccionar."""
    # Con 110 parcelas, más repeticiones reducen el azar de la partición al elegir.
    repeticiones = 3 if modo == "completo" else 1
    splits = list(RepeatedKFold(n_splits=5, n_repeats=repeticiones, random_state=seed + 1).split(X))
    candidatos = generar_candidatos(modo, seed)
    resultados, modelos, predicciones = [], {}, {}
    for i, config in enumerate(candidatos, 1):
        started = time.monotonic()
        name = config["nombre"]
        model = crear_pipeline(config, grupos, seed, threads)
        try:
            cv = evaluar_cv(model, X, y, splits)
            resultados.append(dict(nombre=name, familia=config["familia"], grupo=config["grupo"],
                                   cv_rmse=cv["cv_rmse"], cv_rmse_std=cv["cv_rmse_std"],
                                   segundos=time.monotonic() - started,
                                   configuracion=json.dumps(config, ensure_ascii=False), estado="OK"))
            modelos[name], predicciones[name] = model, cv
            print(f"  {i:02d}/{len(candidatos)} {name}: CV RMSE {cv['cv_rmse']:.4f} t/ha ({time.monotonic()-started:.1f}s)", flush=True)
        except (ValueError, RuntimeError) as exc:
            resultados.append(dict(nombre=name, familia=config["familia"], grupo=config["grupo"],
                                   configuracion=json.dumps(config, ensure_ascii=False), estado=f"ERROR: {exc}"))
            print(f"  {name}: descartado por error: {exc}", flush=True)
        guardar_csv(pd.DataFrame(resultados), salida / "comparacion_cv.csv")
    validos = sorted((r for r in resultados if r["estado"] == "OK"), key=lambda r: r["cv_rmse"])
    if len(validos) < 2:
        raise RuntimeError("No hay suficientes modelos válidos para comparar.")
    # Promedios de familias distintas: pesos fijos, sin sobreajustar un metamodelo.
    diversos = []
    familias = set()
    for r in validos:
        if r["familia"] != "Media" and r["familia"] not in familias:
            diversos.append(r["nombre"])
            familias.add(r["familia"])
    for size in (2, 3):
        if len(diversos) < size:
            continue
        names = diversos[:size]
        by_fold = [np.mean([predicciones[n]["por_fold"][f] for n in names], axis=0) for f in range(len(splits))]
        mse = np.concatenate([(p - y.iloc[va].to_numpy()) ** 2 for p, (_, va) in zip(by_fold, splits)])
        fold_rmse = [float(np.sqrt(mean_squared_error(y.iloc[va], p))) for p, (_, va) in zip(by_fold, splits)]
        name = f"Ensamble_{size}_familias"
        modelos[name] = VotingRegressor([(f"m{i}", clone(modelos[n])) for i, n in enumerate(names)], n_jobs=1)
        config = dict(nombre=name, familia="Ensamble", miembros=[next(c for c in candidatos if c["nombre"] == n) for n in names], pesos=[1/size]*size)
        predicciones[name] = dict(cv_rmse=float(np.sqrt(mse.mean())), cv_rmse_std=float(np.std(fold_rmse)),
                                 oof=np.mean([predicciones[n]["oof"] for n in names], axis=0),
                                 por_fold=by_fold, fold_rmse=fold_rmse)
        resultados.append(dict(nombre=name, familia="Ensamble", grupo="mixto", cv_rmse=float(np.sqrt(mse.mean())),
                               cv_rmse_std=float(np.std(fold_rmse)), configuracion=json.dumps(config, ensure_ascii=False), estado="OK"))
    ranking = pd.DataFrame(resultados).sort_values("cv_rmse", na_position="last").reset_index(drop=True)
    guardar_csv(ranking, salida / "comparacion_cv.csv")
    best = ranking.loc[ranking.estado.eq("OK")].iloc[0]
    name = best["nombre"]
    config = json.loads(best["configuracion"])
    baseline_name = next(c["nombre"] for c in candidatos if c["familia"] == "Media")
    info = dict(configuracion=config, cv_rmse=float(best["cv_rmse"]), cv_rmse_std=float(best["cv_rmse_std"]),
                cv_baseline_rmse=predicciones[baseline_name]["cv_rmse"], folds=5, repeticiones=repeticiones,
                n_candidatos=len(resultados), criterio="Menor RMSE de predicciones fuera de fold, solo entrenamiento.",
                test_utilizado_para_seleccion=False)
    guardar_json(salida / "seleccion_congelada.json", info)
    oof = pd.DataFrame({ID:X.index, "REAL_T_HA":y.to_numpy(), "PREDICCION_OOF_T_HA":predicciones[name]["oof"]})
    guardar_csv(oof, salida / "validacion_interna_oof.csv")
    return modelos[name], info, ranking, splits


def diagnostico_espacial(model, X, y, meta, seed):
    """Diagnóstico en entrenamiento, por celdas de 0.05 grados."""
    geo = meta.loc[X.index, ["longitud", "latitud"]]
    groups = (np.floor((geo.longitud + 180) / .05 + 1e-8).astype(int).astype(str)
              + "_" + np.floor((geo.latitud + 90) / .05 + 1e-8).astype(int).astype(str))
    n = int(groups.nunique())
    if n < 3:
        return {"disponible":False, "motivo":"Menos de tres celdas climáticas en entrenamiento."}, None
    splits = list(GroupKFold(n_splits=min(5, n), shuffle=True, random_state=seed + 2).split(X, y, groups))
    cv = evaluar_cv(model, X, y, splits)
    info = dict(disponible=True, n_celdas=n, RMSE_t_ha=cv["cv_rmse"], RMSE_std=cv["cv_rmse_std"],
                nota="Diagnóstico por zonas SOLO en entrenamiento, con configuración ya seleccionada; no es validación espacial anidada ni interviene en la selección final.")
    frame = pd.DataFrame({ID:X.index, "CELDA_CLIMATICA":groups.to_numpy(), "REAL_T_HA":y.to_numpy(), "PREDICCION_OOF_T_HA":cv["oof"]})
    return info, frame


def familias_variables(columnas):
    families = defaultdict(list)
    for c in columnas:
        lc = c.lower()
        if lc.startswith("fenh_"):
            familia = "historial_" + "_".join(c.split("_")[1:3])
        elif lc.startswith("fen_"):
            familia = "_".join(c.split("_")[1:3])
        elif lc.startswith(("s2_", "planet_", "landsat_")):
            familia = "_".join(c.split("_")[:2])
        elif lc in ("x_km", "y_km") or lc.startswith("municipio_"):
            familia = "Ubicacion"
        elif "precip" in lc or "lluvia" in lc:
            familia = "Clima_lluvia"
        elif "tmin" in lc:
            familia = "Clima_Tmin"
        elif "tmax" in lc:
            familia = "Clima_Tmax"
        elif "amplitud" in lc and "term" in lc:
            familia = "Clima_amplitud_termica"
        elif "temperatura" in lc or lc.startswith("tmedia"):
            familia = "Clima_temperatura_media"
        elif "elevacion" in lc:
            familia = "Topografia_elevacion"
        elif "pendiente" in lc:
            familia = "Topografia_pendiente"
        elif "latitud" in lc or "longitud" in lc or "estado" in lc:
            familia = "Ubicacion"
        elif "area" in lc:
            familia = "Area"
        else:
            familia = "Otros"
        families[familia].append(c)
    return families


def importancia_grupos(model, X, y, splits, seed, repeticiones=3):
    """Permutación conjunta de cada índice/familia en folds de entrenamiento."""
    families = familias_variables(X.columns)
    rng = np.random.default_rng(seed + 3)
    rows = []
    # Primera repetición de CV: cada parcela se usa una vez para importancia.
    for fold, (tr, va) in enumerate(splits[:5], 1):
        est = clone(model).fit(X.iloc[tr], y.iloc[tr])
        base = float(np.sqrt(mean_squared_error(y.iloc[va], est.predict(X.iloc[va]))))
        for family, columns in families.items():
            for _ in range(repeticiones):
                xp = X.iloc[va].copy()
                perm = rng.permutation(len(va))
                xp.loc[:, columns] = xp[columns].iloc[perm].to_numpy()
                score = float(np.sqrt(mean_squared_error(y.iloc[va], est.predict(xp))))
                rows.append({"grupo":family, "incremento_RMSE_t_ha":score-base, "fold":fold})
    raw = pd.DataFrame(rows)
    result = raw.groupby("grupo")["incremento_RMSE_t_ha"].agg(["mean", "std", "count"]).reset_index()
    result.columns = ["GRUPO_VARIABLES", "INCREMENTO_RMSE_MEDIO_T_HA", "DESVIACION_T_HA", "N_PERMUTACIONES"]
    return result.sort_values("INCREMENTO_RMSE_MEDIO_T_HA", ascending=False)
