"""Métricas, partición 80/20 y validación cruzada."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .constantes import TARGET


def metricas(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Etiquetas/predicciones incompatibles o no finitas.")
    error = np.abs(pred - y)
    valid = np.abs(y) > 1e-10
    ape = error[valid] / np.abs(y[valid]) * 100
    return {
        "n": int(len(y)),
        "RMSE_t_ha": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE_t_ha": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else None,
        "sesgo_t_ha": float(np.mean(pred - y)),
        "MAPE_pct": float(ape.mean()) if len(ape) else None,
        "error_relativo_mediano_pct": float(np.median(ape)) if len(ape) else None,
        "dentro_10_pct": float((ape <= 10).mean() * 100) if len(ape) else None,
        "dentro_20_pct": float((ape <= 20).mean() * 100) if len(ape) else None,
    }


def dividir_parcelas(meta, test_size=0.2, seed=42):
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("--test-size debe estar entre 0.10 y 0.40.")
    conocidos = meta.index[meta[TARGET].notna()].sort_values()
    ocultos = meta.index[meta[TARGET].isna()].sort_values()
    train, test = train_test_split(np.asarray(conocidos), test_size=test_size, random_state=seed)
    train, test = pd.Index(sorted(train)), pd.Index(sorted(test))
    assert not set(train) & set(test)
    assert not (set(train) | set(test)) & set(ocultos)
    assert set(train) | set(test) == set(conocidos)
    return train, test, ocultos


def evaluar_cv(model, X, y, splits):
    acumulado = np.zeros(len(y))
    conteos = np.zeros(len(y), int)
    errores, fold_rmse = [], []
    predictions = []
    for tr, va in splits:
        est = clone(model).fit(X.iloc[tr], y.iloc[tr])
        pred = est.predict(X.iloc[va])
        if not np.isfinite(pred).all():
            raise ValueError("Predicción no finita en validación interna.")
        acumulado[va] += pred
        conteos[va] += 1
        errores.extend((pred - y.iloc[va].to_numpy()) ** 2)
        fold_rmse.append(float(np.sqrt(mean_squared_error(y.iloc[va], pred))))
        predictions.append(np.asarray(pred))
    if np.any(conteos == 0):
        raise ValueError("La validación no cubre todas las parcelas de entrenamiento.")
    return dict(cv_rmse=float(np.sqrt(np.mean(errores))), cv_rmse_std=float(np.std(fold_rmse)),
                oof=acumulado / conteos, por_fold=predictions, fold_rmse=fold_rmse)


def bootstrap_metricas(y, pred, seed, n_boot=2000):
    rng = np.random.default_rng(seed + 4)
    y, pred = np.asarray(y), np.asarray(pred)
    values = defaultdict(list)
    for _ in range(n_boot):
        ii = rng.integers(0, len(y), len(y))
        values["RMSE_t_ha"].append(float(np.sqrt(np.mean((pred[ii]-y[ii])**2))))
        values["MAE_t_ha"].append(float(np.abs(pred[ii]-y[ii]).mean()))
    return {k:{"inferior_95":float(np.quantile(v,.025)), "superior_95":float(np.quantile(v,.975))} for k,v in values.items()}
