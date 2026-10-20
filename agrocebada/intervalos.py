"""Intervalos de predicción por zona y variables del modelo ajustado."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .modelos import GeoResidualRegressor


def variables_seleccionadas(model):
    """Variables elegidas por los componentes geográfico-residuales ajustados."""
    componentes = getattr(model, "estimators_", None) or [model]
    out = {}
    for i, est in enumerate(componentes):
        if isinstance(est, GeoResidualRegressor):
            out[f"componente_{i}"] = {"variables": est.seleccionadas_,
                                     "correlacion_con_residuo": est.correlaciones_,
                                     "coeficientes_estandarizados": dict(zip(
                                         est.seleccionadas_, est.residual_.named_steps["ridge"].coef_.tolist()))
                                     if est.residual_ is not None else {}}
    return out


def zonas_intervalo(municipios, minimo=15):
    """Municipios con >=minimo residuos propios; el resto se agrupa."""
    conteo = municipios.value_counts()
    grandes = set(conteo.index[conteo >= minimo])
    return municipios.where(municipios.isin(grandes), "Otros municipios")


def intervalos_por_zona(residuos, zonas, niveles=(0.8, 0.9)):
    """Semiamplitud conformal (cuantil corregido de |residuo| fuera de fold)."""
    out = {}
    for zona, r in pd.Series(np.abs(residuos), index=zonas.index).groupby(zonas):
        n = len(r)
        out[zona] = {f"{int(nivel * 100)}": float(np.quantile(r, min(1.0, math.ceil((n + 1) * nivel) / n)))
                     for nivel in niveles}
        out[zona]["n"] = int(n)
    return out


def tabla_intervalos(estimado, municipios, intervalos):
    """Límites por zona; un municipio sin residuos propios usa su grupo o el global."""
    zona = municipios.where(municipios.isin(list(intervalos)), "Otros municipios")
    zona = zona.where(zona.isin(list(intervalos)), "Todas")
    out = pd.DataFrame({"ZONA_INTERVALO": zona}, index=estimado.index)
    for nivel in ("80", "90"):
        semi = zona.map(lambda z: intervalos[z][nivel]).astype(float)
        out[f"LIMITE_INFERIOR_{nivel}_T_HA"] = estimado - semi
        out[f"LIMITE_SUPERIOR_{nivel}_T_HA"] = estimado + semi
    return out
