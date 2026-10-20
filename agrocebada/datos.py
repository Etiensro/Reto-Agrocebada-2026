"""Preparación de datos: una fila por parcela, sin usar rendimientos para construir predictores."""
from __future__ import annotations

import calendar
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constantes import ANIOS_HISTORICOS, METRICAS_FENOLOGICAS


def metricas_fenologicas(obs, ids, inicio, fin, paso_dias):
    """Resume la curva estacional de un índice en cada parcela, sin etiquetas.

    obs: ID_POLIGONO, fecha y valor, ya filtrados por nubosidad y periodo.
    La curva usa la mediana móvil de tres fechas (atenúa nubes residuales) y se
    interpola en una malla regular. Los días se cuentan desde el inicio.
    base: mediana de los primeros 61 días (abril-mayo), antes de la siembra.
    inicio/fin: cruces del 50 % de la amplitud antes y después del pico.
    integral: área sobre la base (vigor acumulado); integral_total: área bruta.
    Se requieren al menos cuatro fechas válidas; si no, todo queda ausente.
    """
    malla = pd.date_range(inicio, fin, freq=f"{paso_dias}D")
    t = np.asarray((malla - inicio).days, dtype=float)
    en_base = np.asarray(malla < inicio + pd.Timedelta(days=61))
    filas = {}
    for pid, grupo in obs.dropna(subset=["valor"]).groupby("ID_POLIGONO"):
        serie = grupo.groupby("fecha")["valor"].median().sort_index()
        if len(serie) < 4 or len(t) < 2:
            continue
        suave = serie.rolling(3, center=True, min_periods=1).median()
        v = np.interp(t, np.asarray((suave.index - inicio).days, dtype=float), suave.to_numpy())
        ipk = int(v.argmax())
        base = float(np.median(v[en_base]))
        amplitud = float(v[ipk] - base)
        mitad = base + amplitud / 2
        sube = np.flatnonzero(v[:ipk + 1] >= mitad)
        baja = np.flatnonzero(v[ipk:] >= mitad)
        ini = t[sube[0]] if amplitud > 0 and len(sube) else np.nan
        fin_t = t[ipk + baja[-1]] if amplitud > 0 and len(baja) else np.nan
        filas[pid] = dict(pico=float(v[ipk]), p90=float(np.percentile(serie, 90)), base=base,
                          amplitud=amplitud, dia_pico=t[ipk], inicio=ini, fin=fin_t,
                          duracion=fin_t - ini,
                          integral=float(np.trapezoid(np.clip(v - base, 0, None), t)) / 100,
                          integral_total=float(np.trapezoid(v, t)) / 100)
    return pd.DataFrame.from_dict(filas, orient="index", columns=list(METRICAS_FENOLOGICAS)).reindex(ids)


def preparar_datos(dataset: Path, extraidos: Path, cloud_max: float = 20,
                   cobertura_min: float = 80, mes_final: int = 10):
    """Construye variables sin aprender parámetros ni consultar rendimientos.

    Los únicos objetivos se leen del archivo oficial de etiquetas. Las medianas
    satelitales se calculan dentro de cada parcela y mes, nunca entre parcelas.
    Los NaN se conservan para la imputación dentro del entrenamiento/CV.
    """
    if not 4 <= mes_final <= 10:
        raise ValueError("mes_final debe estar entre 4 y 10, dentro del ciclo 2025.")
    if not 0 <= cloud_max <= 100 or not 0 <= cobertura_min <= 100:
        raise ValueError("Los umbrales de nube y cobertura deben estar entre 0 y 100.")
    dataset, extraidos = Path(dataset), Path(extraidos)
    fuentes = {
        "etiquetas": dataset / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        "parcelas": extraidos / "parcelas_datos.csv",
        "clima": extraidos / "clima_mensual.csv",
        "topografia": extraidos / "topografia_parcelas.csv",
        "basico": dataset / "Conjunto_datos_BASICO_AgroCebada2026.csv",
        "pro": dataset / "Conjunto_datos_PRO_AgroCebada.csv",
    }
    meses = list(range(4, mes_final + 1))
    corte = pd.Timestamp(2025, mes_final, calendar.monthrange(2025, mes_final)[1])
    calidad = {"fuentes": {k: str(v.resolve()) for k, v in fuentes.items()},
               "fecha_corte": corte.date().isoformat(), "anio_objetivo": 2025,
               "meses": meses, "cloud_max": float(cloud_max),
               "cobertura_min": float(cobertura_min), "satelites": {}}

    def leer(clave, columnas=None):
        return pd.read_csv(fuentes[clave], usecols=columnas, encoding="utf-8-sig")

    def validar_ids(tabla, nombre, unico=False):
        if tabla["ID_POLIGONO"].isna().any():
            raise ValueError(f"{nombre}: hay identificadores vacíos.")
        if set(tabla["ID_POLIGONO"]) != set(ids):
            raise ValueError(f"{nombre}: IDs diferentes a los 197 oficiales.")
        if unico and tabla["ID_POLIGONO"].duplicated().any():
            raise ValueError(f"{nombre}: hay parcelas duplicadas.")

    objetivo = ["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"]
    meta = leer("etiquetas", objetivo).sort_values("ID_POLIGONO")
    ids = pd.Index(meta["ID_POLIGONO"], name="ID_POLIGONO")
    esperados = {f"AGC_{n:03d}" for n in range(1, 198)}
    if len(meta) != 197 or set(ids) != esperados or ids.has_duplicates:
        raise ValueError("Las etiquetas deben tener exactamente AGC_001 a AGC_197.")
    meta = meta.set_index("ID_POLIGONO")
    for campo in ["AREA_HA", "RENDIMIENTO_T_HA"]:
        meta[campo] = pd.to_numeric(meta[campo], errors="raise")
    conocidos = meta["RENDIMIENTO_T_HA"].notna()
    if (conocidos.sum() != 138 or (~conocidos).sum() != 59 or
            not meta.loc[conocidos, "CONJUNTO"].eq("ENTRENAMIENTO").all() or
            not meta.loc[~conocidos, "CONJUNTO"].eq("PREDICCION").all()):
        raise ValueError("Deben existir 138 etiquetas de entrenamiento y 59 ocultas.")
    if (not np.isfinite(meta.loc[conocidos, "RENDIMIENTO_T_HA"]).all() or
            not np.isfinite(meta["AREA_HA"]).all() or (meta["AREA_HA"] <= 0).any()):
        raise ValueError("Rendimiento/área no finitos o área no positiva.")
    # Whitelist: no se leen objetivos repetidos de las salidas del extractor.
    geo = leer("parcelas", ["ID_POLIGONO", "AREA_HA", "Municipio", "Estado",
                            "longitud", "latitud"])
    validar_ids(geo, "parcelas_datos", unico=True)
    geo = geo.set_index("ID_POLIGONO").reindex(ids)
    if not np.allclose(meta["AREA_HA"], geo["AREA_HA"], atol=1e-5, rtol=0):
        raise ValueError("El área extraída no coincide con el archivo oficial.")
    if not np.isfinite(geo[["longitud", "latitud"]].to_numpy(dtype=float)).all():
        raise ValueError("Las coordenadas de las parcelas deben ser finitas.")
    if (not geo["longitud"].between(-180, 180).all() or
            not geo["latitud"].between(-90, 90).all()):
        raise ValueError("Coordenadas geográficas fuera de rango.")
    estados = ["Hidalgo", "Puebla", "Tlaxcala"]
    if not geo["Estado"].isin(estados).all():
        raise ValueError("Estado ausente o fuera de Hidalgo, Puebla y Tlaxcala.")
    meta = meta.join(geo.drop(columns="AREA_HA"), validate="one_to_one")
    features = {"area_ha": meta["AREA_HA"], "log_area_ha": np.log1p(meta["AREA_HA"]),
                "longitud": geo["longitud"], "latitud": geo["latitud"]}
    for estado in estados:
        features[f"estado_{estado.lower()}"] = geo["Estado"].eq(estado).astype(float)
    # Proyección equirectangular local: distancias en km, isotrópicas, para vecindad.
    lat0 = math.radians(float(geo["latitud"].mean()))
    lon0, lat0_g = float(geo["longitud"].mean()), float(geo["latitud"].mean())
    features["x_km"] = np.radians(geo["longitud"] - lon0) * math.cos(lat0) * 6371.0
    features["y_km"] = np.radians(geo["latitud"] - lat0_g) * 6371.0
    # Municipios con al menos cinco parcelas (de las 197; no usa rendimientos).
    conteo_municipios = geo["Municipio"].value_counts()
    municipios = sorted(conteo_municipios.index[conteo_municipios >= 5])
    zona = []
    for municipio in municipios:
        nombre = "municipio_" + re.sub(r"[^a-z0-9]+", "_", municipio.lower()).strip("_")
        features[nombre] = geo["Municipio"].eq(municipio).astype(float)
        zona.append(nombre)

    topo_cols = ["ID_POLIGONO"]
    for variable in ["elevacion_m", "pendiente_grados"]:
        topo_cols += [variable, f"{variable}_min", f"{variable}_max",
                      f"{variable}_std", f"{variable}_cobertura_pct"]
    topo = leer("topografia", topo_cols)
    validar_ids(topo, "topografia", unico=True)
    topo = topo.set_index("ID_POLIGONO").reindex(ids)
    calidad["topografia_enmascarada"] = {}
    for variable in ["elevacion_m", "pendiente_grados"]:
        invalido = topo[f"{variable}_cobertura_pct"].lt(cobertura_min)
        invalido |= topo[f"{variable}_cobertura_pct"].isna()
        calidad["topografia_enmascarada"][variable] = ids[invalido].tolist()
        features[variable] = topo[variable].mask(invalido)
        features[f"{variable}_std"] = topo[f"{variable}_std"].mask(invalido)
        features[f"{variable}_rango"] = (
            topo[f"{variable}_max"] - topo[f"{variable}_min"]).mask(invalido)

    variables_clima = ["precipitacion_mm", "tmin_promedio_c", "tmax_promedio_c"]
    coberturas = [f"{v}_cobertura_pct" for v in variables_clima]
    clima = leer("clima", ["ID_POLIGONO", "anio", "mes"] + variables_clima + coberturas)
    validar_ids(clima, "clima_mensual")
    if clima.duplicated(["ID_POLIGONO", "anio", "mes"]).any():
        raise ValueError("Hay duplicados parcela/año/mes en clima_mensual.")
    clima = clima.loc[clima["anio"].eq(2025) & clima["mes"].isin(meses)].copy()
    esperado = pd.MultiIndex.from_product([ids, meses], names=["ID_POLIGONO", "mes"])
    clima = clima.set_index(["ID_POLIGONO", "mes"])
    if set(clima.index) != set(esperado):
        raise ValueError("Faltan meses climáticos del ciclo 2025 en alguna parcela.")
    clima = clima.reindex(esperado)
    if not np.isfinite(clima[variables_clima].to_numpy(dtype=float)).all():
        raise ValueError("El ciclo climático solicitado contiene valores ausentes/no finitos.")
    calidad["clima_celdas_enmascaradas"] = {}
    for variable in variables_clima:
        malo = clima[f"{variable}_cobertura_pct"].lt(cobertura_min)
        malo |= clima[f"{variable}_cobertura_pct"].isna()
        calidad["clima_celdas_enmascaradas"][variable] = int(malo.sum())
        clima[variable] = clima[variable].mask(malo)
    tablas = {v: clima[v].unstack("mes").reindex(index=ids, columns=meses)
              for v in variables_clima}
    tablas["temperatura_media_aprox_c"] = (
        tablas["tmin_promedio_c"] + tablas["tmax_promedio_c"]) / 2
    tablas["amplitud_termica_promedio_c"] = (
        tablas["tmax_promedio_c"] - tablas["tmin_promedio_c"])
    etapas = {"temprano": [m for m in meses if m <= 6],
              "medio": [m for m in meses if 7 <= m <= 8],
              "tardio": [m for m in meses if m >= 9], "ciclo": meses}
    for variable, tabla in tablas.items():
        for mes in meses:
            features[f"clima_{variable}_m{mes:02d}"] = tabla[mes]
        for etapa, subset in etapas.items():
            if subset:
                bloque = tabla[subset]
                agregado = (bloque.sum(axis=1, min_count=len(subset))
                            if variable == "precipitacion_mm"
                            else bloque.mean(axis=1).mask(bloque.isna().any(axis=1)))
                features[f"clima_{variable}_{etapa}"] = agregado
    # Las coordenadas en km son redundantes con longitud/latitud en estos grupos.
    contexto = [c for c in features if c not in ("x_km", "y_km")]

    s2_indices = ["ndvi", "evi", "crc", "mcrc", "ndsvi", "srndi", "lai", "fapar",
                  "savi", "ndti", "sti", "ndwi", "endwi", "msi", "emsi", "dswi2",
                  "dswi3", "dswi4", "dswi5", "nddi", "enddi"]
    compactos = {"s2": {"ndvi", "evi", "ndwi", "msi", "fapar", "lai", "savi", "nddi"},
                 "planet": {"ndvi", "evi", "lai", "msavi"}, "landsat": {"vi6t"}}
    # Fenología: todos los índices entregados (MRA solo existe en Sentinel-2).
    indices_fenologia = {"s2": s2_indices + ["mra"], "landsat": ["vi6t"],
                         "planet": ["ndvi", "evi", "lai", "msavi"]}
    paso_malla = {"s2": 8, "landsat": 16, "planet": 5}
    inicio_ciclo = pd.Timestamp(2025, 4, 1)
    columnas_sat, columnas_compactas, columnas_fen, columnas_hist = [], [], [], []
    calidad["fenologia"] = {"inicio": inicio_ciclo.date().isoformat(), "fin": corte.date().isoformat(),
                            "anios_historicos": list(ANIOS_HISTORICOS), "paso_dias": paso_malla,
                            "parcelas_sin_curva": {}}
    for fuente, sensores in [("basico", [("Sentinel-2", "s2", s2_indices),
                                         ("Landsat", "landsat", ["vi6t"])]),
                              ("pro", [("Planet", "planet", ["ndvi", "evi", "lai", "msavi"])])]:
        columnas = ["ID_POLIGONO", "fecha_captura", "sensor", "porcentaje_nubosidad"]
        columnas += [f"{v}_{s}" for _, _, indices in sensores
                     for v in indices for s in ["promedio", "std"]]
        columnas += [f"{v}_promedio" for _, prefijo, _ in sensores
                     for v in indices_fenologia[prefijo] if f"{v}_promedio" not in columnas]
        sat = leer(fuente, columnas)
        validar_ids(sat, fuente)
        sat["fecha"] = pd.to_datetime(sat["fecha_captura"], format="%d/%m/%Y", errors="raise")
        if sat.duplicated(["ID_POLIGONO", "fecha", "sensor"]).any():
            raise ValueError(f"{fuente}: observaciones satelitales duplicadas.")
        nube = sat["porcentaje_nubosidad"]
        if not nube.dropna().between(0, 100).all():
            raise ValueError(f"{fuente}: nubosidad fuera de rango.")
        periodo = sat["fecha"].between(pd.Timestamp(2025, 4, 1), corte)
        for sensor, prefijo, indices in sensores:
            bloque = sat.loc[periodo & sat["sensor"].eq(sensor)].copy()
            buenos = bloque["porcentaje_nubosidad"].le(cloud_max)
            calidad["satelites"][sensor] = {"filas_periodo": int(len(bloque)),
                "filas_nube_aceptable": int(buenos.sum()),
                "filas_descartadas_nube_o_sin_nubosidad": int((~buenos).sum())}
            bloque = bloque.loc[buenos].copy()
            bloque["mes"] = bloque["fecha"].dt.month
            for variable in indices:
                nombre = f"{prefijo}_{variable}"
                nuevos = {}
                tabla = None
                for estadistico, sufijo in [("promedio", ""), ("std", "_std_espacial")]:
                    serie = pd.to_numeric(bloque[f"{variable}_{estadistico}"], errors="raise")
                    serie = serie.where(np.isfinite(serie))
                    temporal = bloque[["ID_POLIGONO", "mes"]].assign(valor=serie)
                    mensual = temporal.groupby(["ID_POLIGONO", "mes"])["valor"].median()
                    mensual = mensual.unstack("mes").reindex(index=ids, columns=meses)
                    for mes in meses:
                        nuevos[f"{nombre}{sufijo}_m{mes:02d}"] = mensual[mes]
                    if estadistico == "promedio":
                        tabla = mensual
                nuevos[f"{nombre}_media"] = tabla.mean(axis=1)
                nuevos[f"{nombre}_max"] = tabla.max(axis=1)
                nuevos[f"{nombre}_amplitud"] = tabla.max(axis=1) - tabla.min(axis=1)
                nuevos[f"{nombre}_delta_tardio_temprano"] = (
                    tabla[etapas["tardio"]].mean(axis=1) - tabla[etapas["temprano"]].mean(axis=1))
                valores = tabla.to_numpy(dtype=float)
                finito = np.isfinite(valores)
                pico = np.asarray(meses, dtype=float)[np.where(finito, valores, -np.inf).argmax(axis=1)]
                nuevos[f"{nombre}_mes_pico"] = pd.Series(
                    np.where(finito.any(axis=1), pico, np.nan), index=ids)
                tiempo = np.asarray(meses, dtype=float)[None, :]
                n = finito.sum(axis=1)
                sx = np.where(finito, tiempo, 0).sum(axis=1)
                sy = np.where(finito, valores, 0).sum(axis=1)
                sxx = np.where(finito, tiempo ** 2, 0).sum(axis=1)
                sxy = np.where(finito, valores * tiempo, 0).sum(axis=1)
                denominador = n * sxx - sx ** 2
                pendiente = np.divide(n * sxy - sx * sy, denominador,
                                      out=np.full(len(ids), np.nan), where=denominador > 0)
                nuevos[f"{nombre}_pendiente"] = pd.Series(pendiente, index=ids)
                features.update(nuevos)
                columnas_sat.extend(nuevos)
                if variable in compactos[prefijo]:
                    columnas_compactas.extend(nuevos)
            limpias = sat.loc[sat["sensor"].eq(sensor) & sat["porcentaje_nubosidad"].le(cloud_max)]
            for variable in indices_fenologia[prefijo]:
                valores = pd.to_numeric(limpias[f"{variable}_promedio"], errors="raise")
                obs = limpias[["ID_POLIGONO", "fecha"]].assign(valor=valores.where(np.isfinite(valores)))
                ciclo = obs.loc[obs["fecha"].between(inicio_ciclo, corte)]
                tabla = metricas_fenologicas(ciclo, ids, inicio_ciclo, corte, paso_malla[prefijo])
                calidad["fenologia"]["parcelas_sin_curva"][f"{prefijo}_{variable}"] = int(tabla["pico"].isna().sum())
                nuevos = {f"fen_{prefijo}_{variable}_{m}": tabla[m] for m in METRICAS_FENOLOGICAS}
                features.update(nuevos)
                columnas_fen.extend(nuevos)
                if prefijo != "s2" or variable not in ("ndvi", "evi"):
                    continue
                # Historial previo al ciclo: productividad habitual de la parcela.
                por_anio = {}
                for anio in ANIOS_HISTORICOS:
                    inicio_h, fin_h = pd.Timestamp(anio, 4, 1), pd.Timestamp(anio, 11, 30)
                    tabla = metricas_fenologicas(obs.loc[obs["fecha"].between(inicio_h, fin_h)], ids,
                                                 inicio_h, fin_h, paso_malla[prefijo])
                    por_anio[anio] = tabla
                    for m in ("pico", "base", "integral"):
                        features[f"fenh_s2_{variable}_{anio}_{m}"] = tabla[m]
                        columnas_hist.append(f"fenh_s2_{variable}_{anio}_{m}")
                for m in ("pico", "base", "integral"):
                    features[f"fenh_s2_{variable}_media_{m}"] = pd.concat(
                        [t[m] for t in por_anio.values()], axis=1).mean(axis=1)
                    columnas_hist.append(f"fenh_s2_{variable}_media_{m}")
    X = pd.DataFrame(features, index=ids).astype(float).replace([np.inf, -np.inf], np.nan)
    geografia = {"longitud", "latitud"} | {f"estado_{e.lower()}" for e in estados} | set(zona)
    integrado = contexto + columnas_compactas
    grupos = {"contexto": contexto, "integrado": integrado,
              "amplio": contexto + columnas_sat,
              "sin_geografia": [c for c in integrado if c not in geografia],
              "satelite": ["area_ha", "log_area_ha"] + columnas_compactas,
              # Candidatas del modelo geográfico-residual y de modelos generales.
              "fenologia": columnas_fen + columnas_hist,
              "integrado_fenologia": contexto + columnas_fen + columnas_hist,
              "geografia_km": ["x_km", "y_km"]}
    # Subconjuntos físicos fijados a priori: evitan que cientos de variables
    # correlacionadas dominen las distancias de los modelos espaciales.
    grupos["geografia"] = ["longitud", "latitud", "estado_hidalgo", "estado_puebla", "estado_tlaxcala"]
    grupos["compacto"] = [c for c in contexto if not c.endswith(tuple(f"_m{m:02d}" for m in meses))]
    calidad.update({"parcelas": len(ids), "etiquetas_visibles": int(conocidos.sum()),
                    "etiquetas_ocultas": int((~conocidos).sum()),
                    "variables_por_grupo": {k: len(v) for k, v in grupos.items()},
                    "faltantes_por_variable": {c: int(n) for c, n in X.isna().sum().items() if n},
                    "columnas_totalmente_ausentes": X.columns[X.isna().all()].tolist()})
    return X, meta, grupos, calidad
