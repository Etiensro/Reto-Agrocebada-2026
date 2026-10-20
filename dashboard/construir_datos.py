#!/usr/bin/env python3
"""Genera dashboard/datos.js a partir de las salidas del modelo.

Ejecución (desde la carpeta del proyecto):
    .venv/bin/python dashboard/construir_datos.py

Lee salidas_prediccion/ y salidas_extractor/parcelas.geojson. No entrena nada:
exporta los parámetros exactos del modelo final (vecinos y regresión Ridge) para
que el panel pueda ejecutarlo en el navegador y reproduzca las predicciones.
Opción --autonomo: además escribe dashboard/panel_autonomo.html, un único archivo
con estilos, código y datos incorporados, útil para enviarlo o publicarlo.

También genera dashboard/territorio.js (clima y relieve por parcela, de
salidas_extractor/datos_por_imagen.csv) y dashboard/panel_mapa_satelital.html:
el mismo panel con fondo satelital o de calles y las capas del mapa del extractor.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "salidas_prediccion"
DASH = Path(__file__).resolve().parent
RADIO_TIERRA_KM = 6371.0
VARIABLES_MODELO = {
    "log_area_ha": "log_area",
    "fenh_s2_evi_2022_base": "evi_2022",
    "fenh_s2_evi_2023_base": "evi_2023",
    "fenh_s2_ndvi_media_base": "ndvi_media",
}


def leer(nombre):
    return pd.read_csv(SALIDA / nombre, encoding="utf-8-sig")


def redondear(valor, cifras=4):
    if valor is None or (isinstance(valor, float) and not math.isfinite(valor)):
        return None
    return round(float(valor), cifras)


def proyeccion(variables: pd.DataFrame, parcelas_geo: pd.DataFrame):
    """Recupera el origen usado por agrocebada/datos.py: la media de las 197 parcelas."""
    lon0 = float(parcelas_geo["longitud"].mean())
    lat0 = float(parcelas_geo["latitud"].mean())
    x = np.radians(parcelas_geo["longitud"] - lon0) * math.cos(math.radians(lat0)) * RADIO_TIERRA_KM
    y = np.radians(parcelas_geo["latitud"] - lat0) * RADIO_TIERRA_KM
    ref = variables.set_index("ID_POLIGONO").loc[parcelas_geo["ID_POLIGONO"], ["x_km", "y_km"]]
    if not (np.allclose(x, ref["x_km"], atol=1e-4) and np.allclose(y, ref["y_km"], atol=1e-4)):
        raise ValueError("La proyección reconstruida no coincide con x_km/y_km del modelo.")
    return lon0, lat0


def construir() -> dict:
    sys.path.insert(0, str(RAIZ))
    importlib.import_module("agrocebada.modelos")  # la clase GeoResidualRegressor debe poder importarse
    paquete = joblib.load(SALIDA / "modelo_final.joblib")
    modelo = paquete["modelo"]
    if type(modelo).__name__ != "GeoResidualRegressor":
        raise ValueError("El panel exporta el modelo geográfico-residual; el modelo final es otro: "
                         f"{type(modelo).__name__}.")
    metricas = json.loads((SALIDA / "metricas.json").read_text(encoding="utf-8"))
    mapa = leer("resultados_para_mapa.csv").set_index("ID_POLIGONO")
    variables = leer("variables_por_parcela.csv")
    geojson = json.loads((RAIZ / "salidas_extractor" / "parcelas.geojson").read_text(encoding="utf-8"))

    geo = pd.DataFrame([f["properties"] for f in geojson["features"]])[
        ["ID_POLIGONO", "longitud", "latitud"]]
    lon0, lat0 = proyeccion(variables, geo)
    cos_lat0 = math.cos(math.radians(lat0))

    def a_km(lon, lat):
        return (math.radians(lon - lon0) * cos_lat0 * RADIO_TIERRA_KM,
                math.radians(lat - lat0) * RADIO_TIERRA_KM)

    poligonos = {}
    for f in geojson["features"]:
        anillo = f["geometry"]["coordinates"][0]
        poligonos[f["properties"]["ID_POLIGONO"]] = [
            [round(c, 3) for c in a_km(lon, lat)] for lon, lat in anillo]

    vars_modelo = variables.set_index("ID_POLIGONO")
    ids = list(mapa.index)
    parcelas = []
    for pid in ids:
        fila = mapa.loc[pid]
        v = vars_modelo.loc[pid]
        parcelas.append({
            "id": pid,
            "estado": fila["Estado"],
            "municipio": fila["Municipio"],
            "lon": redondear(fila["longitud"], 6),
            "lat": redondear(fila["latitud"], 6),
            "x": redondear(v["x_km"], 6),
            "y": redondear(v["y_km"], 6),
            "area": redondear(fila["AREA_HA"], 2),
            "real": redondear(fila["RENDIMIENTO_T_HA"]),
            "estimado": redondear(fila["RENDIMIENTO_ESTIMADO_T_HA"]),
            "origen": "ocultas" if fila["ORIGEN_ESTIMACION"] == "PREDICCION_SIN_ETIQUETA" else "conocidas",
            "zona": fila["ZONA_INTERVALO"],
            "lo80": redondear(fila["LIMITE_INFERIOR_80_T_HA"]),
            "hi80": redondear(fila["LIMITE_SUPERIOR_80_T_HA"]),
            "lo90": redondear(fila["LIMITE_INFERIOR_90_T_HA"]),
            "hi90": redondear(fila["LIMITE_SUPERIOR_90_T_HA"]),
            "variables": {corto: redondear(v[largo], 6) for largo, corto in VARIABLES_MODELO.items()},
            "poligono": poligonos[pid],
        })

    # Parámetros exactos del modelo final.
    if list(modelo.coordenadas) != ["x_km", "y_km"]:
        raise ValueError("Se esperaban coordenadas x_km/y_km.")
    ridge = modelo.residual_.named_steps
    nombres = list(modelo.seleccionadas_)
    entrenamiento = paquete["ids_entrenamiento"]
    mapa_ent = mapa.loc[entrenamiento]
    vecinos = [{"id": pid, "x": redondear(vars_modelo.loc[pid, "x_km"], 6),
                "y": redondear(vars_modelo.loc[pid, "y_km"], 6),
                "real": redondear(mapa_ent.loc[pid, "RENDIMIENTO_T_HA"], 4)} for pid in entrenamiento]
    parametros = {
        "n_vecinos": int(modelo.vecinos_.n_neighbors),
        "variables": [VARIABLES_MODELO[n] for n in nombres],
        "variables_origen": nombres,
        "medianas": [float(x) for x in ridge["imputacion"].statistics_],
        "media": [float(x) for x in ridge["escala"].mean_],
        "escala": [float(x) for x in ridge["escala"].scale_],
        "coeficientes": [float(x) for x in ridge["ridge"].coef_],
        "intercepto": float(ridge["ridge"].intercept_),
        "alpha": float(modelo.alpha),
        "entrenamiento": vecinos,
    }

    # Residuos fuera de fold (real - estimado) por zona, para probabilidades.
    conocidas = mapa[mapa["RENDIMIENTO_T_HA"].notna()]
    residuos = (conocidas["RENDIMIENTO_T_HA"] - conocidas["RENDIMIENTO_ESTIMADO_T_HA"])
    por_zona = {z: [redondear(r, 4) for r in residuos[conocidas["ZONA_INTERVALO"] == z]]
                for z in sorted(conocidas["ZONA_INTERVALO"].unique())}
    por_zona["Todas"] = [redondear(r, 4) for r in residuos]

    prueba = leer("predicciones_prueba.csv")
    importancia = leer("importancia_variables.csv").head(12)
    comparacion = leer("comparacion_cv.csv")
    familias = (comparacion[comparacion["estado"] == "OK"].groupby("familia")
                .agg(mejor_rmse=("cv_rmse", "min"), candidatos=("nombre", "count"))
                .reset_index().sort_values("mejor_rmse"))
    guia = json.loads((SALIDA / "preguntas_guia.json").read_text(encoding="utf-8"))
    p1, p2, p3 = (leer("pregunta1_indices.csv"), leer("pregunta2_clima.csv"),
                  leer("pregunta3_combinaciones.csv"))
    manifiesto = json.loads((SALIDA / "manifiesto.json").read_text(encoding="utf-8"))

    def tabla(frame, columnas):
        return [{c: (redondear(v, 5) if isinstance(v, (float, np.floating)) else
                     (bool(v) if isinstance(v, (np.bool_, bool)) else v))
                 for c, v in zip(columnas, fila) if not (isinstance(v, float) and math.isnan(v))}
                for fila in frame[columnas].itertuples(index=False, name=None)]

    cv = metricas["validacion_cruzada_138"]
    return {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrenado": manifiesto["fecha_utc"],
        "proyeccion": {"lon0": lon0, "lat0": lat0, "radio_km": RADIO_TIERRA_KM},
        "parcelas": parcelas,
        "modelo": parametros,
        "intervalos": paquete["intervalos"],
        "residuos_por_zona": por_zona,
        "metricas": {
            "cv138": {k: redondear(cv[k]) for k in ("RMSE_t_ha", "MAE_t_ha", "R2", "sesgo_t_ha",
                                                    "MAPE_pct", "dentro_10_pct", "dentro_20_pct")}
                     | {"n": cv["n"], "folds": cv["folds"], "repeticiones": cv["repeticiones"]},
            "prueba": {k: redondear(v) for k, v in metricas["prueba"].items()},
            "baseline_prueba": {k: redondear(v) for k, v in metricas["baseline_prueba"].items()},
            "cv_seleccion_rmse": redondear(metricas["seleccion"]["cv_rmse"]),
            "cv_media_rmse": redondear(metricas["seleccion"]["cv_baseline_rmse"]),
            "mejora_vs_media_pct": redondear(metricas["mejora_RMSE_vs_media_pct"], 2),
            "n_candidatos": metricas["seleccion"]["n_candidatos"],
            "modelo": metricas["seleccion"]["configuracion"]["nombre"],
            "espacial_rmse": redondear(metricas["diagnostico_espacial_entrenamiento"].get("RMSE_t_ha")),
        },
        "prueba": tabla(prueba, ["ID_POLIGONO", "REAL_T_HA", "PREDICHO_T_HA", "BASELINE_T_HA"]),
        "importancia": tabla(importancia, ["GRUPO_VARIABLES", "INCREMENTO_RMSE_MEDIO_T_HA", "DESVIACION_T_HA"]),
        "familias": tabla(familias, ["familia", "mejor_rmse", "candidatos"]),
        "guia": {
            "resumen": guia,
            "indices": tabla(p1, ["VARIABLE", "TIPO", "RMSE_SOLO_T_HA", "R2_SOLO", "APORTE_SOBRE_UBICACION_T_HA",
                                  "APORTE_EE_T_HA", "APORTE_CLARO", "METRICA_MAS_ASOCIADA_INTRA_MUNICIPIO"]),
            "clima": tabla(p2, ["VARIABLE", "RMSE_SOLO_T_HA", "R2_SOLO", "APORTE_SOBRE_UBICACION_T_HA"]),
            "combinaciones": tabla(p3.head(20), ["COMBINACION", "RMSE_RIDGE_T_HA", "RMSE_EXTRATREES_T_HA",
                                                 "RMSE_MEJOR_T_HA", "ALGORITMO_MEJOR", "EMPATE_CON_MEJOR"]),
            "n_combinaciones": int(len(p3)),
        },
    }


NOMBRES_TERRITORIO = {
    "PREC": ("Precipitación", False), "Tmin": ("Temperatura mínima", False),
    "Tmax": ("Temperatura máxima", False), "ELEVACION": ("Elevación", True),
    "PENDIENTE": ("Pendiente", True),
}


def construir_territorio() -> dict:
    """Clima y relieve por parcela en formato compacto, tal como los muestra el mapa del extractor."""
    ruta = RAIZ / "salidas_extractor" / "datos_por_imagen.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Ejecuta extraccion.py para generarlo.")
    tabla = pd.read_csv(ruta, encoding="utf-8-sig", usecols=[
        "ID_POLIGONO", "variable", "anio", "mes", "valor", "unidad", "estado_extraccion",
        "cobertura_valida_pct"])
    desconocidas = set(tabla["variable"]) - set(NOMBRES_TERRITORIO)
    if desconocidas:
        raise ValueError(f"Variables sin nombre definido en NOMBRES_TERRITORIO: {sorted(desconocidas)}")
    orden = [v for v in NOMBRES_TERRITORIO if v in set(tabla["variable"])]
    ids = sorted(tabla["ID_POLIGONO"].unique())
    indice_id = {pid: i for i, pid in enumerate(ids)}
    indice_var = {v: i for i, v in enumerate(orden)}
    unidades = tabla.groupby("variable")["unidad"].first().to_dict()
    registros = []
    for fila in tabla.itertuples(index=False):
        valor = None if pd.isna(fila.valor) else round(float(fila.valor), 3)
        cobertura = None if pd.isna(fila.cobertura_valida_pct) else round(float(fila.cobertura_valida_pct), 1)
        registros.append([indice_var[fila.variable],
                          None if pd.isna(fila.anio) else int(fila.anio),
                          None if pd.isna(fila.mes) else int(fila.mes),
                          indice_id[fila.ID_POLIGONO], valor, cobertura,
                          0 if fila.estado_extraccion == "OK" else 1])
    anios = sorted({r[1] for r in registros if r[1] is not None})
    return {
        "variables": [{"id": v, "nombre": NOMBRES_TERRITORIO[v][0], "unidad": unidades[v],
                       "estatica": NOMBRES_TERRITORIO[v][1]} for v in orden],
        "ids": ids,
        "anios": anios,
        "registros": registros,
        "fuente": "salidas_extractor/datos_por_imagen.csv",
    }


def pagina_satelital(html: str) -> str:
    """Deriva del panel principal la versión con fondo cartográfico y capas del territorio."""
    config = {"fondo": True, "fondoInicial": "satelite", "territorio": True, "buscador": True}
    reemplazos = [
        ("<title>Panel AgroCebada</title>", "<title>Panel AgroCebada Satelital</title>"),
        ('            <div class="grupo-control"><span class="etiqueta">Color según</span><div class="seg" id="f-color"></div></div>\n',
         '            <div class="grupo-control"><label class="etiqueta" for="f-capa">Capa</label><select class="selector" id="f-capa"></select></div>\n'
         '            <div class="grupo-control" id="g-anio"><label class="etiqueta" for="f-anio">Año</label><select class="selector" id="f-anio"></select></div>\n'
         '            <div class="grupo-control" id="g-mes"><label class="etiqueta" for="f-mes">Mes</label><select class="selector" id="f-mes"></select></div>\n'
         '            <div class="grupo-control"><label class="etiqueta" for="f-fondo">Fondo del mapa</label><select class="selector" id="f-fondo">'
         '<option value="satelite">Satélite con nombres</option><option value="calles">Calles y localidades</option>'
         '<option value="ninguno">Sin fondo</option></select></div>\n'
         '            <div class="grupo-control"><label class="etiqueta" for="f-buscar">Buscar parcela</label>'
         '<input class="selector" id="f-buscar" type="search" list="ids-mapa" placeholder="AGC_044" autocomplete="off" aria-label="Identificador de parcela">'
         '<datalist id="ids-mapa"></datalist></div>\n'
         '            <div class="grupo-control"><span class="etiqueta">Capa actual</span><button class="boton" id="descargar-capa" type="button">Descargar CSV</button></div>\n'),
        ("contorno oscuro: por estimar", "contorno grueso: por estimar"),
        ('<script src="datos.js"></script>\n',
         f'<script>window.PANEL_CONFIG={json.dumps(config)};</script>\n<script src="datos.js"></script>\n<script src="territorio.js"></script>\n'),
    ]
    for antes, despues in reemplazos:
        if html.count(antes) != 1:
            raise ValueError(f"index.html cambió: no se encontró exactamente una vez: {antes[:70]!r}")
        html = html.replace(antes, despues)
    return html


def incrustar(html: str, archivos: list[str]) -> str:
    """Convierte una página con archivos vecinos en un único HTML autocontenido."""
    css = (DASH / "estilos.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="estilos.css">', f"<style>\n{css}\n</style>")
    for archivo in archivos:
        codigo = (DASH / archivo).read_text(encoding="utf-8").replace("</script", "<\\/script")
        html, n = re.subn(rf'<script src="{re.escape(archivo)}"></script>',
                          lambda _m, c=codigo: f"<script>\n{c}\n</script>", html)
        if n != 1:
            raise ValueError(f"No se encontró la referencia a {archivo}.")
    return html


def escribir(datos: dict, autonomo: bool) -> None:
    contenido = json.dumps(datos, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (DASH / "datos.js").write_text(f"window.DATOS={contenido};\n", encoding="utf-8")
    print(f"datos.js: {len(contenido) / 1024:.0f} KB, {len(datos['parcelas'])} parcelas")
    territorio = construir_territorio()
    cuerpo = json.dumps(territorio, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (DASH / "territorio.js").write_text(f"window.TERRITORIO={cuerpo};\n", encoding="utf-8")
    print(f"territorio.js: {len(cuerpo) / 1024:.0f} KB, {len(territorio['registros'])} registros")
    principal = (DASH / "index.html").read_text(encoding="utf-8")
    satelital = pagina_satelital(principal)
    (DASH / "panel_mapa_satelital.html").write_text(satelital, encoding="utf-8")
    if not autonomo:
        return
    html = incrustar(principal, ["datos.js", "modelo.js", "app.js"])
    (DASH / "panel_autonomo.html").write_text(html, encoding="utf-8")
    print(f"panel_autonomo.html: {len(html) / 1024:.0f} KB")
    html = incrustar(satelital, ["datos.js", "territorio.js", "modelo.js", "app.js"])
    (DASH / "panel_mapa_satelital_autonomo.html").write_text(html, encoding="utf-8")
    print(f"panel_mapa_satelital_autonomo.html: {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--autonomo", action="store_true", help="Escribir también panel_autonomo.html.")
    argumentos = parser.parse_args()
    escribir(construir(), argumentos.autonomo)
