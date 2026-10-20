#!/usr/bin/env python3
"""Extrae GeoTIFF por parcela, reproyecta geometrías y exporta CSV y mapa.

Ejecución: .venv/bin/python extraccion.py
Los archivos originales se leen; las salidas se escriben en salidas_extractor/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import rasterio
    import shapefile
    from pyproj import CRS, Transformer
    from shapely import force_2d, make_valid, union_all
    from shapely.geometry import Polygon, mapping, shape
    from shapely.ops import transform
except ImportError as error:
    raise SystemExit(
        f"Falta una dependencia: {error}. Instala requirements_extractor.txt "
        "y ejecuta con el Python del entorno virtual."
    ) from error

HERE = Path(__file__).resolve().parent
WGS84 = CRS.from_epsg(4326)
CLIMA = {"PREC": "mm/mes", "Tmin": "°C", "Tmax": "°C"}
BASE_FIELDS = ["ID_POLIGONO", "AREA_HA", "CONJUNTO", "RENDIMIENTO_T_HA",
               "Municipio", "Estado", "longitud", "latitud"]
LONG_FIELDS = BASE_FIELDS + [
    "capa", "archivo", "variable", "anio", "mes", "banda", "unidad",
    "valor", "min", "max", "std", "pixeles_intersectados", "pixeles_validos",
    "cobertura_valida_pct", "metodo", "estado_extraccion", "crs_raster",
    "x_muestreo", "y_muestreo", "resolucion_x", "resolucion_y",
]


@dataclass
class Parcela:
    propiedades: dict
    geometria: object  # Polygon/MultiPolygon en WGS84
    punto: object      # Punto interior en WGS84, para el método puntual


@dataclass
class PlanZonal:
    filas: np.ndarray
    columnas: np.ndarray
    areas: np.ndarray
    area_parcela: float


def normalizar(nombre):
    texto = unicodedata.normalize("NFKD", nombre)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()


def numero(valor):
    if valor is None or str(valor).strip() == "":
        return None
    resultado = float(valor)
    if not math.isfinite(resultado):
        raise ValueError(f"Número no finito: {valor}")
    return resultado


def escribir_csv(destino, registros, campos):
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(registros)


def escribir_json(destino, contenido):
    destino.write_text(json.dumps(contenido, ensure_ascii=False, indent=2,
                                  allow_nan=False), encoding="utf-8")


def convertir(geometria, origen, destino):
    """Mantiene siempre el orden x/y: longitud/latitud o este/norte."""
    if origen.equals(destino):
        return geometria
    conversor = Transformer.from_crs(origen, destino, always_xy=True)
    return transform(conversor.transform, geometria)


def solo_poligonos(geometria):
    geometria = force_2d(geometria)
    if not geometria.is_valid:
        geometria = make_valid(geometria)
    if geometria.geom_type == "GeometryCollection":
        geometria = union_all([g for g in geometria.geoms
                               if g.geom_type in ("Polygon", "MultiPolygon")])
    if geometria.is_empty or geometria.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("La geometría no contiene una superficie poligonal válida.")
    return geometria


def cargar_parcelas(shp, etiquetas):
    prj = shp.with_suffix(".prj")
    if not prj.exists():
        raise ValueError(f"Falta {prj.name}; no se puede asumir el sistema de coordenadas.")
    crs_origen = CRS.from_wkt(prj.read_text(encoding="utf-8"))
    datos_objetivo = {}
    if etiquetas is not None:
        with etiquetas.open(encoding="utf-8-sig", newline="") as archivo:
            for registro in csv.DictReader(archivo):
                identificador = registro["ID_POLIGONO"].strip()
                if identificador in datos_objetivo:
                    raise ValueError(f"ID duplicado en etiquetas: {identificador}")
                datos_objetivo[identificador] = registro
    cpg = shp.with_suffix(".cpg")
    encoding = cpg.read_text().strip() if cpg.exists() else "utf-8"
    if encoding == "65001":
        encoding = "utf-8"
    parcelas, ids = [], set()
    with shapefile.Reader(str(shp), encoding=encoding) as lector:
        campos = [c[0] for c in lector.fields[1:]]
        for registro in lector.iterShapeRecords():
            atributos = dict(zip(campos, registro.record))
            campos_norm = {normalizar(k): v for k, v in atributos.items()}
            identificador = str(campos_norm.get("id_poligono", campos_norm.get("id_poligon", ""))).strip()
            if not identificador or identificador in ids:
                raise ValueError(f"ID ausente o repetido en Shapefile: {identificador}")
            ids.add(identificador)
            geometria = solo_poligonos(shape(registro.shape.__geo_interface__))
            geometria = convertir(geometria, crs_origen, WGS84)
            objetivo = datos_objetivo.get(identificador)
            if etiquetas is not None and objetivo is None:
                raise ValueError(f"{identificador} no aparece en el CSV de etiquetas.")
            objetivo = objetivo or {}
            area = numero(objetivo.get("AREA_HA", campos_norm.get("area_ha")))
            conjunto = objetivo.get("CONJUNTO", campos_norm.get("conjunto", ""))
            if "area_ha" in campos_norm and area is not None:
                if not math.isclose(area, numero(campos_norm["area_ha"]), abs_tol=1e-5):
                    raise ValueError(f"Área discordante SHP/CSV: {identificador}")
            if "conjunto" in campos_norm and conjunto != campos_norm["conjunto"]:
                raise ValueError(f"CONJUNTO discordante SHP/CSV: {identificador}")
            centro = geometria.centroid
            punto = geometria.representative_point()
            propiedades = {
                "ID_POLIGONO": identificador, "AREA_HA": area,
                "CONJUNTO": conjunto,
                "RENDIMIENTO_T_HA": numero(objetivo.get("RENDIMIENTO_T_HA")),
                "Municipio": str(campos_norm.get("municipio", "")),
                "Estado": str(campos_norm.get("estado", "")),
                "Cultivo": str(campos_norm.get("cultivo", "")),
                "longitud": centro.x, "latitud": centro.y,
                "longitud_muestreo": punto.x, "latitud_muestreo": punto.y,
            }
            parcelas.append(Parcela(propiedades, geometria, punto))
    if etiquetas is not None and ids != set(datos_objetivo):
        raise ValueError("Los IDs del Shapefile y del CSV de etiquetas no coinciden.")
    parcelas.sort(key=lambda p: p.propiedades["ID_POLIGONO"])
    return parcelas, crs_origen


def identificar_raster(ruta):
    coincidencia = re.fullmatch(r"(PREC|Tmin|Tmax)_(\d{4})_(\d{2})", ruta.stem, re.I)
    if coincidencia:
        raw, anio, mes = coincidencia.groups()
        variable = {"prec": "PREC", "tmin": "Tmin", "tmax": "Tmax"}[raw.lower()]
        mes = int(mes)
        if not 1 <= mes <= 12:
            raise ValueError(f"Mes inválido en {ruta.name}")
        return variable, int(anio), mes, CLIMA[variable]
    if "elevacion" in normalizar(ruta.stem):
        return "ELEVACION", None, None, "m"
    if "pendiente" in normalizar(ruta.stem):
        return "PENDIENTE", None, None, "grados"
    return ruta.stem, None, None, None


def descubrir_rasters(dataset, imagenes, anio):
    rutas = []
    for entrada in (imagenes or [dataset]):
        ruta = Path(entrada).expanduser().resolve()
        if ruta.is_dir():
            rutas.extend(p for p in ruta.rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
        elif ruta.is_file() and ruta.suffix.lower() in (".tif", ".tiff"):
            rutas.append(ruta)
        else:
            raise ValueError(f"No es un TIFF ni una carpeta de TIFF: {ruta}")
    rutas = sorted(set(p.resolve() for p in rutas))
    if anio is not None:
        rutas = [p for p in rutas if identificar_raster(p)[1] in (None, anio)]
    if not rutas:
        raise ValueError("No se encontraron archivos TIFF para extraer.")
    return rutas


def clave_grilla(src):
    return src.crs.to_wkt(), tuple(src.transform), src.width, src.height


def preparar_plan(geometria, src, a_area):
    """Intersecciones fraccionarias; áreas en una proyección equivalente métrica.

    No usa selección de centros: una parcela diminuta dentro de una celda
    conserva su superficie y recibe un valor. Funciona con grillas rotadas.
    """
    xmin, ymin, xmax, ymax = geometria.bounds
    inversa = ~src.transform
    limites = [inversa * p for p in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))]
    c0 = max(0, math.floor(min(p[0] for p in limites)))
    c1 = min(src.width, math.ceil(max(p[0] for p in limites)))
    r0 = max(0, math.floor(min(p[1] for p in limites)))
    r1 = min(src.height, math.ceil(max(p[1] for p in limites)))
    total = transform(a_area.transform, geometria).area
    if total <= 0:
        raise ValueError("Parcela con área proyectada nula.")
    filas, columnas, areas = [], [], []
    for fila in range(r0, r1):
        for columna in range(c0, c1):
            celda = Polygon([src.transform * p for p in
                            ((columna, fila), (columna + 1, fila),
                             (columna + 1, fila + 1), (columna, fila + 1))])
            interseccion = geometria.intersection(celda)
            if interseccion.is_empty:
                continue
            area = transform(a_area.transform, interseccion).area
            if area > total * 1e-12:
                filas.append(fila)
                columnas.append(columna)
                areas.append(area)
    return PlanZonal(np.asarray(filas, dtype=int), np.asarray(columnas, dtype=int),
                     np.asarray(areas, dtype=float), total)


def estadisticas_zonales(valores, plan):
    n = len(plan.areas)
    resultado = dict(valor=None, min=None, max=None, std=None,
                     pixeles_intersectados=n, pixeles_validos=0,
                     cobertura_valida_pct=0.0, estado_extraccion="SIN_COBERTURA")
    if n == 0:
        return resultado
    v = valores[plan.filas, plan.columnas]
    valido = np.isfinite(v)
    if not valido.any():
        resultado["estado_extraccion"] = "SIN_DATOS_VALIDOS"
        return resultado
    v, pesos = v[valido], plan.areas[valido]
    media = float(np.average(v, weights=pesos))
    cobertura = min(100.0, max(0.0, float(pesos.sum() / plan.area_parcela * 100)))
    resultado.update(valor=media, min=float(v.min()), max=float(v.max()),
                     std=float(np.sqrt(np.average((v - media) ** 2, weights=pesos))),
                     pixeles_validos=len(v), cobertura_valida_pct=cobertura,
                     estado_extraccion="OK" if cobertura >= 99.99 else "COBERTURA_PARCIAL")
    return resultado


def estadisticas_puntuales(valores, punto, src):
    fila, columna = src.index(punto.x, punto.y)
    resultado = dict(valor=None, min=None, max=None, std=None,
                     pixeles_intersectados=0, pixeles_validos=0,
                     cobertura_valida_pct=None, estado_extraccion="SIN_COBERTURA")
    if not (0 <= fila < src.height and 0 <= columna < src.width):
        return resultado
    resultado["pixeles_intersectados"] = 1
    valor = float(valores[fila, columna])
    if not math.isfinite(valor):
        resultado["estado_extraccion"] = "SIN_DATOS_VALIDOS"
        return resultado
    resultado.update(valor=valor, min=valor, max=valor, std=0.0,
                     pixeles_validos=1, estado_extraccion="OK")
    return resultado


def extraer(parcelas, rutas, dataset, salida, metodo):
    centro = union_all([p.geometria for p in parcelas]).centroid
    crs_area = CRS.from_proj4(f"+proj=laea +lat_0={centro.y} +lon_0={centro.x} "
                             "+datum=WGS84 +units=m +no_defs")
    cache, coordenadas, crs_vistos = {}, [], set()
    registros, inventario, capas_vistas = [], [], set()
    for numero_archivo, ruta in enumerate(rutas, 1):
        variable, anio, mes, unidad = identificar_raster(ruta)
        with rasterio.open(ruta) as src:
            if src.crs is None:
                raise ValueError(f"{ruta.name} no declara CRS; no se puede geolocalizar.")
            crs = CRS.from_user_input(src.crs)
            clave = clave_grilla(src)
            if clave not in cache:
                a_area = Transformer.from_crs(crs, crs_area, always_xy=True)
                geometrias = [convertir(p.geometria, WGS84, crs) for p in parcelas]
                puntos = [convertir(p.punto, WGS84, crs) for p in parcelas]
                planes = [preparar_plan(g, src, a_area) for g in geometrias] if metodo == "zonal" else []
                cache[clave] = puntos, planes
            puntos, planes = cache[clave]
            if crs.to_wkt() not in crs_vistos:
                crs_vistos.add(crs.to_wkt())
                for parcela, punto in zip(parcelas, puntos):
                    centroide = convertir(parcela.geometria.centroid, WGS84, crs)
                    coordenadas.append({"ID_POLIGONO": parcela.propiedades["ID_POLIGONO"],
                                        "crs": crs.to_string(), "x_centroide": centroide.x,
                                        "y_centroide": centroide.y, "x_muestreo": punto.x,
                                        "y_muestreo": punto.y,
                                        "unidad_coordenadas": crs.axis_info[0].unit_name})
            try:
                archivo_relativo = str(ruta.relative_to(dataset))
            except ValueError:
                archivo_relativo = str(ruta)
            for banda in range(1, src.count + 1):
                capa = ruta.stem if src.count == 1 else f"{ruta.stem}_b{banda}"
                if capa in capas_vistas:
                    raise ValueError(f"Nombre de capa repetido: {capa}; evita TIFF duplicados.")
                capas_vistas.add(capa)
                datos = src.read(banda, masked=True).astype("float64").filled(np.nan)
                escala, offset = float(src.scales[banda - 1]), float(src.offsets[banda - 1])
                datos = datos * escala + offset
                unidad_banda = unidad or src.units[banda - 1] or "sin unidad documentada"
                registros_capa = []
                for i, (parcela, punto) in enumerate(zip(parcelas, puntos)):
                    stats = (estadisticas_zonales(datos, planes[i]) if metodo == "zonal"
                             else estadisticas_puntuales(datos, punto, src))
                    registro = dict(parcela.propiedades, capa=capa, archivo=archivo_relativo,
                                    variable=variable, anio=anio, mes=mes, banda=banda,
                                    unidad=unidad_banda, metodo=("MEDIA_PONDERADA_AREA" if metodo == "zonal"
                                                               else "PIXEL_PUNTO_INTERIOR"),
                                    crs_raster=crs.to_string(), x_muestreo=punto.x,
                                    y_muestreo=punto.y, resolucion_x=src.res[0], resolucion_y=src.res[1], **stats)
                    registros_capa.append(registro)
                registros.extend(registros_capa)
                escribir_csv(salida / "por_imagen" / f"{capa}.csv", registros_capa, LONG_FIELDS)
                inventario.append({"capa": capa, "archivo": archivo_relativo, "variable": variable,
                                   "anio": anio, "mes": mes, "banda": banda, "unidad": unidad_banda,
                                   "crs": crs.to_string(), "crs_wkt": crs.to_wkt(),
                                   "ancho": src.width, "alto": src.height, "tipo": src.dtypes[banda - 1],
                                   "nodata": src.nodatavals[banda - 1], "escala": escala, "offset": offset,
                                   "resolucion_x": src.res[0], "resolucion_y": src.res[1],
                                   "unidad_coordenadas": crs.axis_info[0].unit_name,
                                   "transformacion": list(src.transform)[:6],
                                   "parcelas_con_valor": sum(r["valor"] is not None for r in registros_capa)})
        if numero_archivo == 1 or numero_archivo % 12 == 0 or numero_archivo == len(rutas):
            print(f"  TIFF {numero_archivo}/{len(rutas)}: {ruta.name}", flush=True)
    return registros, inventario, coordenadas, crs_area


def tablas_resumen(parcelas, registros):
    propiedades = {p.propiedades["ID_POLIGONO"]: p.propiedades for p in parcelas}
    mensual, topografia = {}, {identificador: dict(p) for identificador, p in propiedades.items()}
    ancho = {identificador: dict(p) for identificador, p in propiedades.items()}
    campos_capas, variables_topo = [], []
    for r in registros:
        identificador = r["ID_POLIGONO"]
        columna = r["capa"]
        if columna not in campos_capas:
            campos_capas.append(columna)
        ancho[identificador][columna] = r["valor"]
        if r["variable"] in CLIMA and r["banda"] == 1:
            clave = identificador, r["anio"], r["mes"]
            fila = mensual.setdefault(clave, dict(propiedades[identificador], anio=r["anio"], mes=r["mes"]))
            nombre = {"PREC": "precipitacion_mm", "Tmin": "tmin_promedio_c", "Tmax": "tmax_promedio_c"}[r["variable"]]
            fila[nombre] = r["valor"]
            fila[f"{nombre}_cobertura_pct"] = r["cobertura_valida_pct"]
        elif r["variable"] in ("ELEVACION", "PENDIENTE") and r["banda"] == 1:
            nombre = "elevacion_m" if r["variable"] == "ELEVACION" else "pendiente_grados"
            if nombre not in variables_topo:
                variables_topo.append(nombre)
            for sufijo, fuente in (("", "valor"), ("_min", "min"), ("_max", "max"), ("_std", "std"),
                                   ("_cobertura_pct", "cobertura_valida_pct")):
                topografia[identificador][nombre + sufijo] = r[fuente]
    for r in mensual.values():
        tmin, tmax = r.get("tmin_promedio_c"), r.get("tmax_promedio_c")
        r["temperatura_media_aprox_c"] = None if tmin is None or tmax is None else (tmin + tmax) / 2
        r["amplitud_termica_promedio_c"] = None if tmin is None or tmax is None else tmax - tmin
    campos_clima = BASE_FIELDS + ["anio", "mes", "precipitacion_mm", "tmin_promedio_c", "tmax_promedio_c",
                                 "temperatura_media_aprox_c", "amplitud_termica_promedio_c",
                                 "precipitacion_mm_cobertura_pct", "tmin_promedio_c_cobertura_pct",
                                 "tmax_promedio_c_cobertura_pct"]
    campos_topo = BASE_FIELDS + [v + sufijo for v in variables_topo
                               for sufijo in ("", "_min", "_max", "_std", "_cobertura_pct")]
    return ([mensual[k] for k in sorted(mensual)], campos_clima,
            list(topografia.values()), campos_topo, list(ancho.values()), BASE_FIELDS + campos_capas)


def _json_seguro(valor: object) -> str:
    """Evita cerrar el elemento script con contenido de los datos."""
    def limpiar(item: object) -> object:
        if isinstance(item, dict):
            return {str(k): limpiar(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [limpiar(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    contenido = json.dumps(limpiar(valor), ensure_ascii=False, allow_nan=False,
                           separators=(",", ":"))
    return (contenido.replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def escribir_mapa(destino: Path, geojson: dict, datos: list[dict]) -> None:
    """Guarda un mapa portátil con polígonos, registros y descarga de CSV.

    ``geojson`` debe ser una FeatureCollection en longitud/latitud WGS84.
    Cada Feature debe tener ID_POLIGONO en properties. ``datos`` contiene
    ID_POLIGONO, variable, anio, mes, valor, unidad y estado_extraccion;
    opcionalmente, metodo y cobertura_valida_pct.
    Las geometrías admitidas son Polygon y MultiPolygon, incluidos sus huecos.
    """
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("El mapa requiere un GeoJSON FeatureCollection en EPSG:4326.")
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    inicio, _, resto = _HTML.partition("__GEOJSON_JSON__")
    medio, _, final = resto.partition("__REGISTROS_JSON__")
    html = inicio + _json_seguro(geojson) + medio + _json_seguro(datos) + final
    destino.write_text(html, encoding="utf-8")


_HTML = r'''<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agrocebada · Datos por parcela</title>
<style>
:root{color-scheme:light;--ink:#18352d;--muted:#66796f;--line:#d9e2da;--paper:#f3f5ee;--green:#24634a;--gold:#ddb046}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:22px 28px 19px;background:#173e31;color:#fff;display:flex;justify-content:space-between;gap:24px;align-items:center}
h1{font-size:25px;letter-spacing:-.6px;margin:0 0 3px}header p{margin:0;color:#d2e1d2;font-size:13px}.badge{border:1px solid #ffffff35;border-radius:24px;padding:7px 13px;white-space:nowrap;font-size:12px}
main{padding:20px 24px 28px;max-width:1800px;margin:auto}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-bottom:16px}
label{display:block;font-size:12px;font-weight:650;color:var(--muted);margin-bottom:5px}select,input,button{font:inherit;color:var(--ink)}select,input{border:1px solid #bdcfc1;border-radius:7px;background:#fff;padding:9px 11px;min-height:40px}select:disabled{color:#67746d;background:#e9eee7}.controls select{min-width:115px}#variable{min-width:230px}
button{border:1px solid #b8cbbc;background:#fff;padding:9px 14px;border-radius:7px;cursor:pointer;font-size:13px;font-weight:600}button:hover{background:#e8eee5}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #afc79d;outline-offset:2px}button.primary{background:var(--green);color:#fff;border-color:var(--green)}button.primary:hover{background:#1b4c37}
.spacer{flex:1}.summary{font-size:12px;color:var(--muted);max-width:440px;align-self:center}.layout{display:grid;grid-template-columns:minmax(0,1fr) 345px;gap:18px}.card{background:#fff;border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:0 3px 14px #18352d05}
.map-heading{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:14px 17px;border-bottom:1px solid var(--line)}.map-heading h2{font-size:16px;margin:0}.map-heading span{font-size:12px;color:var(--muted)}
.canvas{position:relative;height:590px;background:#eef2e9;overflow:hidden}#mapa{width:100%;height:100%;display:block;touch-action:none;cursor:grab}#mapa.dragging{cursor:grabbing}.parcel{cursor:pointer;stroke:#17442b;stroke-opacity:.55;stroke-width:.8;vector-effect:non-scaling-stroke;fill-rule:evenodd}.parcel:hover{stroke:#101b16;stroke-width:1.6}.parcel.selected{stroke:#a36a0d;stroke-width:2.5;stroke-opacity:1}.parcel-missing{stroke-dasharray:2 2}.marker{stroke:#fff;stroke-width:.8;vector-effect:non-scaling-stroke;cursor:pointer}.marker.selected{stroke:#a36a0d;stroke-width:2.5}
.map-buttons{position:absolute;top:14px;left:14px;display:grid;gap:6px}.map-buttons button{width:36px;height:36px;padding:0;background:#fffffff0;box-shadow:0 2px 5px #173e3110;font-size:22px}.map-buttons #encuadrar{width:auto;font-size:11px;padding:0 9px}.map-help{position:absolute;left:15px;bottom:15px;background:#ffffffeb;padding:7px 10px;border-radius:6px;pointer-events:none;font-size:11px;color:#617467;max-width:280px}.map-geo{position:absolute;right:14px;top:14px;border-radius:6px;background:#ffffffdb;font-size:10px;color:#617467;padding:6px 9px;pointer-events:none}
#tooltip{position:absolute;display:none;pointer-events:none;background:#173e31;color:#fff;padding:7px 10px;border-radius:7px;font-size:12px;max-width:250px;white-space:pre-line;box-shadow:0 3px 12px #0002;z-index:5}
.legend{padding:13px 17px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:12px}.legend-scale{width:210px;max-width:100%}.gradient{height:9px;border-radius:9px;background:linear-gradient(90deg,#e5f0c4,#80b674,#257b5b,#114b43)}.range{display:flex;justify-content:space-between;gap:15px;margin-top:4px;font-variant-numeric:tabular-nums;color:#456050}.empty-dot{display:inline-block;width:10px;height:10px;border-radius:2px;background:#b9c3b6;vertical-align:middle;margin-right:5px}.note{font-size:12px;color:var(--muted);margin:0;padding:0 17px 15px;max-width:920px}.side{padding:19px}.side h2{font-size:17px;margin:0 0 13px}.search-row{display:flex;gap:6px}.search-row input{width:100%;min-width:0}.search-row button{padding:8px 10px}.search-feedback{font-size:12px;color:#7d4c18;min-height:18px;margin:5px 0 9px}.hint{color:var(--muted);font-size:13px}.parcel-id{font-size:23px;letter-spacing:-.6px;font-weight:700;margin:16px 0 4px}.parcel-location{font-size:12px;color:var(--muted);margin-bottom:15px}.metric{background:#edf3e5;border:1px solid #d8e4cc;padding:15px;border-radius:10px;margin-bottom:16px}.metric-label{font-size:12px;color:#527047}.metric-value{font-size:27px;font-weight:650;letter-spacing:-.6px;margin:2px 0}.metric-detail{font-size:11px;color:#627360;overflow-wrap:anywhere}
dl{display:grid;grid-template-columns:1fr auto;gap:8px 16px;margin:0 0 15px;font-size:12px}dt{color:var(--muted)}dd{margin:0;text-align:right;overflow-wrap:anywhere}#coordenadas{font-variant-numeric:tabular-nums}.section-title{border-top:1px solid var(--line);margin:16px 0 8px;padding-top:14px;font-size:12px;font-weight:700}.layer-table{border-collapse:collapse;width:100%;font-size:12px}.layer-table td{border-bottom:1px solid #edf1e9;padding:7px 0;vertical-align:top}.layer-table td:last-child{text-align:right;padding-left:10px;font-variant-numeric:tabular-nums}.mini-note{font-size:11px;color:var(--muted);margin:10px 0 0}.empty{font-size:13px;color:var(--muted);padding:8px 0 20px}.footer{font-size:11px;color:var(--muted);margin-top:14px}#sin-geometrias{display:none;position:absolute;inset:35%;text-align:center;color:var(--muted)}
@media(max-width:1000px){.layout{grid-template-columns:minmax(0,1fr) 300px}.canvas{height:510px}main{padding:15px}.summary{max-width:320px}header{padding:18px 20px}}
@media(max-width:760px){header{align-items:start}.badge{display:none}.layout{grid-template-columns:1fr}.canvas{height:480px}.controls{gap:9px}.controls select{min-width:95px}#variable{min-width:200px}.summary{order:5;max-width:none;width:100%}.spacer{display:none}.side{padding:18px}h1{font-size:23px}}
.metric-warning{font-size:12px;color:#78521e;background:#fff4d4;border:1px solid #e5cb85;border-radius:7px;padding:9px 10px;margin:10px 0 0}.metric-warning[hidden]{display:none}
#fondo-mapa{position:absolute;inset:0;pointer-events:none}#mapa{position:relative;z-index:1}.map-buttons,.map-geo,.map-help{z-index:2}.map-attribution{position:absolute;bottom:0;right:0;z-index:3;padding:3px 7px;background:#ffffffed;font-size:11px;color:#456050;max-width:65%;text-align:right}.map-attribution a{color:#24593f}.parcel{fill-opacity:.68}.map-geo{max-width:55%;text-align:right}
</style>
</head>
<body>
<header><div><h1>Datos del territorio</h1><p>Agrocebada · Clima y relieve por parcela</p></div><div class="badge">Parcelas locales · Fondo en línea</div></header>
<main>
<div class="controls">
 <div><label for="variable">Variable</label><select id="variable"></select></div>
 <div><label for="anio">Año</label><select id="anio"></select></div>
 <div><label for="mes">Mes</label><select id="mes"></select></div>
 <div><label for="fondo">Fondo del mapa</label><select id="fondo"><option value="calles">Calles y localidades</option><option value="ninguno">Sin fondo</option></select></div>
 <div class="spacer"></div><div class="summary" id="resumen"></div>
 <button class="primary" id="csv">Descargar CSV de esta capa</button>
</div>
<div class="layout">
 <section class="card">
  <div class="map-heading"><h2 id="titulo-capa">Valores por parcela</h2><span id="numero-parcelas"></span></div>
  <div class="canvas" id="canvas">
   <div id="fondo-mapa" aria-hidden="true"></div>
   <svg id="mapa" role="img" aria-label="Mapa de las parcelas. Busca una parcela para consultar sus datos." xmlns="http://www.w3.org/2000/svg"><g id="cuadricula"></g><g id="poligonos"></g><g id="puntos"></g><g id="escala"></g></svg>
   <div class="map-buttons"><button id="zoom-mas" aria-label="Acercar">+</button><button id="zoom-menos" aria-label="Alejar">−</button><button id="encuadrar">Ver todas</button></div>
   <div class="map-geo" id="fondo-estado" role="status">Cargando fondo cartográfico…</div>
   <div class="map-attribution" id="fondo-creditos"><a href="https://openfreemap.org/" target="_blank" rel="noopener noreferrer">OpenFreeMap</a> · <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener noreferrer">© OpenMapTiles</a> · <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">© OpenStreetMap</a></div>
   <div class="map-help">Arrastra para mover · Rueda para acercar<br>Haz clic en una parcela o en su punto</div>
   <div id="tooltip"></div><div id="sin-geometrias">No hay geometrías válidas para mostrar.</div>
  </div>
  <div class="legend"><div class="legend-scale"><div class="gradient" id="gradiente"></div><div class="range"><span id="minimo"></span><span id="maximo"></span></div></div><span id="unidad-leyenda"></span><span><i class="empty-dot"></i>Sin dato válido</span><span id="cuenta-datos"></span></div>
  <p class="note" id="nota-resolucion"></p>
 </section>
 <aside class="card side">
  <h2>Consulta una parcela</h2>
  <div class="search-row"><input id="buscar" type="search" placeholder="Ej. AGC_044" list="ids" autocomplete="off" aria-label="Identificador de parcela"><datalist id="ids"></datalist><button id="buscar-boton">Ver</button></div>
  <p id="resultado-busqueda" class="search-feedback" role="status"></p>
  <div id="sin-seleccion" class="empty">Selecciona una parcela en el mapa o búscala por su identificador.</div>
  <div id="ficha" hidden>
   <div class="parcel-id" id="id-parcela"></div><div class="parcel-location" id="ubicacion"></div>
   <div class="metric"><div class="metric-label" id="etiqueta-valor"></div><div class="metric-value" id="valor-parcela"></div><div class="metric-detail" id="estado-valor"></div><div class="metric-detail" id="metodo-valor"></div><div class="metric-warning" id="alerta-cobertura" hidden></div></div>
   <dl><dt>Superficie reportada</dt><dd id="area"></dd><dt>Conjunto</dt><dd id="conjunto"></dd><dt>Rendimiento conocido</dt><dd id="rendimiento"></dd><dt>Longitud, latitud*</dt><dd id="coordenadas"></dd></dl>
   <button id="centrar-parcela">Acercar a esta parcela</button>
   <div class="section-title" id="titulo-tabla">Datos de la fecha seleccionada</div><table class="layer-table" aria-label="Valores de otras variables"><tbody id="tabla"></tbody></table>
   <p class="mini-note">* Centroide de la parcela en WGS84. Las coordenadas geográficas se muestran en grados y se redondean para lectura.</p>
  </div>
 </aside>
</div>
<p class="footer">Los colores representan valores extraídos de los TIFF, no predicciones de rendimiento. Cada descarga contiene una fila por parcela para la capa seleccionada. El fondo de calles y localidades requiere internet; las parcelas y sus datos están incluidos en este archivo.</p>
</main>
<script id="geojson" type="application/json">__GEOJSON_JSON__</script>
<script id="registros" type="application/json">__REGISTROS_JSON__</script>
<script>
"use strict";
(() => {
const GEO = JSON.parse(document.getElementById("geojson").textContent);
const DATOS = JSON.parse(document.getElementById("registros").textContent);
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg", R = 6378137;
const meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"];
const etiquetas = {PREC:"Precipitación",Tmin:"Temperatura mínima",Tmax:"Temperatura máxima",ELEVACION:"Elevación",PENDIENTE:"Pendiente"};
const unidades = {PREC:"mm",Tmin:"°C",Tmax:"°C",ELEVACION:"m",PENDIENTE:"°"};
const paletas = {PREC:["#e5f0c4","#80b674","#257b5b","#114b43"],Tmin:["#dbeaf1","#85b1c8","#4676a7","#283f74"],Tmax:["#fff1ca","#e6be69","#c8783c","#8c3a27"],ELEVACION:["#e8f0d0","#b5c592","#688f7d","#315952"],PENDIENTE:["#efdef2","#c49dbd","#8f669d","#523c77"]};
const fmt = (v,d=2) => v===null||v===undefined||v===""||!Number.isFinite(Number(v))?"Sin dato":new Intl.NumberFormat("es-MX",{maximumFractionDigits:d}).format(Number(v));
const numero = v => v!==null&&v!==undefined&&v!==""&&Number.isFinite(Number(v))?Number(v):null;
const fechaKey = (a,m) => `${a??""}|${m??""}`;
const capaKey = (v,a,m) => `${v}|${fechaKey(a,m)}`;
const capas = new Map(), fechas = new Map();
for (const registro of DATOS) {
 const v=String(registro.variable), a=numero(registro.anio), m=numero(registro.mes), k=capaKey(v,a,m);
 if(!capas.has(k))capas.set(k,new Map());
 capas.get(k).set(String(registro.ID_POLIGONO),registro);
 if(!fechas.has(v))fechas.set(v,new Map());
 fechas.get(v).set(fechaKey(a,m),{a,m});
}
const variables=[...fechas.keys()].sort((a,b)=>["PREC","Tmin","Tmax","ELEVACION","PENDIENTE"].indexOf(a)-["PREC","Tmin","Tmax","ELEVACION","PENDIENTE"].indexOf(b));
let variable=variables[0]||"PREC", anio=null, mes=null, seleccion=null, capa=new Map(), limiteMin=null, limiteMax=null;
function opciones(select, entries, elegida) {
 select.replaceChildren();
 for(const [value,label] of entries){const o=document.createElement("option");o.value=String(value);o.textContent=label;select.append(o);}
 if(elegida!==null&&elegida!==undefined&&entries.some(e=>String(e[0])===String(elegida)))select.value=String(elegida);
}
opciones($("variable"),variables.map(v=>[v,etiquetas[v]||v]),variable);
function ajustarFechas(conservar=true) {
 const disponibles=[...(fechas.get(variable)?.values()||[])];
 const anios=[...new Set(disponibles.filter(d=>d.a!==null).map(d=>d.a))].sort((a,b)=>a-b);
 const previo=conservar?anio:null;
 if(!anios.length){anio=null;mes=null;opciones($("anio"),[["","Estático"]]);opciones($("mes"),[["","Sin fecha"]]);$("anio").disabled=true;$("mes").disabled=true;return;}
 anio=anios.includes(previo)?previo:anios[anios.length-1];
 opciones($("anio"),anios.map(a=>[a,String(a)]),anio);$("anio").disabled=false;
 ajustarMeses();
}
function ajustarMeses(){
 const ms=[...new Set([...(fechas.get(variable)?.values()||[])].filter(d=>d.a===anio&&d.m!==null).map(d=>d.m))].sort((a,b)=>a-b);
 if(!ms.length){mes=null;opciones($("mes"),[["","Sin mes"]]);$("mes").disabled=true;return;}
 mes=ms.includes(mes)?mes:ms[0];opciones($("mes"),ms.map(m=>[m,meses[m-1]||String(m)]),mes);$("mes").disabled=false;
}
const proyectar=([lon,lat])=>[R*Number(lon)*Math.PI/180,-R*Math.log(Math.tan(Math.PI/4+Math.max(-85,Math.min(85,Number(lat)))*Math.PI/360))];
const invertir=([x,y])=>[x/R*180/Math.PI,(2*Math.atan(Math.exp(-y/R))-Math.PI/2)*180/Math.PI];
const parcelas=[];let total=[Infinity,Infinity,-Infinity,-Infinity];
for(const feature of GEO.features||[]){
 const geometry=feature.geometry||{};
 const polygons=geometry.type==="Polygon"?[geometry.coordinates]:geometry.type==="MultiPolygon"?geometry.coordinates:[];
 const proyectados=polygons.map(p=>p.map(r=>r.map(proyectar))).filter(p=>p.length&&p[0].length);
 let bbox=[Infinity,Infinity,-Infinity,-Infinity], path="";
 for(const polygon of proyectados)for(const ring of polygon){
  if(!ring.every(c=>c.every(Number.isFinite)))continue;
  path+=ring.map((c,i)=>(i?"L":"M")+c[0].toFixed(3)+","+c[1].toFixed(3)).join(" ")+" Z ";
  for(const [x,y] of ring){bbox[0]=Math.min(bbox[0],x);bbox[1]=Math.min(bbox[1],y);bbox[2]=Math.max(bbox[2],x);bbox[3]=Math.max(bbox[3],y);}
 }
 if(!path||!bbox.every(Number.isFinite))continue;
 const props=feature.properties||{}, id=String(props.ID_POLIGONO??"");
 if(!id)continue;
 // Centroide del anillo exterior de mayor área; punto de referencia del mapa.
 let mayor=0, centro=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2];
 for(const polygon of proyectados){const ring=polygon[0];let area=0,cx=0,cy=0;for(let i=0;i<ring.length;i++){const p=ring[i],q=ring[(i+1)%ring.length],cross=p[0]*q[1]-q[0]*p[1];area+=cross;cx+=(p[0]+q[0])*cross;cy+=(p[1]+q[1])*cross;}if(Math.abs(area)>mayor&&Math.abs(area)>1e-9){mayor=Math.abs(area);centro=[cx/(3*area),cy/(3*area)];}}
 if(numero(props.longitud)!==null&&numero(props.latitud)!==null)centro=proyectar([props.longitud,props.latitud]);
 const shape=document.createElementNS(NS,"path");shape.setAttribute("d",path);shape.setAttribute("class","parcel");shape.dataset.id=id;$("poligonos").append(shape);
 const point=document.createElementNS(NS,"circle");point.setAttribute("cx",centro[0]);point.setAttribute("cy",centro[1]);point.setAttribute("class","marker");point.dataset.id=id;$("puntos").append(point);
 parcelas.push({id,props,bbox,centro,shape,point});
 total=[Math.min(total[0],bbox[0]),Math.min(total[1],bbox[1]),Math.max(total[2],bbox[2]),Math.max(total[3],bbox[3])];
}
const porId=new Map(parcelas.map(p=>[p.id,p]));
for(const p of [...parcelas].sort((a,b)=>a.id.localeCompare(b.id))){const o=document.createElement("option");o.value=p.id;$("ids").append(o);}
$("numero-parcelas").textContent=`${parcelas.length} parcelas`;
const svg=$("mapa"), canvas=$("canvas");let vista={x:0,y:0,w:100,h:100}, escalaPixel=1;
let fondoMapa=null, fondoListo=false, fondoCargando=false;
let fondoActivo=true;
function sincronizarFondo(){
 if(!fondoMapa||!fondoActivo)return;
 const rect=svg.getBoundingClientRect();if(!rect.width||!rect.height)return;
 const center=invertir([vista.x+vista.w/2,vista.y+vista.h/2]);
 // MapLibre usa un mundo de 512 px en zoom 0; SVG usa metros Web Mercator.
 const nivel=Math.max(0,Math.min(22,Math.log2(2*Math.PI*R/(512*escalaPixel))));
 fondoMapa.resize();fondoMapa.jumpTo({center,zoom:nivel,bearing:0,pitch:0});
}
function estadoFondo(texto){if(fondoActivo)$("fondo-estado").textContent=texto;}
function crearFondo(){
 if(fondoMapa||!fondoActivo)return;
 try{
  fondoMapa=new window.maplibregl.Map({container:"fondo-mapa",style:"https://tiles.openfreemap.org/styles/liberty",interactive:false,attributionControl:false,renderWorldCopies:false,maxZoom:22,center:invertir([vista.x+vista.w/2,vista.y+vista.h/2]),zoom:9,bearing:0,pitch:0});
  fondoMapa.on("load",()=>{fondoListo=true;estadoFondo("Calles y localidades · OpenStreetMap");sincronizarFondo();});
  fondoMapa.on("error",()=>{estadoFondo(fondoListo?"Fondo incompleto · Revisa la conexión":"Fondo no disponible · Revisa la conexión");});
  sincronizarFondo();
 }catch(error){estadoFondo("Fondo no disponible · Las parcelas siguen visibles");}
}
function cargarFondo(){
 if(!fondoActivo||fondoMapa||fondoCargando)return;
 if(window.maplibregl){crearFondo();return;}
 fondoCargando=true;
 const css=document.createElement("link");css.rel="stylesheet";css.href="https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.css";document.head.append(css);
 const script=document.createElement("script");script.src="https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.js";script.async=true;
 script.onload=()=>{fondoCargando=false;crearFondo();};
 script.onerror=()=>{fondoCargando=false;estadoFondo("Fondo no disponible · Se necesita internet");};
 document.head.append(script);
}
$("fondo").onchange=()=>{
 fondoActivo=$("fondo").value==="calles";
 $("fondo-mapa").hidden=!fondoActivo;$("fondo-creditos").hidden=!fondoActivo;
 $("cuadricula").style.opacity=fondoActivo?".25":"1";
 $("fondo-estado").textContent=fondoActivo?(fondoListo?"Calles y localidades · OpenStreetMap":"Cargando fondo cartográfico…"):"Parcelas · Sin fondo cartográfico";
 if(fondoActivo){cargarFondo();sincronizarFondo();}
};
function ajustarVista(bbox,factor=1.15){
 const rect=svg.getBoundingClientRect(), ratio=rect.width/Math.max(1,rect.height);
 let w=Math.max(bbox[2]-bbox[0],30)*factor,h=Math.max(bbox[3]-bbox[1],30)*factor;
 if(w/h<ratio)w=h*ratio;else h=w/ratio;
 const centro=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2];vista={x:centro[0]-w/2,y:centro[1]-h/2,w,h};dibujarVista();
}
function pasoAgradable(valor){const base=10**Math.floor(Math.log10(Math.max(valor,1e-8))),n=valor/base;return(n<=1?1:n<=2?2:n<=5?5:10)*base;}
function crearSVG(tag,attrs,text){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
function dibujarVista(){
 svg.setAttribute("viewBox",`${vista.x} ${vista.y} ${vista.w} ${vista.h}`);
 const rect=svg.getBoundingClientRect();escalaPixel=vista.w/Math.max(rect.width,1);
 for(const p of parcelas){p.point.setAttribute("r",escalaPixel*3);p.point.style.display=Math.max(p.bbox[2]-p.bbox[0],p.bbox[3]-p.bbox[1])/escalaPixel<12?"":"none";}
 const grid=$("cuadricula");grid.replaceChildren();
 const sw=invertir([vista.x,vista.y+vista.h]),ne=invertir([vista.x+vista.w,vista.y]);
 const paso=pasoAgradable(Math.max(ne[0]-sw[0],ne[1]-sw[1])/5), fs=escalaPixel*10;
 const lineAttrs={stroke:"#b7c8b1","stroke-width":.7,"vector-effect":"non-scaling-stroke",opacity:.5};
 const textAttrs={fill:"#7e9178","font-size":fs,"font-family":"system-ui"};
 let n=0;
 for(let lon=Math.ceil(sw[0]/paso)*paso;lon<=ne[0]&&n++<20;lon+=paso){const x=proyectar([lon,0])[0];grid.append(crearSVG("line",{x1:x,y1:vista.y,x2:x,y2:vista.y+vista.h,...lineAttrs}),crearSVG("text",{x:x+fs*.4,y:vista.y+vista.h-fs*.7,...textAttrs},`${fmt(lon,Math.max(0,-Math.floor(Math.log10(paso))))}°`));}
 n=0;for(let lat=Math.ceil(sw[1]/paso)*paso;lat<=ne[1]&&n++<20;lat+=paso){const y=proyectar([0,lat])[1];grid.append(crearSVG("line",{x1:vista.x,y1:y,x2:vista.x+vista.w,y2:y,...lineAttrs}),crearSVG("text",{x:vista.x+vista.w-fs*.6,y:y-fs*.4,"text-anchor":"end",...textAttrs},`${fmt(lat,Math.max(0,-Math.floor(Math.log10(paso))))}°`));}
 const esc=$("escala");esc.replaceChildren();
 const lat=invertir([vista.x+vista.w/2,vista.y+vista.h/2])[1], correccion=Math.cos(lat*Math.PI/180);
 const metros=pasoAgradable(escalaPixel*100*correccion), ancho=metros/correccion;
 const x=vista.x+vista.w-escalaPixel*22-ancho,y=vista.y+vista.h-escalaPixel*42;
 esc.append(crearSVG("path",{d:`M ${x} ${y-escalaPixel*4} V ${y} H ${x+ancho} V ${y-escalaPixel*4}`,stroke:"#51634b","stroke-width":1.2,"vector-effect":"non-scaling-stroke",fill:"none"}),crearSVG("text",{x:x+ancho/2,y:y-escalaPixel*8,"text-anchor":"middle",fill:"#51634b","font-size":escalaPixel*10,"font-family":"system-ui"},`${fmt(metros>=1000?metros/1000:metros,2)} ${metros>=1000?"km":"m"}`));
 sincronizarFondo();
}
function zoom(factor,anchor){
 const rect=svg.getBoundingClientRect(),px=anchor?anchor[0]/rect.width:.5,py=anchor?anchor[1]/rect.height:.5;
 const minimo=2*Math.PI*R/(512*2**22)*Math.max(1,rect.width);
 const nw=Math.max(minimo,Math.min(4e7,vista.w*factor)),nh=nw/Math.max(.1,rect.width/rect.height);
 vista={x:vista.x+px*(vista.w-nw),y:vista.y+py*(vista.h-nh),w:nw,h:nh};dibujarVista();
}
$("zoom-mas").onclick=()=>zoom(.6);$("zoom-menos").onclick=()=>zoom(1/.6);
$("encuadrar").onclick=()=>{if(total.every(Number.isFinite))ajustarVista(total);};
svg.addEventListener("wheel",e=>{e.preventDefault();const r=svg.getBoundingClientRect();zoom(Math.exp(Math.max(-.45,Math.min(.45,e.deltaY*.0015))),[e.clientX-r.left,e.clientY-r.top]);},{passive:false});
let arrastre=null;
svg.addEventListener("pointerdown",e=>{if(e.button!==0)return;arrastre={px:e.clientX,py:e.clientY,x:vista.x,y:vista.y,w:vista.w,h:vista.h,id:e.target.dataset.id,movido:false};svg.setPointerCapture(e.pointerId);});
svg.addEventListener("pointermove",e=>{
 if(arrastre){const r=svg.getBoundingClientRect(),dx=e.clientX-arrastre.px,dy=e.clientY-arrastre.py;arrastre.movido=arrastre.movido||Math.hypot(dx,dy)>4;if(arrastre.movido){svg.classList.add("dragging");vista.x=arrastre.x-dx/r.width*arrastre.w;vista.y=arrastre.y-dy/r.height*arrastre.h;$("tooltip").style.display="none";dibujarVista();}return;}
 const id=e.target.dataset.id;if(!id){$("tooltip").style.display="none";return;}const r=canvas.getBoundingClientRect(),registro=capa.get(id);$("tooltip").textContent=`${id}\n${fmt(registro?.valor)}${numero(registro?.valor)!==null?" "+unidadActual():""}`;$("tooltip").style.display="block";$("tooltip").style.left=Math.min(e.clientX-r.left+12,r.width-190)+"px";$("tooltip").style.top=Math.max(5,e.clientY-r.top-52)+"px";
});
svg.addEventListener("pointerup",e=>{if(arrastre&&!arrastre.movido&&arrastre.id)seleccionar(arrastre.id,false);arrastre=null;svg.classList.remove("dragging");if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);});
svg.addEventListener("pointercancel",()=>{arrastre=null;svg.classList.remove("dragging");});svg.addEventListener("pointerleave",()=>{$("tooltip").style.display="none";});
function colorValor(valor){
 if(valor===null)return"#b9c3b6";const stops=paletas[variable]||paletas.PREC;
 const ratio=limiteMax===limiteMin ? 0.5 : Math.max(0,Math.min(1,(valor-limiteMin)/(limiteMax-limiteMin)));
 const position=ratio*(stops.length-1),i=Math.min(stops.length-2,Math.floor(position)),f=position-i;
 const rgb=h=>h.slice(1).match(/../g).map(v=>parseInt(v,16)),a=rgb(stops[i]),b=rgb(stops[i+1]);return"rgb("+a.map((v,j)=>Math.round(v+(b[j]-v)*f)).join(",")+")";
}
function unidadActual(){return [...capa.values()].find(r=>r.unidad)?.unidad||unidades[variable]||"";}
function fechaActual(){return anio===null?"Capa estática":`${mes!==null?(meses[mes-1]||mes)+" de ":""}${anio}`;}
function actualizarCapa(){
 capa=capas.get(capaKey(variable,anio,mes))||new Map();const valores=parcelas.map(p=>numero(capa.get(p.id)?.valor)).filter(v=>v!==null);
 limiteMin=valores.length?Math.min(...valores):null;limiteMax=valores.length?Math.max(...valores):null;
 for(const p of parcelas){const v=numero(capa.get(p.id)?.valor),color=colorValor(v);p.shape.setAttribute("fill",color);p.shape.classList.toggle("parcel-missing",v===null);p.point.setAttribute("fill",color);}
 $("titulo-capa").textContent=`${etiquetas[variable]||variable} · ${fechaActual()}`;$("minimo").textContent=fmt(limiteMin);$("maximo").textContent=fmt(limiteMax);$("unidad-leyenda").textContent=unidadActual();$("cuenta-datos").textContent=`${valores.length} con dato · ${parcelas.length-valores.length} sin dato`;
 $("gradiente").style.background=`linear-gradient(90deg,${(paletas[variable]||paletas.PREC).join(",")})`;
 $("resumen").textContent=`${etiquetas[variable]||variable} (${unidadActual()}) · ${fechaActual()}`;
 const clima=["PREC","Tmin","Tmax"].includes(variable);
 $("nota-resolucion").textContent=clima?"Clima regional: resolución aproximada de 5 km. Varias parcelas pueden compartir las mismas celdas y valores. Tmin y Tmax son promedios mensuales de las mínimas y máximas diarias.":"Relieve: resolución aproximada de 120 m. En parcelas pequeñas, el detalle disponible depende de las celdas que las intersectan.";
 actualizarFicha();
}
function registroOtraVariable(v,id){
 const lista=[...(fechas.get(v)?.values()||[])];const estatica=lista.find(f=>f.a===null&&f.m===null);
 return capas.get(capaKey(v,estatica?null:anio,estatica?null:mes))?.get(id);
}
function actualizarFicha(){
 if(!seleccion)return;const p=porId.get(seleccion),props=p.props,r=capa.get(p.id);$("sin-seleccion").hidden=true;$("ficha").hidden=false;
 $("id-parcela").textContent=p.id;$("ubicacion").textContent=[props.Municipio,props.Estado].filter(Boolean).join(" · ")||"Parcela del dataset";
 $("etiqueta-valor").textContent=`${etiquetas[variable]||variable} · ${fechaActual()}`;
 $("valor-parcela").textContent=fmt(r?.valor)+(numero(r?.valor)!==null?" "+unidadActual():"");
 $("estado-valor").textContent=r?`Extracción: ${String(r.estado_extraccion||"sin especificar").replaceAll("_"," ")}`:"No existe un registro para esta capa.";
 const cobertura=numero(r?.cobertura_valida_pct);
 $("metodo-valor").textContent=[r?.metodo?`Método: ${String(r.metodo).replaceAll("_"," ")}`:"",cobertura!==null?`Cobertura válida: ${fmt(cobertura)} %`:""].filter(Boolean).join(" · ");
 const parcial=String(r?.estado_extraccion||"").toUpperCase()==="COBERTURA_PARCIAL";
 $("alerta-cobertura").hidden=!parcial;
 $("alerta-cobertura").textContent=parcial?`Este valor describe solo la parte de la parcela con datos válidos${cobertura!==null?" ("+fmt(cobertura)+" %)":""}. La zona sin cobertura no interviene en el promedio.`:"";
 $("area").textContent=fmt(props.AREA_HA)+(numero(props.AREA_HA)!==null?" ha":"");$("conjunto").textContent=props.CONJUNTO||"Sin especificar";$("rendimiento").textContent=fmt(props.RENDIMIENTO_T_HA)+(numero(props.RENDIMIENTO_T_HA)!==null?" t/ha":"");
 $("coordenadas").textContent=invertir(p.centro).map(v=>fmt(v,5)).join(", ");
 $("titulo-tabla").textContent=anio===null?"Capas estáticas":"Otras variables de esta fecha";$("tabla").replaceChildren();
 for(const v of variables){if(v===variable)continue;const r2=registroOtraVariable(v,p.id);if(anio===null&&[...(fechas.get(v)?.values()||[])].some(f=>f.a!==null))continue;const tr=document.createElement("tr"),td=document.createElement("td"),val=document.createElement("td");td.textContent=etiquetas[v]||v;val.textContent=fmt(r2?.valor)+(numero(r2?.valor)!==null?" "+(r2.unidad||unidades[v]||""):"");tr.append(td,val);$("tabla").append(tr);}
}
function seleccionar(id,acercar){
 const p=porId.get(id);if(!p)return;seleccion=id;$("buscar").value=id;$("resultado-busqueda").textContent="";
 for(const parcela of parcelas){parcela.shape.classList.toggle("selected",parcela.id===id);parcela.point.classList.toggle("selected",parcela.id===id);}
 // Dibuja la parcela seleccionada sobre sus vecinas.
 $("poligonos").append(p.shape);$("puntos").append(p.point);actualizarFicha();if(acercar)ajustarVista(p.bbox,2.8);
}
function buscar(){const q=$("buscar").value.trim().toLowerCase();if(!q)return;const exacta=parcelas.find(p=>p.id.toLowerCase()===q),coincidencias=parcelas.filter(p=>p.id.toLowerCase().includes(q));if(exacta)seleccionar(exacta.id,true);else if(coincidencias.length===1)seleccionar(coincidencias[0].id,true);else $("resultado-busqueda").textContent=coincidencias.length?`${coincidencias.length} coincidencias. Escribe el identificador completo.`:"No se encontró esa parcela.";}
$("buscar-boton").onclick=buscar;$("buscar").addEventListener("keydown",e=>{if(e.key==="Enter")buscar();});$("centrar-parcela").onclick=()=>{if(seleccion)ajustarVista(porId.get(seleccion).bbox,2.8);};
$("variable").onchange=()=>{variable=$("variable").value;ajustarFechas();actualizarCapa();};$("anio").onchange=()=>{anio=numero($("anio").value);ajustarMeses();actualizarCapa();};$("mes").onchange=()=>{mes=numero($("mes").value);actualizarCapa();};
function csvCelda(valor){if(valor===null||valor===undefined)return"";let s=String(valor);if(typeof valor==="string"&&/^[=+\-@\t\r]/.test(s))s="'"+s;return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;}
$("csv").onclick=()=>{
 const campos=["ID_POLIGONO","AREA_HA","CONJUNTO","Municipio","Estado","RENDIMIENTO_T_HA","longitud_ref","latitud_ref","variable","anio","mes","valor","unidad","estado_extraccion","metodo","cobertura_valida_pct"],filas=[campos.join(",")];
 for(const p of parcelas){const r=capa.get(p.id)||{},coord=invertir(p.centro),row={...p.props,ID_POLIGONO:p.id,longitud_ref:coord[0],latitud_ref:coord[1],variable,anio,mes,valor:numero(r.valor),unidad:r.unidad||unidadActual(),estado_extraccion:r.estado_extraccion||"sin_registro",metodo:r.metodo||"",cobertura_valida_pct:numero(r.cobertura_valida_pct)};filas.push(campos.map(c=>csvCelda(row[c])).join(","));}
 const blob=new Blob(["\uFEFF"+filas.join("\r\n")+"\r\n"],{type:"text/csv;charset=utf-8"}),url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=`agrocebada_${variable}${anio!==null?"_"+anio:""}${mes!==null?"_"+String(mes).padStart(2,"0"):""}.csv`;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
ajustarFechas(false);actualizarCapa();
if(parcelas.length)ajustarVista(total);else $("sin-geometrias").style.display="block";
new ResizeObserver(()=>{const r=svg.getBoundingClientRect();if(!r.width||!r.height)return;const center=[vista.x+vista.w/2,vista.y+vista.h/2],h=vista.w*r.height/r.width;vista.y=center[1]-h/2;vista.h=h;dibujarVista();}).observe(svg);
$("cuadricula").style.opacity=".25";cargarFondo();
})();
</script>
</body>
</html>'''



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=HERE / "DATASET_RETO_AGRO_2026", help="Carpeta de datos descomprimidos.")
    parser.add_argument("--parcelas", type=Path, help="Shapefile alternativo; debe tener .prj y campo ID.")
    parser.add_argument("--etiquetas", type=Path, help="CSV de ID, área, conjunto y rendimiento.")
    parser.add_argument("--salida", type=Path, default=HERE / "salidas_extractor", help="Carpeta de resultados.")
    parser.add_argument("--imagen", action="append", type=Path, help="Extraer solo este TIFF/carpeta. Se puede repetir.")
    parser.add_argument("--anio", type=int, help="Limitar clima a un año; conserva topografía.")
    parser.add_argument("--metodo", choices=("zonal", "punto"), default="zonal", help="zonal: media por área; punto: píxel de un punto interior.")
    parser.add_argument("--sin-mapa", action="store_true", help="Generar CSV y GeoJSON sin HTML.")
    args = parser.parse_args(argv)
    dataset, salida = args.dataset.expanduser().resolve(), args.salida.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"No existe la carpeta de datos: {dataset}")
    # Evita una ruta de salida que pueda sobrescribir el conjunto de entrada.
    if salida == dataset or dataset in salida.parents or salida in dataset.parents:
        parser.error("La carpeta de salida debe estar separada del dataset original.")
    shp = (args.parcelas or dataset / "Parcelas_Reto_AGC_CONJUNTO_70_30" / "Parcelas_Reto_AGC_CONJUNTO.shp").resolve()
    etiquetas = (args.etiquetas or dataset / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv").resolve()
    if not shp.exists() or not etiquetas.exists():
        parser.error("No se encuentran las parcelas o el CSV de etiquetas; usa --parcelas y --etiquetas.")
    rutas = descubrir_rasters(dataset, args.imagen, args.anio)
    parcelas, crs_origen = cargar_parcelas(shp, etiquetas)
    salida.mkdir(parents=True, exist_ok=True)
    print(f"Extrayendo {len(rutas)} TIFF para {len(parcelas)} parcelas; método {args.metodo}.", flush=True)
    registros, inventario, coordenadas, crs_area = extraer(parcelas, rutas, dataset, salida, args.metodo)
    escribir_csv(salida / "datos_por_imagen.csv", registros, LONG_FIELDS)
    mensual, fc, topo, ft, ancho, fa = tablas_resumen(parcelas, registros)
    escribir_csv(salida / "clima_mensual.csv", mensual, fc)
    escribir_csv(salida / "topografia_parcelas.csv", topo, ft)
    escribir_csv(salida / "parcelas_datos.csv", ancho, fa)
    escribir_csv(salida / "coordenadas_parcelas.csv", coordenadas,
                 ["ID_POLIGONO", "crs", "x_centroide", "y_centroide", "x_muestreo", "y_muestreo", "unidad_coordenadas"])
    inventario_csv = [dict(r, transformacion=json.dumps(r["transformacion"])) for r in inventario]
    escribir_csv(salida / "inventario_rasters.csv", inventario_csv, list(inventario_csv[0]))
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": p.propiedades["ID_POLIGONO"],
         "properties": p.propiedades, "geometry": mapping(p.geometria)} for p in parcelas]}
    escribir_json(salida / "parcelas.geojson", geojson)
    if not args.sin_mapa:
        # Reduce el HTML a los campos de consulta; CSV conserva toda la trazabilidad.
        campos_mapa = ["ID_POLIGONO", "variable", "anio", "mes", "valor", "unidad",
                      "estado_extraccion", "cobertura_valida_pct", "metodo"]
        escribir_mapa(salida / "mapa_parcelas.html", geojson,
                      [{campo: r[campo] for campo in campos_mapa} for r in registros])
    resumen = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset),
        "shapefile": str(shp), "etiquetas": str(etiquetas), "metodo": args.metodo,
        "tiff_procesados": len(rutas), "capas_bandas": len(inventario), "parcelas": len(parcelas),
        "registros_por_imagen": len(registros), "registros_clima_mensual": len(mensual),
        "valores_ausentes": sum(r["valor"] is None for r in registros),
        "estados_extraccion": dict(Counter(r["estado_extraccion"] for r in registros)),
        "crs_shapefile": crs_origen.to_string(), "crs_geojson": "EPSG:4326",
        "crs_area_ponderacion": crs_area.to_wkt(),
        "crs_rasters": sorted({r["crs"] for r in inventario}),
        "notas": ["NoData, máscaras de raster y números no finitos se excluyen; ceros válidos se conservan.",
                  "Tmin/Tmax son promedios mensuales, no extremos absolutos.",
                  "La media ponderada usa superficie intersectada en proyección equivalente LAEA.",
                  "La cobertura es superficie con valor válido / superficie geométrica de la parcela.",
                  "La temperatura media aproximada se calcula como (Tmin + Tmax) / 2.",
                  "Los IDs se vinculan por nombre, nunca por posición de las filas.",
                  "El mapa contiene datos extraídos; el rendimiento conocido se muestra por separado."]}
    escribir_json(salida / "resumen_ejecucion.json", resumen)
    print(f"Listo: {len(registros):,} registros; {resumen['valores_ausentes']} sin valor.")
    print(f"Resultados: {salida}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, rasterio.errors.RasterioError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
