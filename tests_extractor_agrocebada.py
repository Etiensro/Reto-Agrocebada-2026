"""Validación geométrica y numérica del extractor.

Ejecutar: .venv/bin/python -m unittest tests_extractor_agrocebada.py
Los oráculos reales se calcularon independientemente: áreas geodésicas
WGS84 para clima y áreas cartesianas EPSG:6372 para topografía.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely import union_all
from shapely.geometry import Polygon, box

import extraccion as extractor


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "DATASET_RETO_AGRO_2026"
METRICO = CRS.from_epsg(6372)
IDENTIDAD = Transformer.from_crs(METRICO, METRICO, always_xy=True)


@contextlib.contextmanager
def raster_sintetico(valores, transformacion=None):
    """Grilla métrica con áreas conocidas por geometría elemental."""
    datos = np.asarray(valores, dtype="float32")
    with MemoryFile() as memoria:
        with memoria.open(driver="GTiff", width=datos.shape[1],
                          height=datos.shape[0], count=1, dtype="float32",
                          crs=METRICO, nodata=-9999,
                          transform=transformacion or from_origin(0, 2, 1, 1)) as src:
            src.write(datos, 1)
            leidos = src.read(1, masked=True).astype("float64").filled(np.nan)
            yield src, leidos


class ExtraccionGeometrica(unittest.TestCase):
    def extraer(self, geometria, src, datos):
        plan = extractor.preparar_plan(geometria, src, IDENTIDAD)
        return extractor.estadisticas_zonales(datos, plan)

    def test_parcela_menor_que_pixel_sin_centro_de_pixel(self):
        # No contiene el centro de la celda (0.5,1.5), pero tiene datos válidos.
        with raster_sintetico([[7, 99], [50, 60]]) as (src, datos):
            r = self.extraer(box(0.1, 1.8, 0.2, 1.9), src, datos)
        self.assertAlmostEqual(r["valor"], 7, places=12)
        self.assertEqual(r["pixeles_intersectados"], 1)
        self.assertEqual(r["cobertura_valida_pct"], 100)
        self.assertEqual(r["estado_extraccion"], "OK")

    def test_interseccion_ponderada_y_desviacion(self):
        # 75% del área cae en valor 0 y 25% en valor 8: media 2, varianza 12.
        with raster_sintetico([[0, 8], [100, 200]]) as (src, datos):
            r = self.extraer(box(0.25, 1.25, 1.25, 1.75), src, datos)
        self.assertAlmostEqual(r["valor"], 2, places=12)
        self.assertAlmostEqual(r["std"], np.sqrt(12), places=12)
        self.assertEqual((r["min"], r["max"]), (0, 8))
        self.assertEqual(r["pixeles_validos"], 2)

    def test_nodata_excluido_y_cero_conservado(self):
        with raster_sintetico([[0, -9999], [100, 200]]) as (src, datos):
            r = self.extraer(box(0.25, 1.25, 1.25, 1.75), src, datos)
        self.assertEqual(r["valor"], 0)
        self.assertEqual(r["pixeles_intersectados"], 2)
        self.assertEqual(r["pixeles_validos"], 1)
        self.assertAlmostEqual(r["cobertura_valida_pct"], 75, places=12)
        self.assertEqual(r["estado_extraccion"], "COBERTURA_PARCIAL")

    def test_numeros_no_finitos_excluidos(self):
        with raster_sintetico([[3, np.inf], [np.nan, 200]]) as (src, datos):
            r = self.extraer(box(0, 0, 2, 2), src, datos)
        self.assertEqual(r["valor"], 101.5)
        self.assertEqual(r["pixeles_validos"], 2)
        self.assertEqual(r["cobertura_valida_pct"], 50)

    def test_cobertura_parcial_por_extension(self):
        with raster_sintetico([[0, 8], [100, 200]]) as (src, datos):
            r = self.extraer(box(0, 1.25, 3, 1.75), src, datos)
        self.assertEqual(r["valor"], 4)
        self.assertAlmostEqual(r["cobertura_valida_pct"], 200 / 3, places=12)
        self.assertEqual(r["estado_extraccion"], "COBERTURA_PARCIAL")

    def test_fuera_del_raster_y_nodata_tienen_estados_distintos(self):
        with raster_sintetico([[-9999, 8], [100, 200]]) as (src, datos):
            fuera = self.extraer(box(10, 10, 11, 11), src, datos)
            nodata = self.extraer(box(0.1, 1.8, 0.2, 1.9), src, datos)
        self.assertIsNone(fuera["valor"])
        self.assertIsNone(nodata["valor"])
        self.assertEqual(fuera["estado_extraccion"], "SIN_COBERTURA")
        self.assertEqual(nodata["estado_extraccion"], "SIN_DATOS_VALIDOS")

    def test_grilla_rotada(self):
        affine = Affine.translation(10, 20) * Affine.rotation(30) * Affine.scale(1, -1)
        poligono = Polygon([affine * p for p in
                            ((0.25, 0.25), (1.25, 0.25), (1.25, 0.75), (0.25, 0.75))])
        with raster_sintetico([[0, 8], [100, 200]], affine) as (src, datos):
            r = self.extraer(poligono, src, datos)
        self.assertAlmostEqual(r["valor"], 2, places=10)
        self.assertAlmostEqual(r["cobertura_valida_pct"], 100, places=10)

    def test_lectura_aplica_escala_offset_y_mascara(self):
        # El 0 almacenado representa 10 tras scale=2,offset=10; NoData sigue ausente.
        with tempfile.TemporaryDirectory(prefix="agrocebada_prueba_") as carpeta:
            carpeta = Path(carpeta)
            archivo = carpeta / "imagen_sintetica.tif"
            with rasterio.open(archivo, "w", driver="GTiff", width=2, height=2,
                               count=1, dtype="int16", crs="EPSG:4326", nodata=-9999,
                               transform=from_origin(-99, 20, 0.01, 0.01)) as src:
                src.write(np.array([[0, -9999], [10, 20]], dtype="int16"), 1)
                src.scales = (2,)
                src.offsets = (10,)
            g = box(-98.999, 19.991, -98.991, 19.999)
            p = extractor.Parcela({"ID_POLIGONO": "SINTETICA"}, g, g.representative_point())
            with contextlib.redirect_stdout(io.StringIO()):
                registros, inventario, _, _ = extractor.extraer(
                    [p], [archivo], carpeta, carpeta / "salida", "zonal")
            self.assertEqual(registros[0]["valor"], 10)
            self.assertEqual(inventario[0]["escala"], 2)
            self.assertEqual(inventario[0]["offset"], 10)
            self.assertTrue((carpeta / "salida/por_imagen/imagen_sintetica.csv").is_file())


@unittest.skipUnless(DATASET.is_dir(), "Requiere el dataset real AgroCebada")
class DatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parcelas, cls.crs = extractor.cargar_parcelas(
            DATASET / "Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO.shp",
            DATASET / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv")
        cls.por_id = {p.propiedades["ID_POLIGONO"]: p for p in cls.parcelas}
        centro = union_all([p.geometria for p in cls.parcelas]).centroid
        cls.area = CRS.from_proj4(f"+proj=laea +lat_0={centro.y} +lon_0={centro.x} "
                                 "+datum=WGS84 +units=m +no_defs")

    def estadistica(self, nombre, identificador):
        parcela = self.por_id[identificador]
        with rasterio.open(next(DATASET.rglob(nombre))) as src:
            datos = src.read(1, masked=True).astype("float64").filled(np.nan)
            crs = CRS.from_user_input(src.crs)
            g = extractor.convertir(parcela.geometria, extractor.WGS84, crs)
            proyeccion = Transformer.from_crs(crs, self.area, always_xy=True)
            return extractor.estadisticas_zonales(
                datos, extractor.preparar_plan(g, src, proyeccion))

    def test_integridad_ids_etiquetas_y_coordenadas(self):
        # ESRI WKT declara ejes longitud/latitud; EPSG:4326 los nombra al revés.
        self.assertTrue(self.crs.equals(extractor.WGS84, ignore_axis_order=True))
        self.assertEqual(len(self.parcelas), 197)
        self.assertEqual(len(self.por_id), 197)
        self.assertEqual(Counter(p.propiedades["CONJUNTO"] for p in self.parcelas),
                         {"ENTRENAMIENTO": 138, "PREDICCION": 59})
        parcela = self.por_id["AGC_044"]
        self.assertEqual(parcela.propiedades["AREA_HA"], 0.68)
        self.assertEqual(parcela.propiedades["Municipio"], "Emiliano Zapata")
        self.assertTrue(parcela.geometria.contains(parcela.punto))
        self.assertAlmostEqual(parcela.punto.x, -98.58665513880126, places=10)
        self.assertAlmostEqual(parcela.punto.y, 19.629015, places=10)
        self.assertTrue(all(p.geometria.is_valid and p.geometria.has_z is False
                            for p in self.parcelas))

    def test_reproyeccion_sin_intercambiar_latitud_longitud(self):
        p = self.por_id["AGC_044"].punto
        convertido = extractor.convertir(p, extractor.WGS84, METRICO)
        self.assertAlmostEqual(convertido.x, 2856862.5661666915, places=6)
        self.assertAlmostEqual(convertido.y, 851981.0381097617, places=6)
        vuelta = extractor.convertir(convertido, METRICO, extractor.WGS84)
        self.assertLess(vuelta.distance(p), 1e-8)

    def test_inventario_real_y_filtro_anio(self):
        completo = extractor.descubrir_rasters(DATASET, None, None)
        seleccionado = extractor.descubrir_rasters(DATASET, None, 2025)
        self.assertEqual(len(completo), 146)
        self.assertEqual(len(seleccionado), 38)
        self.assertEqual(Counter(extractor.identificar_raster(p)[0] for p in completo),
                         {"PREC": 48, "Tmin": 48, "Tmax": 48,
                          "ELEVACION": 1, "PENDIENTE": 1})

    def test_clima_pixel_unico_y_oraculo_geodesico_multicelda(self):
        esperados = {
            "AGC_044": {"PREC": 112.65020751953125, "Tmin": 12.634757041931152,
                        "Tmax": 23.278505325317383},
            "AGC_180": {"PREC": 107.2708271856337, "Tmin": 12.534463363514204,
                        "Tmax": 22.703683216908967},
            "AGC_123": {"PREC": 162.360028570844, "Tmin": 12.243945775604509,
                        "Tmax": 22.97273716323366},
        }
        # LAEA y área elipsoidal tienen diferencias de redondeo mucho menores a 1e-6.
        for identificador, variables in esperados.items():
            for variable, esperado in variables.items():
                with self.subTest(parcela=identificador, variable=variable):
                    r = self.estadistica(f"{variable}_2025_07.tif", identificador)
                    self.assertAlmostEqual(r["valor"], esperado, delta=1e-6)
                    self.assertEqual(r["pixeles_validos"], 1 if identificador == "AGC_044" else 3)
                    self.assertAlmostEqual(r["cobertura_valida_pct"], 100, delta=0.001)

    def test_topografia_y_cobertura_nodata_real(self):
        elev = self.estadistica("Elevacion_INEGI_CEM4_120m.tif", "AGC_044")
        pendiente = self.estadistica("Pendiente_INEGI_CEM4_120m_grados.tif", "AGC_044")
        self.assertAlmostEqual(elev["valor"], 2529.993625199973, delta=0.0001)
        self.assertEqual(elev["pixeles_validos"], 4)
        self.assertAlmostEqual(pendiente["valor"], 0.3488622925465799, delta=0.00001)
        self.assertEqual(pendiente["pixeles_validos"], 3)
        self.assertEqual(pendiente["estado_extraccion"], "COBERTURA_PARCIAL")
        self.assertAlmostEqual(pendiente["cobertura_valida_pct"], 98.02558524821765, delta=0.002)
        # Caso adverso real: una media existe, pero apenas representa 2.1% de la parcela.
        escasa = self.estadistica("Pendiente_INEGI_CEM4_120m_grados.tif", "AGC_170")
        self.assertAlmostEqual(escasa["cobertura_valida_pct"], 2.0968967815821745, delta=0.002)
        self.assertEqual(escasa["estado_extraccion"], "COBERTURA_PARCIAL")


if __name__ == "__main__":
    unittest.main()
