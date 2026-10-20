"""Pruebas del panel: los datos exportados coinciden con las salidas del modelo y
el modelo portado a JavaScript reproduce las predicciones de Python.

Ejecutar: .venv/bin/python -m unittest tests_dashboard.py
Las pruebas que usan Node se omiten si `node` no está instalado.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DASH = HERE / "dashboard"
SALIDA = HERE / "salidas_prediccion"
NODE = shutil.which("node")


def cargar_datos() -> dict:
    texto = (DASH / "datos.js").read_text(encoding="utf-8")
    cuerpo = re.fullmatch(r"window\.DATOS=(.*);\s*", texto, flags=re.S)
    assert cuerpo, "datos.js no tiene el formato esperado"
    return json.loads(cuerpo.group(1))


class DatosExportadosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datos = cargar_datos()
        cls.parcelas = pd.DataFrame(cls.datos["parcelas"]).set_index("id")
        cls.mapa = pd.read_csv(SALIDA / "resultados_para_mapa.csv", encoding="utf-8-sig").set_index("ID_POLIGONO")

    def test_197_parcelas_con_geometria_y_59_por_estimar(self):
        self.assertEqual(len(self.parcelas), 197)
        self.assertEqual(set(self.parcelas.index), {f"AGC_{i:03d}" for i in range(1, 198)})
        self.assertEqual(int((self.parcelas["origen"] == "ocultas").sum()), 59)
        self.assertTrue(self.parcelas["poligono"].map(lambda p: len(p) >= 4).all())
        self.assertTrue(self.parcelas.loc[self.parcelas["origen"] == "ocultas", "real"].isna().all())
        self.assertTrue(self.parcelas.loc[self.parcelas["origen"] == "conocidas", "real"].notna().all())

    def test_estimaciones_coinciden_con_los_csv_del_modelo(self):
        np.testing.assert_allclose(self.parcelas["estimado"], self.mapa.loc[self.parcelas.index, "RENDIMIENTO_ESTIMADO_T_HA"], atol=1e-4)
        entrega = pd.read_csv(SALIDA / "predicciones_59.csv", encoding="utf-8-sig").set_index("ID_POLIGONO")
        ocultas = self.parcelas.loc[entrega.index]
        self.assertTrue((ocultas["origen"] == "ocultas").all())
        np.testing.assert_allclose(ocultas["estimado"], entrega["RENDIMIENTO_T_HA"], atol=1e-4)
        for nivel in ("80", "90"):
            np.testing.assert_allclose(self.parcelas[f"lo{nivel}"], self.mapa.loc[self.parcelas.index, f"LIMITE_INFERIOR_{nivel}_T_HA"], atol=1e-4)
            np.testing.assert_allclose(self.parcelas[f"hi{nivel}"], self.mapa.loc[self.parcelas.index, f"LIMITE_SUPERIOR_{nivel}_T_HA"], atol=1e-4)

    def test_los_rendimientos_conocidos_no_se_alteran(self):
        oficial = pd.read_csv(HERE / "DATASET_RETO_AGRO_2026" / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
                              encoding="utf-8-sig").set_index("ID_POLIGONO")
        conocidas = self.parcelas[self.parcelas["origen"] == "conocidas"]
        np.testing.assert_allclose(conocidas["real"], oficial.loc[conocidas.index, "RENDIMIENTO_T_HA"], atol=1e-4)
        np.testing.assert_allclose(self.parcelas["area"], oficial.loc[self.parcelas.index, "AREA_HA"], atol=1e-2)

    def test_parametros_del_modelo_usan_solo_parcelas_con_etiqueta(self):
        modelo = self.datos["modelo"]
        ids = [t["id"] for t in modelo["entrenamiento"]]
        self.assertEqual(len(ids), 138)
        self.assertEqual(set(ids), set(self.parcelas.index[self.parcelas["origen"] == "conocidas"]))
        self.assertTrue(all(t["real"] is not None for t in modelo["entrenamiento"]))
        n = len(modelo["variables"])
        for clave in ("medianas", "media", "escala", "coeficientes"):
            self.assertEqual(len(modelo[clave]), n)
        self.assertTrue(all(s > 0 for s in modelo["escala"]))

    def test_intervalos_y_residuos_son_coherentes(self):
        for zona, fila in self.datos["intervalos"].items():
            self.assertLess(fila["80"], fila["90"], zona)
            self.assertEqual(len(self.datos["residuos_por_zona"][zona]), fila["n"], zona)
        self.assertEqual(len(self.datos["residuos_por_zona"]["Todas"]), 138)
        zonas = set(self.parcelas["zona"])
        self.assertLessEqual(zonas, set(self.datos["intervalos"]))

    def test_metricas_coinciden_con_metricas_json(self):
        metricas = json.loads((SALIDA / "metricas.json").read_text(encoding="utf-8"))
        cv = metricas["validacion_cruzada_138"]
        self.assertAlmostEqual(self.datos["metricas"]["cv138"]["R2"], cv["R2"], places=3)
        self.assertAlmostEqual(self.datos["metricas"]["prueba"]["RMSE_t_ha"], metricas["prueba"]["RMSE_t_ha"], places=3)
        self.assertEqual(len(self.datos["prueba"]), metricas["n_prueba"])

    def test_datos_js_esta_al_dia_con_las_salidas(self):
        especificacion = importlib.util.spec_from_file_location("construir_datos", DASH / "construir_datos.py")
        construir = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(construir)
        nuevos = construir.construir()
        self.assertEqual([p["id"] for p in nuevos["parcelas"]], [p["id"] for p in self.datos["parcelas"]])
        np.testing.assert_allclose([p["estimado"] for p in nuevos["parcelas"]],
                                   [p["estimado"] for p in self.datos["parcelas"]], atol=1e-9)
        self.assertEqual(nuevos["modelo"]["coeficientes"], self.datos["modelo"]["coeficientes"])
        self.assertEqual(nuevos["metricas"], self.datos["metricas"])
        self.assertEqual(construir.construir_territorio(), cargar_territorio())


@unittest.skipUnless(NODE, "Node no está instalado")
class ModeloJavaScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        salida = subprocess.run([NODE, str(DASH / "probar_modelo.js")], check=True, text=True,
                                capture_output=True, cwd=HERE).stdout
        cls.resultado = json.loads(salida)

    def test_reproduce_las_59_predicciones_de_python(self):
        self.assertEqual(self.resultado["n_ocultas"], 59)
        self.assertLess(self.resultado["max_dif_ocultas"], 1e-3)

    def test_parcela_conocida_sin_contarse_a_si_misma_se_comporta_como_validacion(self):
        metricas = json.loads((SALIDA / "metricas.json").read_text(encoding="utf-8"))
        mae_cv = metricas["validacion_cruzada_138"]["MAE_t_ha"]
        self.assertAlmostEqual(self.resultado["mae_conocidas_sin_si_misma"], mae_cv, delta=0.03)
        self.assertLess(self.resultado["max_dif_cero_distancia"], 0.01)

    def test_proyeccion_de_coordenadas_coincide_con_x_km_y_km(self):
        self.assertLess(self.resultado["proyeccion_dif"], 1e-3)

    def test_probabilidad_es_una_fraccion_de_los_residuos(self):
        self.assertTrue(0 <= self.resultado["prob_ejemplo"] <= 1)

    def test_sintaxis_de_los_scripts(self):
        for nombre in ("app.js", "modelo.js", "probar_modelo.js"):
            with self.subTest(archivo=nombre):
                subprocess.run([NODE, "--check", str(DASH / nombre)], check=True, capture_output=True)


def cargar_territorio() -> dict:
    texto = (DASH / "territorio.js").read_text(encoding="utf-8")
    cuerpo = re.fullmatch(r"window\.TERRITORIO=(.*);\s*", texto, flags=re.S)
    assert cuerpo, "territorio.js no tiene el formato esperado"
    return json.loads(cuerpo.group(1))


class TerritorioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = cargar_territorio()
        cls.csv = pd.read_csv(HERE / "salidas_extractor" / "datos_por_imagen.csv", encoding="utf-8-sig",
                              usecols=["ID_POLIGONO", "variable", "anio", "mes", "valor", "estado_extraccion"])

    def test_todos_los_registros_del_extractor_estan_y_con_el_mismo_valor(self):
        self.assertEqual(len(self.t["registros"]), len(self.csv))
        self.assertEqual(len(self.t["ids"]), 197)
        nombres = [v["id"] for v in self.t["variables"]]
        self.assertEqual(set(nombres), {"PREC", "Tmin", "Tmax", "ELEVACION", "PENDIENTE"})
        exportado = pd.DataFrame(self.t["registros"], columns=["v", "anio", "mes", "i", "valor", "cob", "est"])
        exportado["variable"] = exportado["v"].map(lambda i: nombres[i])
        exportado["ID_POLIGONO"] = exportado["i"].map(lambda i: self.t["ids"][i])
        llaves = ["ID_POLIGONO", "variable", "anio", "mes"]
        unido = self.csv.merge(exportado, on=llaves, how="outer", indicator=True, suffixes=("_csv", "_js"))
        self.assertTrue((unido["_merge"] == "both").all())
        np.testing.assert_allclose(unido["valor_csv"], unido["valor_js"], atol=1e-3)
        parcial = unido["estado_extraccion"].ne("OK")
        self.assertEqual(parcial.tolist(), unido["est"].eq(1).tolist())

    def test_capas_estaticas_no_tienen_fecha_y_las_mensuales_cubren_2022_2025(self):
        por_variable = {v["id"]: v for v in self.t["variables"]}
        self.assertTrue(por_variable["ELEVACION"]["estatica"] and por_variable["PENDIENTE"]["estatica"])
        self.assertEqual(self.t["anios"], [2022, 2023, 2024, 2025])
        mensual = pd.DataFrame(self.t["registros"], columns=["v", "anio", "mes", "i", "valor", "cob", "est"])
        for indice, variable in enumerate(self.t["variables"]):
            fechas = mensual[mensual["v"] == indice]
            if variable["estatica"]:
                self.assertTrue(fechas["anio"].isna().all() and fechas["mes"].isna().all(), variable["id"])
            else:
                self.assertEqual(fechas.groupby(["anio", "mes"]).ngroups, 48, variable["id"])


class PaginaSatelitalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.principal = (DASH / "index.html").read_text(encoding="utf-8")
        cls.satelital = (DASH / "panel_mapa_satelital.html").read_text(encoding="utf-8")
        cls.app = (DASH / "app.js").read_text(encoding="utf-8")

    def test_el_panel_principal_no_activa_fondo_ni_capas(self):
        self.assertNotIn("PANEL_CONFIG", self.principal)
        self.assertNotIn("territorio.js", self.principal)
        self.assertIn('id="f-color"', self.principal)
        self.assertNotIn('id="f-capa"', self.principal)

    def test_la_pagina_satelital_activa_fondo_territorio_y_busqueda(self):
        self.assertIn("window.PANEL_CONFIG=", self.satelital)
        config = json.loads(re.search(r"window\.PANEL_CONFIG=(\{.*?\});", self.satelital).group(1))
        self.assertEqual(config["fondoInicial"], "satelite")
        self.assertTrue(config["fondo"] and config["territorio"] and config["buscador"])
        orden = [self.satelital.index(f) for f in ('src="datos.js"', 'src="territorio.js"', 'src="modelo.js"', 'src="app.js"')]
        self.assertEqual(orden, sorted(orden))
        for ident in ("f-capa", "f-anio", "f-mes", "f-fondo", "f-buscar", "descargar-capa"):
            self.assertIn(f'id="{ident}"', self.satelital)
        self.assertNotIn('id="f-color"', self.satelital)
        self.assertIn("<title>Panel AgroCebada Satelital</title>", self.satelital)

    def test_las_dos_paginas_solo_difieren_en_lo_esperado(self):
        # Todo lo demás del panel principal está idéntico en la versión satelital.
        for fragmento in ('<main>', 'id="vista-simulador"', 'id="vista-credito"', 'id="vista-desempeno"',
                          'id="vista-guia"', 'id="vista-metodo"', 'id="tabla-parcelas"', 'id="csv-texto"'):
            self.assertIn(fragmento, self.principal)
            self.assertIn(fragmento, self.satelital)
        ids_principal = set(re.findall(r'id="([\w-]+)"', self.principal))
        ids_satelital = set(re.findall(r'id="([\w-]+)"', self.satelital))
        self.assertEqual(ids_principal - ids_satelital, {"f-color"})
        self.assertEqual(ids_satelital - ids_principal,
                         {"f-capa", "g-anio", "f-anio", "g-mes", "f-mes", "f-fondo", "f-buscar", "ids-mapa", "descargar-capa"})

    def test_todo_id_que_usa_app_js_existe_en_alguna_pagina_o_se_crea_dinamicamente(self):
        usados = set(re.findall(r'\$\("#([\w-]+)"\)', self.app))
        presentes = set(re.findall(r'id="([\w-]+)"', self.principal)) | set(re.findall(r'id="([\w-]+)"', self.satelital))
        creados = set(re.findall(r'id: "([\w-]+)"', self.app))
        self.assertEqual(usados - presentes - creados, set())

    def test_el_codigo_no_deja_ganchos_de_depuracion(self):
        self.assertNotIn("__panel", self.app)
        self.assertNotIn("console.log", self.app)


class PaginaTests(unittest.TestCase):
    def test_index_referencia_archivos_existentes_y_sin_dependencias_de_datos_externas(self):
        html = (DASH / "index.html").read_text(encoding="utf-8")
        for recurso in re.findall(r'(?:src|href)="([^"#]+)"', html):
            if recurso.startswith("http"):
                self.assertTrue(recurso.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com")), recurso)
            else:
                self.assertTrue((DASH / recurso).exists(), recurso)
        self.assertIn('<html lang="es">', html)
        self.assertIn("<title>Panel AgroCebada</title>", html)

    def test_estilos_definen_tokens_claro_y_oscuro(self):
        css = (DASH / "estilos.css").read_text(encoding="utf-8")
        self.assertIn("prefers-color-scheme: dark", css)
        self.assertIn(':root[data-theme="dark"]', css)
        claros = set(re.findall(r"--[\w-]+(?=:)", css.split("@media (prefers-color-scheme: dark)")[0]))
        oscuros = set(re.findall(r"--[\w-]+(?=:)", css.split(':root[data-theme="dark"]')[1].split("}")[0]))
        self.assertTrue(oscuros <= claros, f"Tokens oscuros sin definición base: {oscuros - claros}")


if __name__ == "__main__":
    unittest.main()
