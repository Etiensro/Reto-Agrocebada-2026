"""Pruebas del protocolo de predicción, sin evaluar la prueba real reservada.

Ejecutar: .venv/bin/python -m unittest tests_prediccion.py
Las pruebas de ajuste usan datos sintéticos; del dataset real se comprueban
solamente integridad, corte temporal, fenología y ausencia de objetivos entre
predictores.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prediccion_bajo_prueba", HERE / "predicción.py")
prediccion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prediccion)
cli = importlib.import_module("agrocebada.cli")


def metadatos_sinteticos():
    ids = pd.Index([f"AGC_{i:03d}" for i in range(1, 198)], name=prediccion.ID)
    return pd.DataFrame({
        prediccion.TARGET: np.concatenate([np.arange(138, dtype=float) / 100 + 1,
                                          np.full(59, np.nan)]),
        "CONJUNTO": ["ENTRENAMIENTO"] * 138 + ["PREDICCION"] * 59,
    }, index=ids)


def pipeline_sintetico():
    X = pd.DataFrame({"vegetacion": [1., 2., np.nan, 4.],
                      "lluvia": [10., np.nan, 30., 40.],
                      prediccion.TARGET: [999., 998., 997., 996.]})
    y = pd.Series([1., 3., 2., 7.])
    config = {"familia": "Ridge", "grupo": "prueba", "k": "all",
              "params": {"alpha": 1.}}
    model = prediccion.crear_pipeline(config, {"prueba": ["vegetacion", "lluvia"]})
    return model.fit(X, y), X


class MetricasTests(unittest.TestCase):
    def test_metricas_con_resultados_calculados_a_mano(self):
        result = prediccion.metricas([2., 4., 8., 16.], [2., 5., 6., 16.])
        expected = {"n": 4, "RMSE_t_ha": np.sqrt(5 / 4), "MAE_t_ha": .75,
                    "R2": 1 - 5 / 115, "sesgo_t_ha": -.25,
                    "MAPE_pct": 12.5, "error_relativo_mediano_pct": 12.5,
                    "dentro_10_pct": 50., "dentro_20_pct": 50.}
        self.assertEqual(set(result), set(expected))
        for name, value in expected.items():
            with self.subTest(metrica=name):
                self.assertAlmostEqual(result[name], value, places=12)

    def test_errores_absolutos_incluyen_ceros_y_porcentuales_los_excluyen(self):
        result = prediccion.metricas([0., 2.], [1., 2.])
        self.assertEqual(result["MAE_t_ha"], .5)
        self.assertEqual(result["MAPE_pct"], 0.)
        self.assertEqual(result["dentro_10_pct"], 100.)
        self.assertIsNone(prediccion.metricas([0.], [1.])["MAPE_pct"])
        self.assertIsNone(prediccion.metricas([2.], [3.])["R2"])

    def test_rechaza_formas_incompatibles_y_valores_no_finitos(self):
        for y, yp in [([1., 2.], [1.]), ([1., np.nan], [1., 2.]),
                      ([1., 2.], [1., np.inf])]:
            with self.subTest(y=y, pred=yp):
                with self.assertRaises(ValueError):
                    prediccion.metricas(y, yp)


class ParticionTests(unittest.TestCase):
    def test_80_20_reproducible_sin_solapamientos_ni_ocultas(self):
        meta = metadatos_sinteticos()
        train, test, hidden = prediccion.dividir_parcelas(meta)
        self.assertEqual((len(train), len(test), len(hidden)), (110, 28, 59))
        self.assertFalse(set(train) & set(test))
        self.assertFalse((set(train) | set(test)) & set(hidden))
        self.assertEqual(set(train) | set(test) | set(hidden), set(meta.index))
        for first, second in zip((train, test, hidden), prediccion.dividir_parcelas(meta)):
            self.assertTrue(first.equals(second))

    def test_particion_independiente_de_rendimientos_y_orden_de_filas(self):
        meta = metadatos_sinteticos()
        changed = meta.copy()
        known = changed[prediccion.TARGET].notna()
        changed.loc[known, prediccion.TARGET] = -changed.loc[known, prediccion.TARGET] * 1e8
        changed = changed.sample(frac=1, random_state=91)
        for first, second in zip(prediccion.dividir_parcelas(meta),
                                 prediccion.dividir_parcelas(changed)):
            self.assertTrue(first.equals(second))

    def test_rechaza_fracciones_fuera_del_rango(self):
        for size in [0., .09, .41, 1.]:
            with self.subTest(test_size=size):
                with self.assertRaises(ValueError):
                    prediccion.dividir_parcelas(metadatos_sinteticos(), test_size=size)


class PipelineTests(unittest.TestCase):
    def test_imputacion_y_escalado_solo_aprenden_de_fit(self):
        model, X = pipeline_sintetico()
        np.testing.assert_allclose(model.named_steps["imputacion"].statistics_, [2., 30.])
        self.assertEqual(model.named_steps["columnas"].get_feature_names_out().tolist(),
                         ["vegetacion", "lluvia"])
        statistics = model.named_steps["imputacion"].statistics_.copy()
        scaling_mean = model.named_steps["escala"].mean_.copy()
        scaling_scale = model.named_steps["escala"].scale_.copy()
        extreme_validation = pd.DataFrame({"vegetacion": [1e12, np.nan],
                                           "lluvia": [np.nan, -1e12],
                                           prediccion.TARGET: [-1e30, 1e30]})
        prediction = model.predict(extreme_validation)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_array_equal(model.named_steps["imputacion"].statistics_, statistics)
        np.testing.assert_array_equal(model.named_steps["escala"].mean_, scaling_mean)
        np.testing.assert_array_equal(model.named_steps["escala"].scale_, scaling_scale)
        no_target = X.drop(columns=prediccion.TARGET)
        np.testing.assert_allclose(model.predict(X), model.predict(no_target))

    def test_modelo_joblib_se_recupera_en_proceso_python_nuevo(self):
        model, X = pipeline_sintetico()
        with tempfile.TemporaryDirectory(prefix="agrocebada_modelo_") as tmp:
            model_file = Path(tmp) / "modelo.joblib"
            data_file = Path(tmp) / "variables.csv"
            joblib.dump(model, model_file)
            X.to_csv(data_file, index=False)
            restored = joblib.load(model_file)
            np.testing.assert_allclose(restored.predict(X), model.predict(X))
            code = ("import json,joblib,pandas as pd,sys; "
                    "model=joblib.load(sys.argv[1]); "
                    "print(json.dumps(model.predict(pd.read_csv(sys.argv[2])).tolist()))")
            child = subprocess.run([sys.executable, "-c", code, str(model_file), str(data_file)],
                                   check=True, text=True, capture_output=True, cwd=tmp)
            np.testing.assert_allclose(json.loads(child.stdout), model.predict(X))

    def test_candidatos_reproducibles_y_representan_todas_las_familias(self):
        expected = {"Media", "Ridge", "ElasticNet", "SVR", "ExtraTrees", "RandomForest", "CatBoost"}
        for mode in ["rapido", "completo"]:
            with self.subTest(modo=mode):
                configs = prediccion.generar_candidatos(mode, seed=42)
                self.assertEqual(configs, prediccion.generar_candidatos(mode, seed=42))
                self.assertLessEqual(expected, {c["familia"] for c in configs})
                self.assertEqual(len({c["nombre"] for c in configs}), len(configs))
        self.assertGreater(len(prediccion.generar_candidatos("completo")),
                           len(prediccion.generar_candidatos("rapido")))


class FenologiaTests(unittest.TestCase):
    def test_curva_sintetica_recupera_base_pico_y_fechas(self):
        inicio, fin = pd.Timestamp(2025, 4, 1), pd.Timestamp(2025, 10, 31)
        fechas = pd.date_range(inicio, fin, freq="5D")
        dias = np.asarray((fechas - inicio).days, dtype=float)
        valores = 0.15 + 0.6 * np.exp(-((dias - 140) / 30) ** 2)
        obs = pd.DataFrame({"ID_POLIGONO": "AGC_001", "fecha": fechas, "valor": valores})
        ids = pd.Index(["AGC_001", "AGC_002"], name=prediccion.ID)
        tabla = prediccion.metricas_fenologicas(obs, ids, inicio, fin, 5)
        fila = tabla.loc["AGC_001"]
        self.assertAlmostEqual(fila["base"], 0.15, places=3)
        # La mediana móvil de tres fechas recorta un poco el máximo puntual.
        self.assertAlmostEqual(fila["pico"], 0.75, delta=0.03)
        self.assertAlmostEqual(fila["dia_pico"], 140, delta=5)
        self.assertLess(fila["inicio"], 140)
        self.assertGreater(fila["fin"], 140)
        self.assertAlmostEqual(fila["duracion"], fila["fin"] - fila["inicio"])
        self.assertGreater(fila["integral"], 0)
        self.assertTrue(tabla.loc["AGC_002"].isna().all())

    def test_menos_de_cuatro_fechas_quedan_ausentes(self):
        inicio, fin = pd.Timestamp(2025, 4, 1), pd.Timestamp(2025, 10, 31)
        obs = pd.DataFrame({"ID_POLIGONO": "AGC_001", "fecha": pd.date_range(inicio, periods=3, freq="20D"),
                            "valor": [0.2, 0.5, 0.3]})
        tabla = prediccion.metricas_fenologicas(obs, pd.Index(["AGC_001"]), inicio, fin, 8)
        self.assertTrue(tabla.isna().all().all())


def datos_geo_sinteticos(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x_km": rng.uniform(0, 20, n), "y_km": rng.uniform(0, 20, n),
                      "log_area_ha": rng.normal(1, .5, n), "senal": rng.normal(0, 1, n)})
    for i in range(30):
        X[f"ruido_{i}"] = rng.normal(0, 1, n)
    X.loc[X.sample(frac=.1, random_state=seed).index, "ruido_0"] = np.nan
    y = 3 + 1.5 * (X["x_km"] > 10) + 0.4 * X["senal"] + rng.normal(0, .1, n)
    return X, y


class GeoResidualTests(unittest.TestCase):
    def test_cribado_elige_la_senal_y_mejora_la_vecindad_sola(self):
        X, y = datos_geo_sinteticos()
        tr, te = X.index[:90], X.index[90:]
        candidatas = tuple(c for c in X if c.startswith(("senal", "ruido")))
        modelo = prediccion.GeoResidualRegressor(candidatas=candidatas, k_sel=1, alpha=1.0).fit(X.loc[tr], y.loc[tr])
        self.assertEqual(modelo.seleccionadas_, ["log_area_ha", "senal"])
        solo_vecinos = prediccion.GeoResidualRegressor(k_sel=0, fijas=()).fit(X.loc[tr], y.loc[tr])
        error = lambda m: np.sqrt(np.mean((m.predict(X.loc[te]) - y.loc[te]) ** 2))
        self.assertLess(error(modelo), error(solo_vecinos))
        self.assertTrue(np.isfinite(modelo.predict(X.loc[te])).all())

    def test_ajuste_reproducible_y_prediccion_no_modifica_el_estado(self):
        X, y = datos_geo_sinteticos(seed=3)
        candidatas = tuple(c for c in X if c.startswith(("senal", "ruido")))
        a = prediccion.GeoResidualRegressor(candidatas=candidatas, k_sel=3).fit(X, y)
        b = prediccion.GeoResidualRegressor(candidatas=candidatas, k_sel=3).fit(X.copy(), y.copy())
        self.assertEqual(a.seleccionadas_, b.seleccionadas_)
        antes = a.seleccionadas_.copy()
        extremo = X.copy()
        extremo[list(candidatas)] = 1e9
        self.assertTrue(np.isfinite(a.predict(extremo)).all())
        self.assertEqual(a.seleccionadas_, antes)
        np.testing.assert_allclose(a.predict(X), b.predict(X))

    def test_candidatos_incluyen_geo_residual_y_pipeline_lo_construye(self):
        configs = prediccion.generar_candidatos("completo", seed=42)
        geo = [c for c in configs if c["familia"] == "GeoResidual"]
        self.assertTrue(geo)
        self.assertIn(0, {c["params"]["k_sel"] for c in geo})
        modelo = prediccion.crear_pipeline(geo[1], {"fenologia": ["a", "b"]}, seed=5)
        self.assertIsInstance(modelo, prediccion.GeoResidualRegressor)
        self.assertEqual(modelo.candidatas, ("a", "b"))
        self.assertEqual(modelo.semilla, 5)

    def test_modelo_geo_residual_se_recupera_en_proceso_nuevo(self):
        sys.path.insert(0, str(HERE))
        try:
            modulo = importlib.import_module("agrocebada.modelos")
        finally:
            sys.path.remove(str(HERE))
        X, y = datos_geo_sinteticos(seed=4)
        modelo = modulo.GeoResidualRegressor(candidatas=("senal",), k_sel=1).fit(X, y)
        with tempfile.TemporaryDirectory(prefix="agrocebada_geo_") as tmp:
            archivo, datos = Path(tmp) / "modelo.joblib", Path(tmp) / "X.csv"
            joblib.dump(modelo, archivo)
            X.to_csv(datos, index=False)
            code = ("import json,sys,joblib,pandas as pd; sys.path.insert(0, sys.argv[3]); "
                    "m=joblib.load(sys.argv[1]); print(json.dumps(m.predict(pd.read_csv(sys.argv[2])).tolist()))")
            child = subprocess.run([sys.executable, "-c", code, str(archivo), str(datos), str(HERE)],
                                   check=True, text=True, capture_output=True, cwd=tmp)
            np.testing.assert_allclose(json.loads(child.stdout), modelo.predict(X))


class IntervalosTests(unittest.TestCase):
    def test_cuantil_conformal_con_correccion_finita(self):
        residuos = np.arange(1, 21, dtype=float)
        zonas = pd.Series(["A"] * 20)
        out = prediccion.intervalos_por_zona(residuos, zonas, niveles=(0.8,))
        # ceil(21 * 0.8) / 20 = 0.85 -> cuantil 0.85 de 1..20
        self.assertAlmostEqual(out["A"]["80"], np.quantile(residuos, .85))
        self.assertEqual(out["A"]["n"], 20)

    def test_municipios_pequenos_y_desconocidos_usan_grupo_o_global(self):
        municipios = pd.Series(["A"] * 15 + ["B"] * 3 + ["C"] * 2)
        zonas = prediccion.zonas_intervalo(municipios)
        self.assertEqual(set(zonas), {"A", "Otros municipios"})
        intervalos = {"A": {"80": .1, "90": .2}, "Otros municipios": {"80": .3, "90": .4}}
        estimado = pd.Series([4., 4., 4.], index=["p1", "p2", "p3"])
        tabla = prediccion.tabla_intervalos(estimado, pd.Series(["A", "Z", "B"], index=estimado.index), intervalos)
        self.assertEqual(tabla["ZONA_INTERVALO"].tolist(), ["A", "Otros municipios", "Otros municipios"])
        np.testing.assert_allclose(tabla["LIMITE_INFERIOR_80_T_HA"], [3.9, 3.7, 3.7])
        sin_grupo = prediccion.tabla_intervalos(estimado, pd.Series(["Z"] * 3, index=estimado.index),
                                                {"Todas": {"80": .5, "90": .6}})
        self.assertEqual(set(sin_grupo["ZONA_INTERVALO"]), {"Todas"})


class PreparacionRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = HERE / "DATASET_RETO_AGRO_2026"
        cls.extraidos = HERE / "salidas_extractor"
        cls.X, cls.meta, cls.groups, cls.quality = prediccion.preparar_datos(cls.dataset, cls.extraidos)

    def test_197_filas_unicas_numericas_y_objetivos_excluidos(self):
        self.assertEqual(len(self.X), 197)
        self.assertTrue(self.X.index.is_unique)
        self.assertEqual(set(self.X.index), {f"AGC_{i:03d}" for i in range(1, 198)})
        self.assertTrue(self.X.index.equals(self.meta.index))
        for prohibited in [prediccion.ID, prediccion.TARGET, "CONJUNTO"]:
            self.assertNotIn(prohibited, self.X.columns)
        self.assertFalse(any("rendimiento" in c.lower() or "conjunto" in c.lower()
                             or "id_poligono" in c.lower() for c in self.X.columns))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in self.X.dtypes))
        self.assertFalse(np.isinf(self.X.to_numpy()).any())
        self.assertEqual(self.quality["etiquetas_visibles"], 138)
        self.assertEqual(self.quality["etiquetas_ocultas"], 59)
        for group, columns in self.groups.items():
            with self.subTest(grupo=group):
                self.assertTrue(columns)
                self.assertEqual(len(columns), len(set(columns)))
                self.assertLessEqual(set(columns), set(self.X.columns))

    def test_fenologia_historial_y_coordenadas_km(self):
        fen = [c for c in self.X if c.startswith("fen_")]
        hist = [c for c in self.X if c.startswith("fenh_")]
        self.assertEqual(set(self.groups["fenologia"]), set(fen) | set(hist))
        self.assertIn("fen_s2_mra_base", fen)
        self.assertIn("fenh_s2_evi_media_base", hist)
        for familia in ["fen_s2_ndvi", "fen_planet_ndvi", "fen_landsat_vi6t"]:
            with self.subTest(familia=familia):
                self.assertLess(self.X[f"{familia}_pico"].isna().mean(), .05)
        extension = np.hypot(self.X["x_km"].max() - self.X["x_km"].min(),
                             self.X["y_km"].max() - self.X["y_km"].min())
        self.assertTrue(20 < extension < 300)
        self.assertTrue(any(c.startswith("municipio_") for c in self.X))
        self.assertNotIn("x_km", self.groups["contexto"])

    def test_corte_agosto_excluye_meses_y_observaciones_posteriores(self):
        original_read = pd.read_csv
        mutated_satellite_files = []

        def read_with_extreme_future(path, *args, **kwargs):
            frame = original_read(path, *args, **kwargs)
            if Path(path).name.startswith("Conjunto_datos_"):
                future = pd.to_datetime(frame["fecha_captura"], format="%d/%m/%Y").gt("2025-08-31")
                for column in frame.columns:
                    if column.endswith(("_promedio", "_std")):
                        frame.loc[future, column] = 1e20
                mutated_satellite_files.append(Path(path).name)
            elif Path(path).name == "clima_mensual.csv":
                future = frame["anio"].gt(2025) | (frame["anio"].eq(2025) & frame["mes"].gt(8))
                for column in ["precipitacion_mm", "tmin_promedio_c", "tmax_promedio_c"]:
                    frame.loc[future, column] = 1e20
            return frame

        cutoff_X, _, _, quality = prediccion.preparar_datos(self.dataset, self.extraidos, mes_final=8)
        with patch.object(pd, "read_csv", side_effect=read_with_extreme_future):
            mutated_X, _, _, _ = prediccion.preparar_datos(self.dataset, self.extraidos, mes_final=8)
        self.assertEqual(quality["fecha_corte"], "2025-08-31")
        self.assertEqual(quality["meses"], [4, 5, 6, 7, 8])
        self.assertFalse(any(c.endswith(("_m09", "_m10")) for c in cutoff_X.columns))
        self.assertEqual(len(mutated_satellite_files), 2)
        assert_frame_equal(cutoff_X, mutated_X)

    def test_rendimientos_repetidos_en_extractor_no_se_leen_como_predictores(self):
        original_read = pd.read_csv
        checked = []

        def read_with_corrupted_duplicate_targets(path, *args, **kwargs):
            if Path(path).parent == self.extraidos:
                requested = kwargs.pop("usecols", None)
                self.assertIsNotNone(requested)
                self.assertNotIn(prediccion.TARGET, requested)
                self.assertNotIn("CONJUNTO", requested)
                frame = original_read(path, *args, **kwargs)
                frame[prediccion.TARGET] = np.arange(len(frame)) * 1e20
                frame["CONJUNTO"] = "OBJETIVO_DUPLICADO_CORRUPTO"
                checked.append(Path(path).name)
                return frame.loc[:, requested]
            return original_read(path, *args, **kwargs)

        with patch.object(pd, "read_csv", side_effect=read_with_corrupted_duplicate_targets):
            X, _, _, _ = prediccion.preparar_datos(self.dataset, self.extraidos)
        self.assertEqual(set(checked), {"parcelas_datos.csv", "clima_mensual.csv", "topografia_parcelas.csv"})
        assert_frame_equal(self.X, X)


class ProtocoloSeleccionTests(unittest.TestCase):
    def test_seleccion_solo_recibe_110_parcelas_y_no_evalua_prueba(self):
        parameters = inspect.signature(prediccion.seleccionar_modelo).parameters
        self.assertNotIn("X_test", parameters)
        self.assertNotIn("y_test", parameters)
        meta = metadatos_sinteticos()
        X = pd.DataFrame({"vegetacion": np.linspace(0, 1, 197)}, index=meta.index)
        expected_train, expected_test, hidden = prediccion.dividir_parcelas(meta)
        quality = {"fuentes": {"codigo_prueba": str(HERE / "predicción.py")}}
        received = []

        def select_spy(Xtrain, ytrain, groups, output, mode, seed, threads):
            self.assertTrue(Xtrain.index.equals(expected_train))
            self.assertTrue(ytrain.index.equals(expected_train))
            self.assertFalse(set(Xtrain.index) & (set(expected_test) | set(hidden)))
            assert_frame_equal(Xtrain, X.loc[expected_train])
            pd.testing.assert_series_equal(ytrain, meta.loc[expected_train, prediccion.TARGET])
            received.append(len(Xtrain))
            return None, {"configuracion": {"nombre": "sintetico"}, "cv_rmse": 1.}, pd.DataFrame(), []

        with tempfile.TemporaryDirectory(prefix="agrocebada_protocolo_") as tmp:
            with patch.object(cli, "preparar_datos", return_value=(X, meta, {"contexto": list(X)}, quality)), \
                 patch.object(cli, "seleccionar_modelo", side_effect=select_spy), \
                 patch.object(cli, "metricas", side_effect=AssertionError("No evaluar prueba en selección")), \
                 patch.object(cli, "diagnostico_espacial", side_effect=AssertionError("No diagnóstico en solo selección")), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = cli.main(["--solo-seleccion", "--salida", tmp])
            self.assertEqual(result, 0)
            self.assertEqual(received, [110])
            self.assertFalse((Path(tmp) / "metricas.json").exists())
            self.assertFalse((Path(tmp) / "predicciones_prueba_20.csv").exists())
            self.assertFalse((Path(tmp) / "predicciones_prueba.csv").exists())
            split = pd.read_csv(Path(tmp) / "particion.csv")
            self.assertEqual(split["USO"].value_counts().to_dict(),
                             {"ENTRENAMIENTO_80": 110, "PREDICCION_CONCURSO": 59, "PRUEBA_20": 28})

    def test_inferencia_con_modelo_guardado_conserva_archivos_de_auditoria(self):
        meta = metadatos_sinteticos()
        X = pd.DataFrame({"vegetacion": np.linspace(0, 1, 197)}, index=meta.index)
        hidden = meta.index[meta[prediccion.TARGET].isna()]
        model = Mock()

        def predict_only_hidden(variables):
            assert_frame_equal(variables, X.loc[hidden])
            return np.full(len(hidden), 3.5)

        model.predict.side_effect = predict_only_hidden
        bundle = {"modelo": model, "columnas": list(X),
                  "preparacion": {"nubosidad_max": 20, "cobertura_min": 80, "mes_final": 10}}
        with tempfile.TemporaryDirectory(prefix="agrocebada_inferencia_") as tmp:
            audit_names = ["manifiesto.json", "particion.csv", "calidad_datos.json",
                           "grupos_variables.json", "variables_por_parcela.csv",
                           "seleccion_congelada.json", "comparacion_cv.csv", "metricas.json"]
            originals = {}
            for name in audit_names:
                content = f"Auditoría original: {name}\n".encode("utf-8")
                (Path(tmp) / name).write_bytes(content)
                originals[name] = content
            with patch.object(cli, "preparar_datos", return_value=(X, meta, {}, {})), \
                 patch.object(cli.joblib, "load", return_value=bundle), \
                 patch.object(cli, "dividir_parcelas", side_effect=AssertionError("No volver a dividir en inferencia")), \
                 patch.object(cli, "seleccionar_modelo", side_effect=AssertionError("No entrenar en inferencia")), \
                 patch.object(cli, "metricas", side_effect=AssertionError("No evaluar en inferencia")), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = cli.main(["--modelo", str(Path(tmp) / "modelo.joblib"), "--salida", tmp])
            self.assertEqual(result, 0)
            model.predict.assert_called_once()
            for name, content in originals.items():
                with self.subTest(archivo=name):
                    self.assertEqual((Path(tmp) / name).read_bytes(), content)
            predictions = pd.read_csv(Path(tmp) / "predicciones_59.csv")
            self.assertEqual(predictions[prediccion.ID].tolist(), hidden.tolist())
            np.testing.assert_allclose(predictions[prediccion.TARGET], 3.5)


if __name__ == "__main__":
    unittest.main()
