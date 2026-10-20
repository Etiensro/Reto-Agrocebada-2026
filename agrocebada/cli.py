"""Línea de comandos: orquesta preparación, selección, prueba y exportación."""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from threadpoolctl import threadpool_limits

from . import __doc__ as DESCRIPCION
from .constantes import ID, TARGET
from .datos import preparar_datos
from .evaluacion import bootstrap_metricas, dividir_parcelas, evaluar_cv, metricas
from .intervalos import intervalos_por_zona, tabla_intervalos, variables_seleccionadas, zonas_intervalo
from .preguntas_guia import analisis_preguntas_guia
from .reporte import escribir_reporte, figura_preguntas, figuras_prueba
from .seleccion import diagnostico_espacial, importancia_grupos, seleccionar_modelo
from .utilidades import guardar_csv, guardar_json, hash_archivo, hash_codigo


HERE = Path(__file__).resolve().parent.parent  # carpeta del proyecto


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPCION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=HERE / "DATASET_RETO_AGRO_2026")
    parser.add_argument("--extraidos", type=Path, default=HERE / "salidas_extractor")
    parser.add_argument("--salida", type=Path, default=HERE / "salidas_prediccion")
    parser.add_argument("--test-size", type=float, default=.20)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--modo", choices=("rapido", "completo"), default="completo")
    parser.add_argument("--hilos", type=int, default=2)
    parser.add_argument("--nubosidad-max", type=float, default=20)
    parser.add_argument("--cobertura-min", type=float, default=80)
    parser.add_argument("--mes-final", type=int, default=10, choices=range(4,11))
    parser.add_argument("--reextraer", action="store_true", help="Volver a extraer TIFF con extraccion.py.")
    parser.add_argument("--solo-preparar", action="store_true")
    parser.add_argument("--solo-seleccion", action="store_true", help="Seleccionar con CV sin consultar la prueba.")
    parser.add_argument("--modelo", type=Path, help="Predecir con un modelo guardado, sin entrenar.")
    parser.add_argument("--sin-preguntas-guia", action="store_true",
                        help="Omitir el análisis de las preguntas guía (más rápido).")
    args = parser.parse_args(argv)
    if args.hilos < 1 or not 0 <= args.nubosidad_max <= 100 or not 0 <= args.cobertura_min <= 100:
        parser.error("Revisar hilos, nubosidad y cobertura.")
    salida = args.salida.resolve()
    salida.mkdir(parents=True, exist_ok=True)
    required = [args.extraidos / f for f in ("clima_mensual.csv", "topografia_parcelas.csv", "parcelas_datos.csv")]
    if args.reextraer or not all(p.exists() for p in required):
        print("Extrayendo clima y topografía con extraccion.py...", flush=True)
        subprocess.run([sys.executable, str(HERE / "extraccion.py"), "--dataset", str(args.dataset), "--salida", str(args.extraidos), "--sin-mapa"], check=True)
    print("Preparando una fila por parcela...", flush=True)
    X, meta, grupos, calidad = preparar_datos(args.dataset, args.extraidos, args.nubosidad_max, args.cobertura_min, args.mes_final)
    if TARGET in X.columns or ID in X.columns or "CONJUNTO" in X.columns:
        raise ValueError("Una columna prohibida ha entrado como predictor.")
    if args.modelo:
        hidden = meta.index[meta[TARGET].isna()].sort_values()
        bundle = joblib.load(args.modelo)
        expected = bundle["columnas"]
        if set(expected) - set(X.columns):
            raise ValueError("Faltan columnas necesarias para el modelo guardado.")
        if bundle["preparacion"] != dict(nubosidad_max=args.nubosidad_max, cobertura_min=args.cobertura_min, mes_final=args.mes_final):
            raise ValueError("La configuración de preparación difiere de la del modelo guardado.")
        pred = bundle["modelo"].predict(X.loc[hidden, expected])
        guardar_csv(pd.DataFrame({ID:hidden, TARGET:pred}), salida / "predicciones_59.csv")
        if bundle.get("intervalos") and "Municipio" in meta:
            estimado = pd.Series(pred, index=hidden, name=TARGET)
            limites = tabla_intervalos(estimado, meta.loc[hidden, "Municipio"], bundle["intervalos"])
            guardar_csv(pd.concat([estimado, limites], axis=1).rename_axis(ID).reset_index(),
                        salida / "predicciones_59_intervalos.csv")
        print(f"Predicciones recuperadas sin entrenar: {salida / 'predicciones_59.csv'}")
        return 0
    tr_ids, te_ids, hidden = dividir_parcelas(meta, args.test_size, args.semilla)
    assert set(X.index) == set(meta.index) and X.index.is_unique
    calidad["cloud_max"] = args.nubosidad_max
    guardar_json(salida / "calidad_datos.json", calidad)
    guardar_json(salida / "grupos_variables.json", grupos)
    guardar_csv(X.rename_axis(ID).reset_index(), salida / "variables_por_parcela.csv")
    particion = meta[["CONJUNTO"]].copy()
    particion["USO"] = "PREDICCION_CONCURSO"
    particion.loc[tr_ids, "USO"] = f"ENTRENAMIENTO_{round((1-args.test_size)*100):02d}"
    particion.loc[te_ids, "USO"] = f"PRUEBA_{round(args.test_size*100):02d}"
    guardar_csv(particion.rename_axis(ID).reset_index(), salida / "particion.csv")
    sources = calidad.get("fuentes", {})
    source_paths = list(dict.fromkeys(str(p) for p in (sources.values() if isinstance(sources, dict) else sources)))
    if not source_paths:
        source_paths = [str(p) for p in required] + [str(args.dataset / f) for f in ("Conjunto_datos_BASICO_AgroCebada2026.csv", "Conjunto_datos_PRO_AgroCebada.csv", "ID_area_rendimiento_70_30_Reto_AgroCebada.csv")]
    manifesto = dict(fecha_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                     versiones={"numpy":np.__version__, "pandas":pd.__version__, "scipy":scipy.__version__, "sklearn":sklearn.__version__, "catboost":__import__('catboost').__version__, "joblib":joblib.__version__},
                     argumentos=vars(args), fuentes_sha256={str(p):hash_archivo(p) for p in source_paths},
                     codigo_sha256=hash_codigo(HERE), entrenamiento_ids=tr_ids.tolist(), prueba_ids=te_ids.tolist(),
                     prediccion_ids=hidden.tolist(), n_variables=len(X.columns), prueba_usada_para_seleccion=False)
    guardar_json(salida / "manifiesto.json", manifesto)
    print(f"  {len(X)} parcelas; {X.shape[1]} variables; {len(tr_ids)} entrenamiento, {len(te_ids)} prueba reservada, {len(hidden)} ocultas.", flush=True)
    if args.solo_preparar:
        print(f"Preparación terminada: {salida}")
        return 0
    # No acceder a los rendimientos de te_ids durante selección/diagnóstico/importancia.
    Xtr, ytr = X.loc[tr_ids], meta.loc[tr_ids, TARGET].astype(float)
    print("Comparando modelos mediante validación interna...", flush=True)
    with threadpool_limits(limits=args.hilos):
        model, seleccion, ranking, splits = seleccionar_modelo(Xtr, ytr, grupos, salida, args.modo, args.semilla, args.hilos)
        print(f"Selección congelada: {seleccion['configuracion']['nombre']} | CV RMSE {seleccion['cv_rmse']:.4f}", flush=True)
        if args.solo_seleccion:
            print("Prueba reservada sin evaluar.")
            return 0
        print("Comprobando transferencia entre celdas climáticas e importancia de variables...", flush=True)
        spatial, spatial_frame = diagnostico_espacial(model, Xtr, ytr, meta, args.semilla)
        if spatial_frame is not None:
            guardar_csv(spatial_frame, salida / "diagnostico_espacial_oof.csv")
        importance = importancia_grupos(model, Xtr, ytr, splits, args.semilla)
        guardar_csv(importance, salida / "importancia_variables.csv")
        evaluado = clone(model).fit(Xtr, ytr)
        # Primera y única consulta del objetivo reservado para la evaluación.
        Xtest, ytest = X.loc[te_ids], meta.loc[te_ids, TARGET].astype(float)
        pred = evaluado.predict(Xtest)
        reference = np.full(len(ytest), ytr.mean())
        resultado = dict(n_entrenamiento=len(tr_ids), n_prueba=len(te_ids), n_conocidas=len(tr_ids)+len(te_ids), n_ocultas=len(hidden),
                         prueba=metricas(ytest, pred), baseline_prueba=metricas(ytest, reference),
                         bootstrap_prueba=bootstrap_metricas(ytest, pred, args.semilla),
                         diagnostico_espacial_entrenamiento=spatial,
                         seleccion=seleccion, test_utilizado_para_seleccion=False)
        resultado["mejora_RMSE_vs_media_pct"] = float(100*(1-resultado["prueba"]["RMSE_t_ha"]/resultado["baseline_prueba"]["RMSE_t_ha"]))
        guardar_json(salida / "metricas.json", resultado)
        holdout = pd.DataFrame({ID:te_ids, "REAL_T_HA":ytest.to_numpy(), "PREDICHO_T_HA":pred, "BASELINE_T_HA":reference,
                                "ERROR_ABS_T_HA":np.abs(pred-ytest.to_numpy()), "ERROR_RELATIVO_PCT":np.abs(pred-ytest.to_numpy())/np.abs(ytest.to_numpy())*100})
        guardar_csv(holdout, salida / "predicciones_prueba.csv")
        print("\nPrueba independiente:", flush=True)
        for k,v in resultado["prueba"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}", flush=True)
        bundle_common = dict(columnas=X.columns.tolist(), seleccion=seleccion,
                             preparacion=dict(nubosidad_max=args.nubosidad_max, cobertura_min=args.cobertura_min, mes_final=args.mes_final),
                             unidad="t/ha", ciclo=2025)
        joblib.dump(dict(bundle_common, modelo=evaluado, etapa="evaluacion_80", ids_entrenamiento=tr_ids.tolist()), salida / "modelo_evaluacion.joblib")
        all_ids = tr_ids.union(te_ids).sort_values()
        y_all = meta.loc[all_ids, TARGET].astype(float)
        final = clone(model).fit(X.loc[all_ids], y_all)
        hidden_pred = final.predict(X.loc[hidden])
        if not np.isfinite(hidden_pred).all():
            raise ValueError("Predicciones finales no finitas.")
        # Configuración ya congelada: la CV sobre 138 no cambia la elección. Da una
        # estimación más estable y residuos fuera de fold para mapa e intervalos.
        print("Validación cruzada de la configuración congelada sobre las 138 conocidas...", flush=True)
        splits_all = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=args.semilla + 5).split(X.loc[all_ids]))
        cv_all = evaluar_cv(model, X.loc[all_ids], y_all, splits_all)
        resultado["validacion_cruzada_138"] = dict(
            metricas(y_all, cv_all["oof"]), RMSE_std_folds=cv_all["cv_rmse_std"], folds=5, repeticiones=3,
            nota="Configuración fija; incluye las 110 usadas para elegirla, por lo que es complementaria a la prueba.")
        residuos = y_all.to_numpy() - cv_all["oof"]
        intervalos = intervalos_por_zona(residuos, zonas_intervalo(meta.loc[all_ids, "Municipio"]))
        intervalos["Todas"] = intervalos_por_zona(residuos, pd.Series("Todas", index=all_ids))["Todas"]
        resultado["intervalos_prediccion_t_ha"] = intervalos
        resultado["variables_modelo_final"] = variables_seleccionadas(final)
        guardar_json(salida / "metricas.json", resultado)
        joblib.dump(dict(bundle_common, modelo=final, etapa="final_138", ids_entrenamiento=all_ids.tolist(),
                         intervalos=intervalos), salida / "modelo_final.joblib")
        guardar_csv(pd.DataFrame({ID:hidden, TARGET:hidden_pred}), salida / "predicciones_59.csv")
        export = meta[["AREA_HA", "Estado", "Municipio", "longitud", "latitud", "CONJUNTO", TARGET]].copy()
        export["RENDIMIENTO_ESTIMADO_T_HA"] = np.nan
        export.loc[all_ids, "RENDIMIENTO_ESTIMADO_T_HA"] = cv_all["oof"]
        export.loc[hidden, "RENDIMIENTO_ESTIMADO_T_HA"] = hidden_pred
        export["ORIGEN_ESTIMACION"] = "VALIDACION_CRUZADA_SIN_SU_ETIQUETA"
        export.loc[hidden, "ORIGEN_ESTIMACION"] = "PREDICCION_SIN_ETIQUETA"
        export["ERROR_T_HA"] = export["RENDIMIENTO_ESTIMADO_T_HA"] - export[TARGET]
        export = pd.concat([export, tabla_intervalos(export["RENDIMIENTO_ESTIMADO_T_HA"], export["Municipio"], intervalos)], axis=1)
        export["PRODUCCION_ESTIMADA_T"] = export["AREA_HA"]*export["RENDIMIENTO_ESTIMADO_T_HA"]
        guardar_csv(export.rename_axis(ID).reset_index(), salida / "resultados_para_mapa.csv")
        con_intervalo = ["RENDIMIENTO_ESTIMADO_T_HA"] + [c for c in export if c.startswith(("ZONA_", "LIMITE_"))]
        guardar_csv(export.loc[hidden, con_intervalo].rename(columns={"RENDIMIENTO_ESTIMADO_T_HA": TARGET})
                    .rename_axis(ID).reset_index(), salida / "predicciones_59_intervalos.csv")
        guia = None
        if not args.sin_preguntas_guia:
            print("Respondiendo preguntas guía con validación cruzada sobre las 138...", flush=True)
            guia = analisis_preguntas_guia(X.loc[all_ids], y_all, meta, salida, args.semilla, args.hilos)
    figuras_prueba(ytest.to_numpy(), pred, reference, salida)
    if guia is not None:
        figura_preguntas(guia, salida)
    escribir_reporte(salida, resultado, seleccion, calidad, importance, guia)
    print(f"\nArchivos guardados en {salida}", flush=True)
    return 0
