"""Modelo geográfico-residual y catálogo de candidatos comparados."""
from __future__ import annotations

import warnings

import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class GeoResidualRegressor(BaseEstimator, RegressorMixin):
    """Rendimiento = vecindad geográfica + corrección por superficie y satélite.

    1) KNN ponderado por distancia en km: parcelas cercanas comparten clima,
       suelo y manejo (en los datos, la semivarianza a <0.5 km es la mitad).
    2) Ridge sobre el residuo FUERA DE PLIEGUE de (1), con las variables fijas y
       las k_sel candidatas más correlacionadas con ese residuo. Se descartan
       candidatas casi duplicadas (|r| > corr_max). El cribado ocurre dentro de
       fit: en validación cruzada se repite con cada partición, sin fuga.
    """

    def __init__(self, candidatas=(), fijas=("log_area_ha",), coordenadas=("x_km", "y_km"),
                 n_vecinos=25, k_sel=2, alpha=40.0, corr_max=0.9, semilla=0):
        self.candidatas = candidatas
        self.fijas = fijas
        self.coordenadas = coordenadas
        self.n_vecinos = n_vecinos
        self.k_sel = k_sel
        self.alpha = alpha
        self.corr_max = corr_max
        self.semilla = semilla

    def _vecinos(self, n):
        return KNeighborsRegressor(n_neighbors=max(1, min(self.n_vecinos, n)), weights="distance")

    def fit(self, X, y):
        coords = list(self.coordenadas)
        y = np.asarray(y, dtype=float)
        oof = np.empty(len(y))
        for a, b in KFold(min(5, len(y)), shuffle=True, random_state=self.semilla).split(X):
            oof[b] = self._vecinos(len(a)).fit(X.iloc[a][coords], y[a]).predict(X.iloc[b][coords])
        residuo = y - oof
        self.vecinos_ = self._vecinos(len(y)).fit(X[coords], y)
        fijas = [c for c in self.fijas if c in X.columns]
        candidatas = [c for c in self.candidatas if c not in fijas]
        elegidas, self.correlaciones_ = [], {}
        if self.k_sel and candidatas:
            M = X[candidatas].to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                M = np.where(np.isnan(M), np.nanmedian(M, axis=0), M)
            M = np.nan_to_num(M - M.mean(axis=0))
            r = residuo - residuo.mean()
            denominador = np.sqrt((M ** 2).sum(axis=0) * (r ** 2).sum())
            corr = np.divide(M.T @ r, denominador, out=np.zeros(len(candidatas)), where=denominador > 0)
            for j in np.argsort(-np.abs(corr), kind="stable"):
                if len(elegidas) >= self.k_sel or corr[j] == 0:
                    break
                duplicada = any(abs(np.corrcoef(M[:, j], M[:, q])[0, 1]) > self.corr_max for q in elegidas)
                if not duplicada:
                    elegidas.append(j)
            self.correlaciones_ = {candidatas[j]: float(corr[j]) for j in elegidas}
        self.seleccionadas_ = fijas + [candidatas[j] for j in elegidas]
        self.residual_ = None
        if self.seleccionadas_:
            self.residual_ = Pipeline([
                ("imputacion", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("escala", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))]).fit(X[self.seleccionadas_], residuo)
        return self

    def predict(self, X):
        pred = self.vecinos_.predict(X[list(self.coordenadas)])
        if self.residual_ is not None:
            pred = pred + self.residual_.predict(X[self.seleccionadas_])
        return pred


def crear_pipeline(config, grupos, seed=42, threads=2):
    """Todo paso aprendido vive aquí y se reajusta dentro de cada fold."""
    columnas = grupos[config["grupo"]]
    familia, params = config["familia"], dict(config.get("params", {}))
    if familia == "GeoResidual":
        return GeoResidualRegressor(candidatas=tuple(columnas), semilla=seed, **params)
    modelos = {
        "Media": lambda: DummyRegressor(strategy="mean"),
        "Ridge": lambda: Ridge(**params),
        "ElasticNet": lambda: ElasticNet(max_iter=15000, tol=1e-4, random_state=seed, **params),
        "SVR": lambda: SVR(kernel="rbf", **params),
        "KNeighbors": lambda: KNeighborsRegressor(weights="distance", **params),
        "GaussianProcess": lambda: GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (0.01, 100)) * Matern(length_scale=1.0, length_scale_bounds=(0.05, 100), nu=1.5)
                   + WhiteKernel(noise_level=0.25, noise_level_bounds=(0.01, 2.0)),
            normalize_y=True, n_restarts_optimizer=1, random_state=seed, **params),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=threads, random_state=seed, **params),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, n_jobs=threads, random_state=seed, **params),
        "CatBoost": lambda: CatBoostRegressor(
            loss_function="RMSE", verbose=False, allow_writing_files=False,
            random_seed=seed, thread_count=threads, **params),
    }
    if familia not in modelos:
        raise ValueError(f"Familia desconocida: {familia}")
    steps = [
        ("columnas", ColumnTransformer([("numericas", "passthrough", columnas)], remainder="drop", verbose_feature_names_out=False)),
        ("imputacion", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("constantes", VarianceThreshold()),
    ]
    k = config.get("k", "all")
    if k != "all":
        k = min(int(k), len(columnas))
    steps.append(("seleccion", SelectKBest(f_regression, k=k)))
    if familia in ("Ridge", "ElasticNet", "SVR", "GaussianProcess", "KNeighbors"):
        steps.append(("escala", StandardScaler()))
    steps.append(("modelo", modelos[familia]()))
    return Pipeline(steps)


def generar_candidatos(modo="completo", seed=42):
    """Espacio acotado y fijado antes de observar la prueba."""
    out = []
    def add(familia, grupo, k="all", **params):
        out.append(dict(nombre=f"{len(out):03d}_{familia}_{grupo}", familia=familia, grupo=grupo, k=k, params=params))
    add("Media", "contexto")
    for grupo in ("contexto", "integrado", "amplio", "sin_geografia", "satelite"):
        for alpha in (10.0, 100.0, 1000.0):
            add("Ridge", grupo, alpha=alpha)
    for alpha, ratio in ((0.01, 0.1), (0.05, 0.5), (0.1, 0.9), (0.2, 0.5)):
        add("ElasticNet", "integrado", 32, alpha=alpha, l1_ratio=ratio)
    for grupo in ("contexto", "integrado", "sin_geografia"):
        for c, gamma in ((1.0, "scale"), (10.0, "scale"), (3.0, 0.01)):
            add("SVR", grupo, 32 if grupo != "contexto" else "all", C=c, gamma=gamma, epsilon=0.1)
    for grupo in ("geografia", "compacto"):
        for alpha in (10.0, 100.0):
            add("Ridge", grupo, alpha=alpha)
        for c in (1.0, 10.0):
            add("SVR", grupo, C=c, gamma="scale", epsilon=0.1)
        for vecinos in (5, 12, 25):
            add("KNeighbors", grupo, n_neighbors=vecinos)
        add("GaussianProcess", grupo, alpha=1e-5)
    rng = np.random.default_rng(seed + 19)
    n_arboles = 8 if modo == "completo" else 3
    for fam in ("ExtraTrees", "RandomForest"):
        for i in range(n_arboles):
            grupo = ("contexto", "integrado", "amplio", "sin_geografia")[i % 4]
            add(fam, grupo, max_depth=int(rng.choice([3, 5, 8])), min_samples_leaf=int(rng.choice([2, 4, 7])), max_features=float(rng.choice([0.4, 0.7, 1.0])))
    n_cat = 18 if modo == "completo" else 4
    for i in range(n_cat):
        grupo = ("contexto", "integrado", "amplio", "sin_geografia", "satelite")[i % 5]
        add("CatBoost", grupo, "all" if grupo == "contexto" else int(rng.choice([24, 48, 80])),
            iterations=int(rng.choice([350, 600, 850])), depth=int(rng.choice([2, 3, 4, 5])),
            learning_rate=float(rng.choice([0.025, 0.05, 0.08])),
            l2_leaf_reg=float(rng.choice([5.0, 15.0, 50.0])),
            random_strength=float(rng.choice([0.5, 1.0, 3.0])),
            bootstrap_type="Bayesian", bagging_temperature=float(rng.choice([0.5, 1.0, 3.0])),
            boosting_type="Ordered")
    # Geográfico-residual: k_sel=0 equivale a vecindad + superficie, sin satélite.
    for vecinos in (20, 30):
        for k_sel in (0, 1, 2, 3):
            for alpha in ((40.0, 80.0) if k_sel else (40.0,)):
                add("GeoResidual", "fenologia", n_vecinos=vecinos, k_sel=k_sel, alpha=alpha)
    # Modelos generales con las nuevas métricas fenológicas e historial.
    for alpha in (100.0, 1000.0):
        add("Ridge", "integrado_fenologia", alpha=alpha)
    add("ExtraTrees", "integrado_fenologia", max_depth=8, min_samples_leaf=4, max_features=0.4)
    add("ExtraTrees", "integrado_fenologia", max_depth=5, min_samples_leaf=7, max_features=0.7)
    add("CatBoost", "integrado_fenologia", 48, iterations=600, depth=3, learning_rate=0.05,
        l2_leaf_reg=15.0, random_strength=1.0, bootstrap_type="Bayesian",
        bagging_temperature=1.0, boosting_type="Ordered")
    if modo == "rapido":
        # Conserva representación de todas las familias, sin alterar la prueba.
        out = [c for c in out if c["familia"] not in ("Ridge", "SVR") or int(c["nombre"][:3]) % 3 == 1]
    return out
