"""InfoSelector + variantes (Hito 4): orquesta las etapas 0-3.

Flujo de ``fit(df)`` (docs/DESIGN.md §2):
- **Etapa 0**: esquema + preprocesado (``prepare``) → ``df_prep``.
- **Etapa 1**: screening univariante (``screen``) → candidatas C.
- **Etapa 2**: caché de tablas conjuntas (``build_joint_tables`` +
  ``TableCache``) sobre un subsample ≤ ``subsample`` filas.
- **Etapa 3**: selección greedy detrás del oráculo de información
  (``greedy_select`` + ``HistogramOracle``/``KsgOracle``, solo driver)
  → ``SelectorModel``.

Variantes: ``JMIMSelector`` (default), ``CMIMSelector``, ``MRMRSelector``,
``MIMSelector``.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from pyspark.sql import DataFrame, SparkSession

from .config import SelectorConfig
from .criteria import CRITERIA
from .info.ksg import ksg_mi
from .model import SelectorModel
from .oracles import HistogramOracle, KsgOracle
from .preprocess import prepare
from .screen import screen
from .selection import greedy_select
from .tables import (
    TableCache,
    build_joint_tables,
    cache_cell_budget,
    dense_from_joints,
)

__all__ = [
    "InfoSelector",
    "JMIMSelector",
    "CMIMSelector",
    "MRMRSelector",
    "MIMSelector",
]


class InfoSelector:
    """Selector informacional greedy (base).

    Args:
        criterion: ``"jmim"`` (default) | ``"cmim"`` | ``"mrmr"`` | ``"mim"``.
        **config_kwargs: campos de ``SelectorConfig`` (ver su docstring).

    Uso:
        sel = InfoSelector(target="y", criterion="jmim", max_features=50)
        model = sel.fit(df)
    """

    def __init__(self, criterion: str = "jmim", **config_kwargs):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion debe ser uno de {CRITERIA}")
        self.criterion = criterion
        self.config = SelectorConfig(**config_kwargs)

    # ------------------------------------------------------------------
    # fit: orquesta las etapas 0-3.
    # ------------------------------------------------------------------
    def fit(self, df: DataFrame) -> SelectorModel:
        config = self.config
        target_col = config.target
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # Modo KSG (opcional): subsample al driver + MI/CMI por kNN (sin binning).
        if config.estimator == "ksg":
            model = self._fit_ksg(df, config)
            model.timings_ = {"total": time.perf_counter() - t0}
            return model

        # Etapa 0: esquema + preprocesado.
        t = time.perf_counter()
        prepared = prepare(df, config)
        timings["etapa0"] = time.perf_counter() - t
        df_prep = prepared.df_prep
        schema = prepared.schema
        n_rows = prepared.n_rows

        # Etapa 1: screening → candidatas.
        t = time.perf_counter()
        screen_result = screen(df_prep, schema, config)
        timings["etapa1"] = time.perf_counter() - t
        candidates = list(screen_result.candidates)
        K = len(candidates)
        if K == 0:
            timings["total"] = time.perf_counter() - t0
            return SelectorModel(
                selected_features=[],
                scores_=[],
                ranking_=[],
                criterion=self.criterion,
                n_rows=n_rows,
                target=target_col,
                timings_=timings,
            )

        # Columnas y códigos de las candidatas (orden por MI descendente).
        feature_names = schema.feature_names()
        candidate_cols = [feature_names[i] for i in candidates]
        candidate_n_codes = [schema.n_codes()[i] for i in candidates]
        n_y = schema.target.n_codes

        # Presupuesto de celdas: reduce K (descarta candidatas de menor MI)
        # si se excede ``max_cache_cells``.
        while K > 1 and cache_cell_budget(candidate_n_codes[:K], n_y) > config.max_cache_cells:
            K -= 1
        candidates = candidates[:K]
        candidate_cols = candidate_cols[:K]
        candidate_n_codes = candidate_n_codes[:K]

        # Etapa 2: subsample + tablas conjuntas → TableCache.
        t = time.perf_counter()
        sub = self._subsample(df_prep, config)
        rows_df = build_joint_tables(sub, candidate_cols, target_col, candidate_n_codes, n_y)
        triples = dense_from_joints(rows_df, candidate_n_codes, n_y)
        # Univariantes: tabla de screening de cada candidata (índice de candidata).
        univariate = {i: screen_result.tables[candidates[i]] for i in range(K)}
        cache = TableCache(
            n_codes=candidate_n_codes,
            n_y=n_y,
            univariate=univariate,
            triples=triples,
        )
        timings["etapa2"] = time.perf_counter() - t

        # Etapa 3: oráculo de información + selección greedy (driver).
        t = time.perf_counter()
        oracle = HistogramOracle(
            cache, sub, candidate_cols, target_col, candidate_n_codes, n_y
        )
        result = greedy_select(
            oracle,
            self.criterion,
            config.cmim_approx,
            config.min_score,
            config.max_features,
            config.cmim_m,
        )
        mi = oracle.mi_all()
        # Ranking: candidatas (K) por MI univariante.
        ranking = sorted(
            [(feature_names[candidates[i]], float(mi[i])) for i in range(K)],
            key=lambda t: -t[1],
        )
        model = SelectorModel(
            selected_features=[feature_names[candidates[i]] for i in result.selected],
            scores_=result.scores,
            ranking_=ranking,
            criterion=self.criterion,
            n_rows=n_rows,
            target=target_col,
        )
        timings["etapa3"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0
        model.timings_ = timings
        return model

    # ------------------------------------------------------------------
    # Subsample (lo comparten la etapa 2 y el pase CMIM).
    # ------------------------------------------------------------------
    def _subsample(self, df: DataFrame, config: SelectorConfig) -> DataFrame:
        """Subsample sin reemplazo a ≤ ``subsample`` filas (semilla fija)."""
        n_total = int(df.count())
        if n_total <= config.subsample:
            return df
        fraction = config.subsample / n_total
        return df.sample(withReplacement=False, fraction=fraction, seed=config.seed)

    # ------------------------------------------------------------------
    # Ruta KSG: sin binning, subsample ≤ ``ksg_subsample`` al driver.
    # ------------------------------------------------------------------
    def _fit_ksg(self, df: DataFrame, config: SelectorConfig) -> SelectorModel:
        """Ruta KSG: sin binning, subsample ≤ ``ksg_subsample`` al driver.

        Etapa 1: MI por KSG (driver, población completa). Etapa 3: greedy
        detrás de ``KsgOracle`` (posiciones de candidata).
        """
        target_col = config.target
        feature_cols = [c for c in df.columns if c != target_col]
        n_total = int(df.count())
        if n_total <= config.ksg_subsample:
            df_sub = df
        else:
            fraction = config.ksg_subsample / n_total
            df_sub = df.sample(withReplacement=False, fraction=fraction, seed=config.seed)

        # Al driver como pandas → numpy (variables continuas).
        pdf = df_sub.toPandas()
        X = pdf[feature_cols].to_numpy(dtype=float)  # n_sub × N
        y = pdf[target_col].to_numpy(dtype=float)    # n_sub
        N = X.shape[1]
        if N == 0:
            return SelectorModel(
                selected_features=[],
                scores_=[],
                ranking_=[],
                criterion=self.criterion,
                n_rows=n_total,
                target=target_col,
            )

        # Etapa 1: MI por KSG para cada feature (población completa).
        mi_all = np.array([ksg_mi(X[:, i], y, config.ksg_k) for i in range(N)])

        # Screening: top-K por MI (descendente). Sin filtro FDR en modo KSG.
        order = np.argsort(-mi_all, kind="stable")
        K = min(config.screen_top_k, N)
        cand = [int(i) for i in order[:K]]

        # Etapa 3: oráculo + greedy (posiciones de candidata).
        oracle = KsgOracle(X[:, cand], y, config.ksg_k)
        result = greedy_select(
            oracle,
            self.criterion,
            config.cmim_approx,
            config.min_score,
            config.max_features,
            config.cmim_m,
        )

        # Ranking: población completa (N features) por MI.
        ranking = sorted(
            [(feature_cols[i], float(mi_all[i])) for i in range(N)],
            key=lambda t: -t[1],
        )
        selected_features = [feature_cols[cand[i]] for i in result.selected]
        return SelectorModel(
            selected_features=selected_features,
            scores_=result.scores,
            ranking_=ranking,
            criterion=self.criterion,
            n_rows=n_total,
            target=target_col,
        )


class JMIMSelector(InfoSelector):
    """Selector JMIM (default): ``min_{Xi∈S} CMI(X;Y|Xi)``."""

    def __init__(self, **config_kwargs):
        super().__init__(criterion="jmim", **config_kwargs)


class CMIMSelector(InfoSelector):
    """Selector CMIM: ``CMI(X;Y|S_m)`` con ``S_m`` = top-m por MI (m=2)."""

    def __init__(self, **config_kwargs):
        super().__init__(criterion="cmim", **config_kwargs)


class MRMRSelector(InfoSelector):
    """Selector mRMR: ``MI(X;Y) − (1/|S|)·Σ MI(X;Xi)``."""

    def __init__(self, **config_kwargs):
        super().__init__(criterion="mrmr", **config_kwargs)


class MIMSelector(InfoSelector):
    """Selector mIM: ``MI(X;Y) − Σ MI(X;Xi)``."""

    def __init__(self, **config_kwargs):
        super().__init__(criterion="mim", **config_kwargs)
