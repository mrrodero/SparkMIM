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
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)

from .config import SelectorConfig
from .criteria import CRITERIA
from .info.ksg import ksg_mi
from .model import SelectorModel
from .schema import resolve_task
from .oracles import HistogramOracle, KsgOracle
from .preprocess import prepare
from .screen import rank, screen, select_candidates
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
    "subsample",
]

# Tipos numéricos que el modo KSG acepta como features.
_KSG_NUMERIC_TYPES = (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
)
# Tipos que el modo KSG acepta como target: numéricos (toda Task) o
# categóricos (solo Task de clasificación; se codifican por valor distinto).
_KSG_TARGET_TYPES = _KSG_NUMERIC_TYPES + (
    StringType,
    BooleanType,
    DateType,
    TimestampType,
)


def subsample(df: DataFrame, n: int, seed: int) -> DataFrame:
    """Submuestra sin reemplazo de ≈ ``n`` filas (semilla fija).

    Política compartida de reducción de filas: la etapa 2 del modo
    histograma (``config.subsample``) y la preparación del modo KSG
    (``config.ksg_subsample``). Identidad si el frame ya tiene ≤ ``n``
    filas; determinista con ``seed``.
    """
    n_total = int(df.count())
    if n_total <= n:
        return df
    fraction = n / n_total
    return df.sample(withReplacement=False, fraction=fraction, seed=seed)


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
        # Los timings por etapa (etapa0/1/3 + total) los pone ``_fit_ksg``.
        if config.estimator == "ksg":
            return self._fit_ksg(df, config)

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
        feature_names = schema.feature_names()
        # Ranking: población completa (N features) por MI univariante de
        # screening, descendente. Sin coste adicional: reutiliza la etapa 1.
        ranking = rank(screen_result.mi, feature_names)
        if K == 0:
            timings["total"] = time.perf_counter() - t0
            return SelectorModel(
                selected_features=[],
                scores_=[],
                ranking_=ranking,
                criterion=self.criterion,
                n_rows=n_rows,
                target=target_col,
                task=prepared.task,
                timings_=timings,
            )

        # Columnas y códigos de las candidatas (orden por MI descendente).
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
        sub = subsample(df_prep, config.subsample, config.seed)
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
        model = SelectorModel(
            selected_features=[feature_names[candidates[i]] for i in result.selected],
            scores_=result.scores,
            ranking_=ranking,
            criterion=self.criterion,
            n_rows=n_rows,
            target=target_col,
            task=prepared.task,
        )
        timings["etapa3"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0
        model.timings_ = timings
        return model

    # ------------------------------------------------------------------
    # Ruta KSG: sin binning, subsample ≈ ``ksg_subsample`` al driver.
    # ------------------------------------------------------------------
    def _fit_ksg(self, df: DataFrame, config: SelectorConfig) -> SelectorModel:
        """Ruta KSG: sin binning, subsample ≈ ``ksg_subsample`` al driver.

        Etapa 0: Task + validación de dtypes + nombres de features.
        Etapa 1: subsample al driver + MI por KSG (población completa) +
        corte top-K (``select_candidates``) + ranking (``rank``).
        Etapa 3: greedy detrás de ``KsgOracle`` (posiciones de candidata).

        La Task se resuelve una vez aquí (regla compartida,
        ``schema.resolve_task``) y se lleva en el ``SelectorModel``. Las
        features deben ser numéricas; el target debe ser numérico si la Task
        es continua, y numérico o categórico si es de clasificación (los
        valores distintos se codifican sin pérdida). Errores claros aquí, no
        crashes profundos.
        """
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # Etapa 0: Task + validación de dtypes + nombres de features.
        target_col = config.target
        task = resolve_task(df, target_col, config.task)
        feature_cols = [c for c in df.columns if c != target_col]
        bad_feature_cols = [
            c
            for c in feature_cols
            if not isinstance(df.schema[c].dataType, _KSG_NUMERIC_TYPES)
        ]
        if bad_feature_cols:
            raise ValueError(
                "el modo 'ksg' requiere features numéricas; no numéricas: "
                f"{[(c, df.schema[c].dataType.typeName()) for c in bad_feature_cols]}"
            )
        target_dtype = df.schema[target_col].dataType
        if task == "continuous":
            if not isinstance(target_dtype, _KSG_NUMERIC_TYPES):
                raise ValueError(
                    "el modo 'ksg' con task='continuous' requiere un target "
                    f"numérico; el target '{target_col}' es {target_dtype}"
                )
        else:
            if not isinstance(target_dtype, _KSG_TARGET_TYPES):
                raise ValueError(
                    f"el modo 'ksg' con task={task!r} no soporta el dtype del "
                    f"target '{target_col}': {target_dtype}"
                )
        n_total = int(df.count())
        timings["etapa0"] = time.perf_counter() - t0

        # Etapa 1: subsample al driver + MI por KSG (población completa).
        t = time.perf_counter()
        df_sub = subsample(df, config.ksg_subsample, config.seed)
        pdf = df_sub.toPandas()  # al driver como pandas → numpy.
        X = pdf[feature_cols].to_numpy(dtype=float)  # n_sub × N
        if task == "continuous":
            y = pdf[target_col].to_numpy(dtype=float)  # n_sub
        else:
            # Clasificación: códigos sin pérdida (un entero por valor distinto).
            _, y = np.unique(pdf[target_col].to_numpy(), return_inverse=True)
        N = X.shape[1]
        if N == 0:
            timings["etapa1"] = time.perf_counter() - t
            timings["total"] = time.perf_counter() - t0
            return SelectorModel(
                selected_features=[],
                scores_=[],
                ranking_=[],
                criterion=self.criterion,
                n_rows=n_total,
                target=target_col,
                task=task,
                timings_=timings,
            )

        # MI por KSG para cada feature (población completa).
        mi_all = np.array([ksg_mi(X[:, i], y, config.ksg_k) for i in range(N)])

        # Top-K: la política compartida de la etapa 1. Sin filtro de
        # significancia con el estimador KSG (implementación pendiente).
        cand = select_candidates(mi_all, np.ones(N, dtype=bool), config.screen_top_k)
        # Ranking: población completa (N features) por MI.
        ranking = rank(mi_all, feature_cols)
        timings["etapa1"] = time.perf_counter() - t

        # Etapa 3: oráculo + greedy (posiciones de candidata).
        t = time.perf_counter()
        oracle = KsgOracle(X[:, cand], y, config.ksg_k)
        result = greedy_select(
            oracle,
            self.criterion,
            config.cmim_approx,
            config.min_score,
            config.max_features,
            config.cmim_m,
        )
        timings["etapa3"] = time.perf_counter() - t

        selected_features = [feature_cols[cand[i]] for i in result.selected]
        timings["total"] = time.perf_counter() - t0
        return SelectorModel(
            selected_features=selected_features,
            scores_=result.scores,
            ranking_=ranking,
            criterion=self.criterion,
            n_rows=n_total,
            target=target_col,
            task=task,
            timings_=timings,
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
