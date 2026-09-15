"""InfoSelector + variantes (Hito 4): orquesta las etapas 0-3.

Flujo de ``fit(df)`` (PLAN_IMPLEMENTACION.md §6):
- **Etapa 0**: esquema + preprocesado (``prepare``) → ``df_prep``.
- **Etapa 1**: screening univariante (``screen``) → candidatas C.
- **Etapa 2**: caché de tablas conjuntas (``build_joint_tables`` + ``TableCache``)
  sobre un subsample ≤ ``subsample`` filas.
- **Etapa 3**: selección greedy (solo driver) → ``SelectorModel``.

Variantes: ``JMIMSelector`` (default), ``CMIMSelector``, ``MRMRSelector``,
``MIMSelector``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from pyspark.sql import DataFrame, SparkSession

from .config import SelectorConfig
from .criteria import CRITERIA, cmim_scores, criterion_score
from .info.entropy import mutual_information
from .model import SelectorModel
from .preprocess import prepare
from .screen import screen
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

        # Etapa 0: esquema + preprocesado.
        prepared = prepare(df, config)
        df_prep = prepared.df_prep
        schema = prepared.schema
        n_rows = prepared.n_rows

        # Etapa 1: screening → candidatas.
        screen_result = screen(df_prep, schema, config)
        candidates = list(screen_result.candidates)
        K = len(candidates)
        if K == 0:
            return SelectorModel(
                selected_features=[],
                scores_=[],
                ranking_=[],
                criterion=self.criterion,
                n_rows=n_rows,
                target=target_col,
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

        # Etapa 2: tablas conjuntas (subsample) → TableCache.
        cache = self._build_cache(
            df_prep, candidate_cols, target_col, candidate_n_codes, n_y,
            screen_result, candidates, config,
        )

        # Etapa 3: selección greedy (driver).
        return self._greedy(
            df_prep, feature_names, candidates, candidate_cols, candidate_n_codes,
            n_y, cache, screen_result, config, n_rows,
        )

    # ------------------------------------------------------------------
    # Etapa 2: caché de tablas conjuntas.
    # ------------------------------------------------------------------
    def _subsample(self, df: DataFrame, config: SelectorConfig) -> DataFrame:
        n_total = int(df.count())
        if n_total <= config.subsample:
            return df
        fraction = config.subsample / n_total
        return df.sample(withReplacement=False, fraction=fraction, seed=config.seed)

    def _build_cache(
        self,
        df_prep: DataFrame,
        candidate_cols: Sequence[str],
        target_col: str,
        candidate_n_codes: Sequence[int],
        n_y: int,
        screen_result,
        candidates: Sequence[int],
        config: SelectorConfig,
    ) -> TableCache:
        sub = self._subsample(df_prep, config)
        rows_df = build_joint_tables(sub, candidate_cols, target_col, candidate_n_codes, n_y)
        triples = dense_from_joints(rows_df, candidate_n_codes, n_y)
        # Univariate: tablas de screening de cada candidata (índice de candidata).
        univariate = {i: screen_result.tables[candidates[i]] for i in range(len(candidates))}
        return TableCache(
            n_codes=candidate_n_codes,
            n_y=n_y,
            univariate=univariate,
            triples=triples,
        )

    # ------------------------------------------------------------------
    # Etapa 3: selección greedy.
    # ------------------------------------------------------------------
    def _greedy(
        self,
        df_prep: DataFrame,
        feature_names: Sequence[str],
        candidates: Sequence[int],
        candidate_cols: Sequence[str],
        candidate_n_codes: Sequence[int],
        n_y: int,
        cache: TableCache,
        screen_result,
        config: SelectorConfig,
        n_rows: int,
    ) -> SelectorModel:
        K = len(candidates)
        # MI univariante por candidata (tablas de screening).
        mi_xy = np.array([mutual_information(cache.uni(i)) for i in range(K)])

        selected: List[int] = []
        scores: List[float] = []

        # CMIM exacto (pase por ronda) vs. aproximación "max_min" (= JMIM).
        use_cmim_pass = self.criterion == "cmim" and config.cmim_approx != "max_min"
        crit = "jmim" if use_cmim_pass is False and self.criterion == "cmim" else self.criterion
        if use_cmim_pass:
            crit = None  # no se usa criterion_score en el camino CMIM.

        sub = self._subsample(df_prep, config) if use_cmim_pass else None
        # S_m (CMIM): top-m por MI univariante (fijo).
        m = min(config.cmim_m, K)
        s_m = [int(i) for i in np.argsort(-mi_xy, kind="stable")[:m]] if use_cmim_pass else []

        while len(selected) < config.max_features and len(selected) < K:
            if not selected:
                # Ronda 1: argmax MI univariante.
                best = int(np.argmax(mi_xy))
                best_score = float(mi_xy[best])
            elif use_cmim_pass:
                # CMIM: CMI(X_i; Y | S_m \ {i}) para todas las candidatas.
                cmis = cmim_scores(sub, candidate_cols, config.target, candidate_n_codes, n_y, s_m)
                best = -1
                best_score = -np.inf
                for x in range(K):
                    if x in selected:
                        continue
                    if cmis[x] > best_score:
                        best_score = float(cmis[x])
                        best = x
            else:
                # Criterios rápidos sobre el caché.
                best = -1
                best_score = -np.inf
                for x in range(K):
                    if x in selected:
                        continue
                    score = criterion_score(x, selected, cache, mi_xy, crit)
                    if score > best_score:
                        best_score = score
                        best = x
            if best_score < config.min_score:
                break
            selected.append(best)
            scores.append(float(best_score))

        # Ranking completo de candidatas por MI univariante (descendente).
        ranking = sorted(
            [(feature_names[candidates[i]], float(mi_xy[i])) for i in range(K)],
            key=lambda t: -t[1],
        )
        selected_features = [feature_names[candidates[i]] for i in selected]
        return SelectorModel(
            selected_features=selected_features,
            scores_=scores,
            ranking_=ranking,
            criterion=self.criterion,
            n_rows=n_rows,
            target=config.target,
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
