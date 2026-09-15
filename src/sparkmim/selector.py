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

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from pyspark.sql import DataFrame, SparkSession

from .config import SelectorConfig
from .criteria import CRITERIA, cmim_scores, criterion_score
from .info.entropy import mutual_information
from .info.ksg import ksg_cmi, ksg_mi
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

        # Etapa 2: tablas conjuntas (subsample) → TableCache.
        t = time.perf_counter()
        cache = self._build_cache(
            df_prep, candidate_cols, target_col, candidate_n_codes, n_y,
            screen_result, candidates, config,
        )
        timings["etapa2"] = time.perf_counter() - t

        # Etapa 3: selección greedy (driver).
        t = time.perf_counter()
        model = self._greedy(
            df_prep, feature_names, candidates, candidate_cols, candidate_n_codes,
            n_y, cache, screen_result, config, n_rows,
        )
        timings["etapa3"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0
        model.timings_ = timings
        return model

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


    # ------------------------------------------------------------------
    # Modo KSG (opcional): subsample al driver + MI/CMI por kNN.
    # ------------------------------------------------------------------
    def _fit_ksg(self, df: DataFrame, config: SelectorConfig) -> SelectorModel:
        """Ruta KSG: sin binning, subsample ≤ ``ksg_subsample`` al driver.

        Etapa 1: MI por KSG (driver). Etapa 3: greedy con CMI por KSG
        (identidad ``MI(XZ;Y) − MI(Z;Y)``). Única ruta con kNN global en driver.
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
                selected_features=[], scores_=[], ranking_=[],
                criterion=self.criterion, n_rows=n_total, target=target_col,
            )

        # Etapa 1: MI por KSG para cada feature.
        mi_xy = np.array([ksg_mi(X[:, i], y, config.ksg_k) for i in range(N)])

        # Screening: top-K por MI (descendente). Sin filtro FDR en modo KSG.
        order = np.argsort(-mi_xy, kind="stable")
        K = min(config.screen_top_k, N)
        candidates = [int(i) for i in order[:K]]

        # Greedy con CMI por KSG.
        use_cmim_pass = self.criterion == "cmim" and config.cmim_approx != "max_min"
        crit = None if use_cmim_pass else ("jmim" if self.criterion == "cmim" else self.criterion)
        m = min(config.cmim_m, K)
        s_m = [candidates[int(i)] for i in np.argsort(-mi_xy, kind="stable")[:m]] if use_cmim_pass else []

        mi_pair_cache: dict = {}
        cmi_cache: dict = {}

        def _mi_pair(i: int, j: int) -> float:
            key = (min(i, j), max(i, j))
            if key not in mi_pair_cache:
                mi_pair_cache[key] = ksg_mi(X[:, key[0]], X[:, key[1]], config.ksg_k)
            return mi_pair_cache[key]

        def _cmi(x: int, z_cols: Sequence[int]) -> float:
            key = (x, tuple(sorted(z_cols)))
            if key not in cmi_cache:
                if not z_cols:
                    cmi_cache[key] = float(mi_xy[x])
                elif len(z_cols) == 1:
                    cmi_cache[key] = ksg_cmi(X[:, x], y, X[:, z_cols[0]], config.ksg_k)
                else:
                    cmi_cache[key] = ksg_cmi(X[:, x], y, X[:, list(z_cols)], config.ksg_k)
            return cmi_cache[key]

        selected: List[int] = []
        scores: List[float] = []
        while len(selected) < config.max_features and len(selected) < K:
            if not selected:
                best = candidates[int(np.argmax(mi_xy))]
                best_score = float(mi_xy[best])
            elif use_cmim_pass:
                best = -1
                best_score = -np.inf
                for x in candidates:
                    if x in selected:
                        continue
                    z_cols = [s for s in s_m if s != x]
                    score = _cmi(x, z_cols)
                    if score > best_score:
                        best_score = float(score)
                        best = x
            else:
                best = -1
                best_score = -np.inf
                for x in candidates:
                    if x in selected:
                        continue
                    score = self._ksg_criterion_score(x, selected, mi_xy, _mi_pair, _cmi, crit)
                    if score > best_score:
                        best_score = float(score)
                        best = x
            if best_score < config.min_score:
                break
            selected.append(best)
            scores.append(float(best_score))

        ranking = sorted(
            [(feature_cols[i], float(mi_xy[i])) for i in range(N)],
            key=lambda t: -t[1],
        )
        selected_features = [feature_cols[i] for i in selected]
        return SelectorModel(
            selected_features=selected_features,
            scores_=scores,
            ranking_=ranking,
            criterion=self.criterion,
            n_rows=n_total,
            target=target_col,
        )

    def _ksg_criterion_score(
        self,
        x: int,
        S: Sequence[int],
        mi_xy: np.ndarray,
        _mi_pair,
        _cmi,
        crit: str,
    ) -> float:
        """Criterio greedy en modo KSG (MI/CMI por kNN, sin caché de tablas)."""
        if not S:
            return float(mi_xy[x])
        if crit == "mrmr":
            redundancy = sum(_mi_pair(x, s) for s in S) / len(S)
            return float(mi_xy[x]) - redundancy
        if crit == "mim":
            redundancy = sum(_mi_pair(x, s) for s in S)
            return float(mi_xy[x]) - redundancy
        if crit == "jmi":
            return float(sum(_cmi(x, [s]) for s in S))
        if crit == "jmim":
            return float(min(_cmi(x, [s]) for s in S))
        raise ValueError(f"criterio {crit!r} no soportado en modo KSG")


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
