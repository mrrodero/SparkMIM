"""Configuración del selector (todos los hiperparámetros del plan, §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from pyspark.sql import SparkSession

__all__ = ["SelectorConfig"]

_MISSING_MODES = ("category", "drop")
_SIGNIFICANCE_MODES = ("chi2", "permutation", None)
_ESTIMATORS = ("histogram", "ksg")
_CMIM_APPROX = ("max_min", None)


@dataclass
class SelectorConfig:
    """Hiperparámetros de ``InfoSelector`` y sus variantes (JMIM/CMIM/...).

    Campos (valores por defecto del plan):
    - ``target``: nombre de la columna objetivo.
    - ``max_features``: nº de features a seleccionar.
    - ``screen_top_k``: candidatas conservadas tras el screening (etapa 1).
    - ``bins``: bins para continuas (int) o ``"auto"`` = clamp(round(log2 n), 4, 20).
    - ``bins_target``: bins cuantiles del target en modo histograma (regresión).
    - ``max_categories``: tope de cardinalidad categóricas (top-C + "other").
    - ``missing``: ``"category"`` (código dedicado) | ``"drop"``.
    - ``significance``: ``"chi2"`` | ``"permutation"`` | None.
    - ``fdr_q``: nivel FDR (Benjamini-Hochberg).
    - ``n_permutations``: permutaciones B del permutation test.
    - ``permutation_rows``: filas del subsample del permutation test.
    - ``min_score``: umbral de parada del greedy (nats).
    - ``subsample``: filas para las tablas conjuntas (etapa 2).
    - ``estimator``: ``"histogram"`` | ``"ksg"``.
    - ``ksg_k``: k del estimador KSG.
    - ``ksg_subsample``: filas del subsample KSG al driver.
    - ``seed``: semilla (subsample, permutation test).
    - ``max_cache_cells``: presupuesto de celdas de la caché (auto-reducción).
    - ``cmim_m``: tamaño del conjunto condicionante de CMIM (top-m por MI).
    - ``cmim_approx``: ``"max_min"`` usa JMIM como aproximación ultrarrápida.
    - ``spark``: sesión Spark (si None, ``getOrCreate()``).
    - ``numeric_features`` / ``categorical_features``: override manual del esquema.
    """

    target: str
    max_features: int = 50
    screen_top_k: int = 100
    bins: Union[int, str] = 10
    bins_target: int = 10
    max_categories: int = 50
    missing: str = "category"
    significance: Optional[str] = "chi2"
    fdr_q: float = 0.05
    n_permutations: int = 100
    permutation_rows: int = 50_000
    min_score: float = 1e-4
    subsample: int = 1_000_000
    estimator: str = "histogram"
    ksg_k: int = 10
    ksg_subsample: int = 250_000
    seed: int = 42
    max_cache_cells: int = 250_000_000
    cmim_m: int = 2
    cmim_approx: Optional[str] = None
    spark: Optional[SparkSession] = None
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.max_features < 1:
            raise ValueError("max_features debe ser >= 1")
        if self.screen_top_k < 1:
            raise ValueError("screen_top_k debe ser >= 1")
        if self.screen_top_k < self.max_features:
            raise ValueError(
                f"screen_top_k ({self.screen_top_k}) debe ser >= max_features "
                f"({self.max_features})"
            )
        if self.bins == "auto":
            pass
        elif not isinstance(self.bins, int) or self.bins < 2:
            raise ValueError('bins debe ser int >= 2 o "auto"')
        if self.bins_target < 2:
            raise ValueError("bins_target debe ser >= 2")
        if self.max_categories < 1:
            raise ValueError("max_categories debe ser >= 1")
        if self.missing not in _MISSING_MODES:
            raise ValueError(f"missing debe ser uno de {_MISSING_MODES}")
        if self.significance not in _SIGNIFICANCE_MODES:
            raise ValueError(f"significance debe ser uno de {_SIGNIFICANCE_MODES}")
        if not 0 < self.fdr_q < 1:
            raise ValueError("fdr_q debe estar en (0, 1)")
        if self.n_permutations < 1:
            raise ValueError("n_permutations debe ser >= 1")
        if self.permutation_rows < 100:
            raise ValueError("permutation_rows debe ser >= 100")
        if self.min_score < 0:
            raise ValueError("min_score debe ser >= 0")
        if self.subsample < 1000:
            raise ValueError("subsample debe ser >= 1000")
        if self.estimator not in _ESTIMATORS:
            raise ValueError(f"estimator debe ser uno de {_ESTIMATORS}")
        if self.ksg_k < 1:
            raise ValueError("ksg_k debe ser >= 1")
        if self.ksg_subsample < 1000:
            raise ValueError("ksg_subsample debe ser >= 1000")
        if self.max_cache_cells < 1000:
            raise ValueError("max_cache_cells debe ser >= 1000")
        if self.cmim_m < 1:
            raise ValueError("cmim_m debe ser >= 1")
        if self.cmim_approx not in _CMIM_APPROX:
            raise ValueError(f"cmim_approx debe ser uno de {_CMIM_APPROX}")
        if self.numeric_features and self.categorical_features:
            overlap = set(self.numeric_features) & set(self.categorical_features)
            if overlap:
                raise ValueError(f"features en ambas listas: {sorted(overlap)}")
