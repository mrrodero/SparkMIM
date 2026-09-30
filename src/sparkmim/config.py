"""Configuración del selector (todos los hiperparámetros del plan, §5)."""

from __future__ import annotations

from dataclasses import dataclass, fields, MISSING
from typing import List, Optional, Union

from pyspark.sql import SparkSession

from .schema import _TASKS

__all__ = ["SelectorConfig"]

_MISSING_MODES = ("category", "drop")
_SIGNIFICANCE_MODES = ("chi2", "permutation", None)
_ESTIMATORS = ("histogram", "ksg")
_CMIM_APPROX = ("max_min", None)

# Relevancia por modo (fuente única; el modo lo fija ``estimator``).
# Un campo de un grupo "solo X" no aplica al modo opuesto: debe dejar su
# valor por defecto, si no la construcción lanza ValueError.
_KSG_ONLY_FIELDS = frozenset({"ksg_k", "ksg_subsample"})
_HISTOGRAM_ONLY_FIELDS = frozenset(
    {
        "bins",
        "bins_target",
        "max_categories",
        "missing",
        "significance",
        "fdr_q",
        "n_permutations",
        "permutation_rows",
        "subsample",
        "max_cache_cells",
        "numeric_features",
        "categorical_features",
    }
)
_SHARED_FIELDS = frozenset(
    {
        "target",
        "task",
        "max_features",
        "screen_top_k",
        "min_score",
        "seed",
        "cmim_m",
        "cmim_approx",
        "spark",
        "estimator",
    }
)


@dataclass
class SelectorConfig:
    """Hiperparámetros de ``InfoSelector`` y sus variantes (JMIM/CMIM/...).

    Campos (valores por defecto del plan):
    - ``target``: nombre de la columna objetivo.
    - ``task``: Task del Target declarada por el usuario: ``"auto"`` (default),
      ``"classifier_binary"``, ``"classifier_multiclass"`` o ``"continuous"``.
      Se resuelve una vez en la etapa 0 validando contra los datos
      (``schema.resolve_task``).
    - ``max_features``: nº de features a seleccionar.
    - ``screen_top_k``: candidatas conservadas tras el screening (etapa 1).
    - ``bins``: bins para continuas (int) o ``"auto"`` = clamp(round(log2 n), 4, 20).
    - ``bins_target``: bins cuantiles del target con el estimador de histograma (regresión).
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

    Modo (``estimator``) — relevancia por modo declarada en un solo sitio
    (``_HISTOGRAM_ONLY_FIELDS`` / ``_KSG_ONLY_FIELDS`` / ``_SHARED_FIELDS``):
    - ``"histogram"`` (default): ``bins``, ``bins_target``, ``max_categories``,
      ``missing``, ``significance``, ``fdr_q``, ``n_permutations``,
      ``permutation_rows``, ``subsample``, ``max_cache_cells`` y los overrides
      de esquema (``numeric_features`` / ``categorical_features``).
    - ``"ksg"``: ``ksg_k`` y ``ksg_subsample``; las features deben ser
      numéricas y el target numérico si ``task`` es continua (error claro en
      ``fit``).
    - Compartidos: ``target``, ``task``, ``max_features``, ``screen_top_k``,
      ``min_score``, ``seed``, ``cmim_m``, ``cmim_approx``, ``spark``.
    Un campo del otro modo distinto de su valor por defecto lanza ``ValueError``
    en la construcción.
    """

    target: str
    task: str = "auto"
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
        if self.task not in _TASKS:
            raise ValueError(f"task debe ser uno de {_TASKS}")
        self._validate_mode()
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

    def _validate_mode(self) -> None:
        """Los campos del otro modo deben dejar su valor por defecto.

        La relevancia por modo vive en los grupos ``_HISTOGRAM_ONLY_FIELDS`` /
        ``_KSG_ONLY_FIELDS`` (fuente única); aquí solo se comprueba que el
        modo activo no recibe campos ajenos distintos de su defecto.
        """
        other = (
            _HISTOGRAM_ONLY_FIELDS
            if self.estimator == "ksg"
            else _KSG_ONLY_FIELDS
        )
        defaults = {
            f.name: f.default for f in fields(self) if f.default is not MISSING
        }
        bad = sorted(name for name in other if getattr(self, name) != defaults[name])
        if bad:
            raise ValueError(
                f"campos {', '.join(bad)} no aplican al estimador "
                f"'{self.estimator}' (déjalos en su valor por defecto)"
            )
