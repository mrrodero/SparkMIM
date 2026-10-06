"""Significancia estadística (Hito 3, etapa 1).

Costura de la etapa 1 (docs/DESIGN.md §2, §4): ``screen()`` compone
tablas → MI → significancia → top-K y solo ve la interfaz
``SignificanceTest``. Todo el conocimiento de p-valores vive en este
módulo:

- ``SignificanceTest``: interfaz — "dadas las tablas de screening y la MI
  por feature (y, para la permutación, el df preparado), devuelve los
  p-valores y la máscara de significancia por feature".
- Tres adaptadores: ``Chi2Test`` (``G² = 2·n·MI ~ χ²``, O(1) por feature),
  ``PermutationTest`` (test de permutación distribuido: UN ``mapInPandas``
  sobre un subsample, solo para las top-K por MI) y ``NoTest`` (sin filtro).
- ``make_significance_test``: fábrica de adaptadores desde ``SelectorConfig``.
- ``bh_fdr``: control de tasa de descubrimiento falsa de Benjamini-Hochberg
  (``q = 0.05`` por defecto); cada adaptador lo aplica sobre sus p-valores.

Helpers puros (también la superficie de test): ``chi2_pvalue`` y
``permutation_pvalue``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from pyspark.sql.types import BinaryType, IntegerType, StructField, StructType

from .config import SelectorConfig
from .info.entropy import mutual_information

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

__all__ = [
    "SignificanceInput",
    "SignificanceResult",
    "SignificanceTest",
    "Chi2Test",
    "PermutationTest",
    "NoTest",
    "make_significance_test",
    "chi2_pvalue",
    "permutation_pvalue",
    "bh_fdr",
]


# Esquema de la salida del mapInPandas de permutación: (fid, b, tabla).
# b = -1 => tabla observada; b = 0..B-1 => tabla de la permutación b.
_PermSchema = StructType(
    [
        StructField("fid", IntegerType(), False),
        StructField("b", IntegerType(), False),
        StructField("table", BinaryType(), False),
    ]
)


@dataclass
class SignificanceInput:
    """Datos de la etapa 1 para computar la significancia.

    - ``tables``: tablas de screening densas ``(n_x, n_y)`` por feature (fid).
    - ``mi``: ``MI(X_i; Y)`` por feature (nats), longitud N.
    - ``df_prep``: DataFrame preprocesado (solo lo usa ``PermutationTest``).
    - ``feature_cols`` / ``target_col`` / ``n_x`` / ``n_y``: nombres y
      cardinalidades (solo los usa ``PermutationTest``).
    """

    tables: Dict[int, np.ndarray]
    mi: np.ndarray
    df_prep: "DataFrame | None" = None
    feature_cols: Sequence[str] = ()
    target_col: str = ""
    n_x: Sequence[int] = ()
    n_y: int = 0


@dataclass
class SignificanceResult:
    """Resultado de la significancia por feature (longitud N).

    - ``pvalues``: p-valor por feature (1.0 si no se computó: ``NoTest`` o
      feature excluida por el pre-filtro de ``PermutationTest``).
    - ``significant``: máscara tras control FDR (todas True en ``NoTest``).
    """

    pvalues: np.ndarray
    significant: np.ndarray


class SignificanceTest(Protocol):
    """Interfaz de la costura de significancia (etapa 1).

    Dadas las tablas de screening y la MI por feature (y, para la
    permutación, el df preparado), devuelve los p-valores y la máscara de
    significancia por feature. Cada adaptador decide su coste: ``Chi2Test``
    es O(1) por feature; ``PermutationTest`` ejecuta UN ``mapInPandas``
    sobre un subsample de las top-K por MI; ``NoTest`` no computa nada.
    """

    def test(self, data: SignificanceInput) -> SignificanceResult:
        ...


class Chi2Test:
    """Adaptador χ²: ``G² = 2·n·MI ~ χ²_{(n_x-1)(n_y-1)}``, O(1) por feature.

    Aplica el control FDR (``q``) sobre los p-valores.
    """

    def __init__(self, q: float = 0.05) -> None:
        self._q = q

    def test(self, data: SignificanceInput) -> SignificanceResult:
        pvalues = np.array(
            [chi2_pvalue(data.tables[i], data.mi[i]) for i in range(len(data.mi))]
        )
        return SignificanceResult(
            pvalues=pvalues, significant=bh_fdr(pvalues, self._q)
        )


class PermutationTest:
    """Adaptador de permutación: test distribuido (UN ``mapInPandas``).

    El test es caro, así que se aplica solo a las top-``k`` por MI
    (pre-filtro); el resto queda con p = 1 y no significativa. El control
    FDR se aplica a los p-valores de las features testeadas.
    """

    def __init__(
        self,
        q: float,
        n_permutations: int,
        permutation_rows: int,
        screen_top_k: int,
        seed: int,
    ) -> None:
        self._q = q
        self._n_b = n_permutations
        self._rows = permutation_rows
        self._k = screen_top_k
        self._seed = seed

    def test(self, data: SignificanceInput) -> SignificanceResult:
        n = len(data.mi)
        k = min(self._k, n)
        pvalues = np.ones(n)
        significant = np.zeros(n, dtype=bool)
        if k > 0:
            top_idx = np.argsort(-data.mi, kind="stable")[:k]
            cand_cols = [data.feature_cols[i] for i in top_idx]
            cand_nx = [data.n_x[i] for i in top_idx]
            pvals_k = _permutation_pvalues(
                data.df_prep,
                cand_cols,
                data.target_col,
                cand_nx,
                data.n_y,
                self._n_b,
                self._rows,
                self._seed,
            )
            sig_k = bh_fdr(pvals_k, self._q)
            for pos, i in enumerate(top_idx):
                pvalues[i] = pvals_k[pos]
                significant[i] = sig_k[pos]
        return SignificanceResult(pvalues=pvalues, significant=significant)


class NoTest:
    """Adaptador sin test: p = 1 y todas las features "significativas"."""

    def test(self, data: SignificanceInput) -> SignificanceResult:
        n = len(data.mi)
        return SignificanceResult(
            pvalues=np.ones(n), significant=np.ones(n, dtype=bool)
        )


def make_significance_test(config: SelectorConfig) -> SignificanceTest:
    """Fábrica de adaptadores desde ``SelectorConfig``.

    ``config.significance``: ``"chi2"`` → ``Chi2Test``; ``"permutation"`` →
    ``PermutationTest``; ``None`` → ``NoTest``.
    """
    if config.significance is None:
        return NoTest()
    if config.significance == "chi2":
        return Chi2Test(q=config.fdr_q)
    if config.significance == "permutation":
        return PermutationTest(
            q=config.fdr_q,
            n_permutations=config.n_permutations,
            permutation_rows=config.permutation_rows,
            screen_top_k=config.screen_top_k,
            seed=config.seed,
        )
    raise ValueError(
        f"significance debe ser 'chi2', 'permutation' o None, no {config.significance!r}"
    )


def _permutation_pvalues(
    df_prep: DataFrame,
    candidate_cols: List[str],
    target_col: str,
    n_x: List[int],
    n_y: int,
    n_b: int,
    n_rows_sub: int,
    seed: int,
) -> np.ndarray:
    """P-valores de permutación para un conjunto de features (distribuido).

    Un único ``mapInPandas`` sobre un subsample de ``df_prep`` emite, por
    feature y por permutación, la tabla de contingencia ``(n_x, n_y)``
    serializada; el driver acumula, computa ``MI`` de cada tabla y el
    p-valor de permutación contra la distribución nula.

    Args:
        df_prep: DataFrame preprocesado (códigos enteros).
        candidate_cols: nombres de las features candidatas.
        target_col: nombre del target.
        n_x: nº de códigos por feature candidata.
        n_y: nº de códigos del target.
        n_b: nº de permutaciones B (distribución nula).
        n_rows_sub: tamaño máximo del subsample.
        seed: semilla base (cada permutación b usa ``seed + b``).

    Returns:
        Array de p-valores (longitud ``len(candidate_cols)``).
    """
    k = len(candidate_cols)
    cols = list(candidate_cols) + [target_col]
    n_rows = df_prep.count()
    fraction = min(1.0, n_rows_sub / max(n_rows, 1))
    df_sub = (
        df_prep.select(*cols)
        .sample(withReplacement=False, fraction=fraction, seed=seed)
        .cache()
    )

    def _emit(partition):
        # ``mapInPandas`` pasa un iterador de pandas.DataFrames (uno por
        # batch de Arrow); se procesa cada batch y se emiten sus filas.
        for pdf in partition:
            ys = pdf[target_col].to_numpy().astype(np.int64)
            rows: List[tuple] = []
            for i, col in enumerate(candidate_cols):
                x = pdf[col].to_numpy().astype(np.int64)
                ni = n_x[i]
                # b = -1: observado (sin permutar).
                c = np.bincount(x * n_y + ys, minlength=ni * n_y).reshape(ni, n_y)
                rows.append((i, -1, c.tobytes()))
                for b in range(n_b):
                    rng = np.random.default_rng(seed + b)
                    yp = rng.permutation(ys)
                    c = np.bincount(x * n_y + yp, minlength=ni * n_y).reshape(ni, n_y)
                    rows.append((i, b, c.tobytes()))
            out = pd.DataFrame(rows, columns=["fid", "b", "table"])
            yield out.astype({"fid": "int64", "b": "int64", "table": "object"})

    rows_df = df_sub.mapInPandas(_emit, schema=_PermSchema).collect()
    df_sub.unpersist()

    # Acumular las tablas por (i, b) en el driver.
    acc_obs = [np.zeros((n_x[i], n_y), dtype=np.int64) for i in range(k)]
    acc_null = [
        [np.zeros((n_x[i], n_y), dtype=np.int64) for _ in range(n_b)] for i in range(k)
    ]
    for r in rows_df:
        i, b = r.fid, r.b
        arr = np.frombuffer(r.table, dtype=np.int64).reshape(n_x[i], n_y)
        if b == -1:
            acc_obs[i] += arr
        else:
            acc_null[i][b] += arr

    mi_obs = np.array([mutual_information(acc_obs[i]) for i in range(k)])
    pvals = np.empty(k)
    for i in range(k):
        mi_null = np.array([mutual_information(acc_null[i][b]) for b in range(n_b)])
        pvals[i] = permutation_pvalue(mi_obs[i], mi_null)
    return pvals


def chi2_pvalue(table: np.ndarray, mi: float) -> float:
    """P-valor del test de χ² para una tabla de contingencia 2D.

    Usa la aproximación de Guttman: ``G² = 2·n·MI`` (nats) ~
    ``χ²_{(n_x-1)(n_y-1)}``.

    Args:
        table: tabla de contingencia ``(n_x, n_y)`` de enteros.
        mi: ``MI(X; Y)`` en nats de esa tabla.

    Returns:
        P-valor (1.0 si la tabla está vacía o el df es degenerado).
    """
    n = int(table.sum())
    if n <= 0:
        return 1.0
    n_x, n_y = table.shape
    df = (n_x - 1) * (n_y - 1)
    if df <= 0:
        return 1.0
    g2 = 2.0 * n * mi
    return float(stats.chi2.sf(g2, df))


def permutation_pvalue(mi_observed: float, mi_null: np.ndarray) -> float:
    """P-valor de permutación (proporción de nulas ≥ observada).

    Args:
        mi_observed: MI observada (nats).
        mi_null: MI bajo la distribución nula (B permutaciones).

    Returns:
        P-valor ``= (1 + #null >= observed) / (B + 1)`` (1.0 si B = 0).
    """
    if mi_null.size == 0:
        return 1.0
    return float((1.0 + np.sum(mi_null >= mi_observed)) / (mi_null.size + 1.0))


def bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Control de FDR de Benjamini-Hochberg.

    Rechaza las hipótesis con p-valor más pequeño hasta el mayor ``k`` tal
    que ``p_(k) <= (k/m)·q``.

    Args:
        pvalues: array de p-valores (longitud m).
        q: tasa de descubrimiento falsa deseada.

    Returns:
        Máscara booleana de longitud m (True = significativo).
    """
    m = pvalues.size
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvalues, kind="stable")
    ranks = np.arange(1, m + 1)
    threshold = (ranks / m) * q
    rejected = pvalues[order] <= threshold
    # BH: si la hipótesis k se rechaza, se rechazan todas con p menor.
    cut = int(np.max(np.where(rejected, ranks, 0)))
    mask = np.zeros(m, dtype=bool)
    mask[order[:cut]] = True
    return mask
