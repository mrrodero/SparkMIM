"""Significancia estadística (Hito 3, etapa 1).

Tres piezas, todas puras (numpy/scipy, sin Spark):

- ``chi2_pvalue``: estadístico de razón de verosimilitud ``G² = 2·n·MI``.
  Bajo la nula de independencia, ``G² ~ χ²_{(n_x-1)(n_y-1)}``. O(1) por feature.
- ``permutation_pvalue``: p-valor empírico a partir de la distribución nula de
  MI generada por B permutaciones de y: ``(1 + #null >= observed) / (B + 1)``.
- ``bh_fdr``: control de tasa de descubrimiento falsa de Benjamini-Hochberg
  (q = 0.05 por defecto).

El *test de permutación distribuido* (el ``mapInPandas`` que genera la
distribución nula sobre un subsample) vive en ``screen.py``; este módulo solo
consume los MI nulos ya calculados.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["chi2_pvalue", "permutation_pvalue", "bh_fdr"]


def chi2_pvalue(table: np.ndarray, mi: float) -> float:
    """p-valor χ² para ``MI(X;Y)`` dada la tabla conjunta ``(n_x, n_y)``.

    ``G² = 2·n·MI`` con ``n`` el total de la tabla; bajo la nula,
    ``G² ~ χ²`` con ``df = (n_x - 1)(n_y - 1)``. Devuelve 1.0 si la tabla es
    vacía o el df no es positivo (una dimensión degenerada => no hay test).
    """
    n = int(table.sum())
    if n == 0:
        return 1.0
    n_x, n_y = table.shape
    df = (n_x - 1) * (n_y - 1)
    if df <= 0:
        return 1.0
    g2 = 2.0 * n * float(mi)
    return float(stats.chi2.sf(g2, df))


def permutation_pvalue(mi_observed: float, mi_null: np.ndarray) -> float:
    """p-valor de permutación: proporción de MI nulos >= observado.

    ``mi_null`` son los MI recomputados bajo B permutaciones de y. Se usa la
    corrección ``+1`` (el observado cuenta como un valor más de la nula):
    ``(1 + #null >= observed) / (B + 1)``. Con B = 0 no hay evidencia => 1.0.
    """
    mi_null = np.asarray(mi_null, dtype=float)
    B = int(mi_null.size)
    if B == 0:
        return 1.0
    return float((1.0 + float(np.sum(mi_null >= float(mi_observed)))) / (B + 1))


def bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Máscara de significancia de Benjamini-Hochberg a nivel FDR ``q``.

    Ordena los p-valores ascendentes ``p_(1) <= ... <= p_(m)`` y rechaza las
    ``k`` primeras hipótesis donde ``k`` es el mayor índice con
    ``p_(k) <= (k/m)·q``. Devuelve un array bool de la misma longitud que
    ``pvalues`` (True = significativa). Con m = 0 devuelve array vacío.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = int(pvalues.size)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvalues, kind="stable")
    ranked = pvalues[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    sig = ranked <= thresholds
    if not sig.any():
        return np.zeros(m, dtype=bool)
    k = int(np.max(np.where(sig)[0])) + 1
    result = np.zeros(m, dtype=bool)
    result[order[:k]] = True
    return result
