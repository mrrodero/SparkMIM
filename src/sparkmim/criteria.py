"""Criterios informacionales greedy (Hito 4, etapa 3).

Criterios rápidos (solo aritmética sobre la ``TableCache`` de la etapa 2 y el
MI univariante de la etapa 1, O(1) por candidata en el driver):

- ``mrmr``: ``MI(X;Y) − (1/|S|)·Σ_{Xi∈S} MI(X;Xi)``   [tablas par]
- ``mim``:  ``MI(X;Y) − Σ_{Xi∈S} MI(X;Xi)``             [tablas par]
- ``jmi``:  ``Σ_{Xi∈S} CMI(X;Y|Xi)``                     [tablas triple]
- ``jmim``: ``min_{Xi∈S} CMI(X;Y|Xi)``                   [tablas triple] (default)

``cmim`` es el único que requiere un pase ``mapInPandas`` extra por ronda:
construye la conjunta ``(X, Y, S_m)`` sobre el subsample con ``S_m`` = top-m
por MI univariante (m=2). Ver :func:`cmim_scores`. Aproximación documentada
(CMIM exacto inabordable → acotado por diseño).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, IntegerType, StructField, StructType

from .info.entropy import (
    conditional_mi,
    conditional_mi_multi,
    mutual_information,
)

__all__ = ["CRITERIA", "criterion_score", "cmim_scores"]

CRITERIA = ("mrmr", "mim", "jmi", "jmim", "cmim")


def _triple_oriented(i: int, j: int, cache) -> np.ndarray:
    """Tabla triple con ``X_i`` en eje 0, ``X_j`` en eje 1, ``Y`` en eje 2.

    ``cache.triple(i, j)`` devuelve la tabla en orden canónico ``(min, max)``;
    si ``i > j`` se intercambian los ejes 0 y 1.
    """
    t = cache.triple(i, j)
    if i < j:
        return t
    return t.transpose(1, 0, 2)


def _mi_xy(i: int, cache) -> float:
    return mutual_information(cache.uni(i))


def _mi_pair(i: int, j: int, cache) -> float:
    """MI(X_i; X_j): simétrica, la orientación del par canónico no importa."""
    return mutual_information(cache.pair(i, j))


def _cmi_y_given(i: int, j: int, cache) -> float:
    """CMI(X_i; Y | X_j) desde la triple (n_i, n_j, n_y)."""
    t = _triple_oriented(i, j, cache)  # (n_i, n_j, n_y)
    return conditional_mi(t.transpose(0, 2, 1))  # → (n_i, n_y, n_j)


def criterion_score(
    x: int,
    S: Sequence[int],
    cache,
    mi_xy: np.ndarray,
    criterion: str,
) -> float:
    """Puntaje del criterio rápido para la candidata ``x`` dado el conjunto ``S``.

    Args:
        x: índice de la candidata (0..K-1), ``x ∉ S``.
        S: lista de índices ya seleccionados (0..K-1).
        cache: ``TableCache`` de la etapa 2.
        mi_xy: array ``MI(X_i; Y)`` por candidata (etapa 1).
        criterion: ``"mrmr" | "mim" | "jmi" | "jmim"``.

    Returns:
        Puntaje en nats. Con ``S`` vacío (ronda 1) devuelve ``MI(X;Y)``.
    """
    if criterion == "mrmr":
        if not S:
            return float(mi_xy[x])
        redundancy = float(np.mean([_mi_pair(x, s, cache) for s in S]))
        return float(mi_xy[x] - redundancy)
    if criterion == "mim":
        if not S:
            return float(mi_xy[x])
        redundancy = float(sum(_mi_pair(x, s, cache) for s in S))
        return float(mi_xy[x] - redundancy)
    if criterion == "jmi":
        if not S:
            return float(mi_xy[x])
        return float(sum(_cmi_y_given(x, s, cache) for s in S))
    if criterion == "jmim":
        if not S:
            return float(mi_xy[x])
        return float(min(_cmi_y_given(x, s, cache) for s in S))
    raise ValueError(f"criterio no soportado por criterion_score: {criterion!r}")


# ---------------------------------------------------------------------------
# CMIM: pase mapInPandas por ronda sobre el subsample.
# ---------------------------------------------------------------------------


def _cmim_partition(
    pdf: pd.DataFrame,
    candidate_cols: Sequence[str],
    target_col: str,
    n_codes: Sequence[int],
    n_y: int,
    s_m: Sequence[int],
):
    """Emite, por candidata ``i``, la conjunta ``(X_i, Y, S_m \\ {i})`` densa.

    El conjunto de condicionamiento efectivo es ``S_m`` menos ``i`` (evita
    condicionar en la propia candidata). Forma resultante:
    - ``S_m \\ {i}`` vacío: ``(n_i, n_y)``
    - 1 elemento: ``(n_i, n_y, n_c)``
    - 2 elementos: ``(n_i, n_y, n_c0, n_c1)``
    """
    y = pdf[target_col].to_numpy().astype(np.int64)
    k = len(candidate_cols)
    xs = [pdf[col].to_numpy().astype(np.int64) for col in candidate_cols]
    for i in range(k):
        cond = [s for s in s_m if s != i]
        ni = n_codes[i]
        if len(cond) == 0:
            packed = xs[i] * n_y + y
            counts = np.bincount(packed, minlength=ni * n_y).reshape(ni, n_y)
        elif len(cond) == 1:
            c0 = cond[0]
            n0 = n_codes[c0]
            packed = (xs[i] * n_y + y) * n0 + xs[c0]
            counts = np.bincount(packed, minlength=ni * n_y * n0).reshape(ni, n_y, n0)
        else:
            c0, c1 = cond[0], cond[1]
            n0, n1 = n_codes[c0], n_codes[c1]
            packed = ((xs[i] * n_y + y) * n0 + xs[c0]) * n1 + xs[c1]
            counts = np.bincount(
                packed, minlength=ni * n_y * n0 * n1
            ).reshape(ni, n_y, n0, n1)
        yield (i, counts.tobytes())


_CMIM_SCHEMA = StructType(
    [
        StructField("fid", IntegerType()),
        StructField("table", BinaryType()),
    ]
)


def cmim_scores(
    df: DataFrame,
    candidate_cols: Sequence[str],
    target_col: str,
    n_codes: Sequence[int],
    n_y: int,
    s_m: Sequence[int],
) -> np.ndarray:
    """CMI(X_i; Y | S_m \\ {i}) para todas las candidatas ``i`` (UN pase).

    Args:
        df: DataFrame con códigos enteros (features + target), ya subsampleado.
        candidate_cols: nombres de las columnas de candidatas (en orden).
        target_col: nombre de la columna target.
        n_codes: nº de códigos por candidata (misma longitud que ``candidate_cols``).
        n_y: nº de códigos del target.
        s_m: índices (0..K-1) del conjunto de condicionamiento ``S_m`` (m ≤ 2).

    Returns:
        Array ``CMI(X_i; Y | S_m \\ {i})`` por candidata (longitud K).
    """
    k = len(candidate_cols)

    def _func(iterator):
        for pdf in iterator:
            rows = list(
                _cmim_partition(pdf, candidate_cols, target_col, n_codes, n_y, s_m)
            )
            out = pd.DataFrame(rows, columns=["fid", "table"])
            yield out.astype({"fid": "int64", "table": "object"})

    rows_df = df.mapInPandas(_func, schema=_CMIM_SCHEMA)

    # Suma las tablas por candidata en el driver (datos acotados).
    tables: Dict[int, np.ndarray] = {}
    for r in rows_df.collect():
        i = r.fid
        cond = [s for s in s_m if s != i]
        ni = n_codes[i]
        if len(cond) == 0:
            shape = (ni, n_y)
        elif len(cond) == 1:
            shape = (ni, n_y, n_codes[cond[0]])
        else:
            shape = (ni, n_y, n_codes[cond[0]], n_codes[cond[1]])
        arr = np.frombuffer(r.table, dtype=np.int64).reshape(shape)
        if i in tables:
            tables[i] += arr
        else:
            tables[i] = arr

    cmis = np.zeros(k, dtype=np.float64)
    for i in range(k):
        t = tables.get(i)
        if t is None or t.sum() == 0:
            cmis[i] = 0.0
            continue
        if t.ndim == 2:
            cmis[i] = mutual_information(t)
        elif t.ndim == 3:
            cmis[i] = conditional_mi(t)
        else:
            cmis[i] = conditional_mi_multi(t)
    return cmis
