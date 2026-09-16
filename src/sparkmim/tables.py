"""Builders de tablas de conteo en pase único (``mapInPandas``) + caché en driver.

Diseño (ver docs/DESIGN.md §2):
- **Etapa 1 (screening):** UN ``mapInPandas`` sobre ``df_prep`` emite las celdas
  no nulas de ``crosstab(X_i, Y)`` por feature → un único ``groupBy`` → tablas
  densas (n_x, n_y) en el driver.
- **Etapa 2 (tablas conjuntas):** UN ``mapInPandas`` sobre el subsample emite,
  por partición y por par (i, j) de candidatas, la tabla densa
  ``crosstab(X_i, X_j, Y)`` serializada → se recolectan y se suman en el
  driver → ``TableCache``. Las tablas **par** (X_i, X_j) se derivan en el
  driver como marginal sobre Y de la triple (evita duplicar el cómputo en el
  pase; mismo resultado que calcularlas aparte).

Regla de diseño: nada de tamaño O(n) cruza al driver; todo lo que llega es
agregado y acotado por ``max_cache_cells``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)

__all__ = [
    "TableCache",
    "build_screening_tables",
    "build_joint_tables",
    "dense_from_screening",
    "dense_from_joints",
    "cache_cell_budget",
]

ScreeningSchema = StructType(
    [
        StructField("fid", IntegerType()),
        StructField("x", IntegerType()),
        StructField("y", IntegerType()),
        StructField("count", LongType()),
    ]
)

JointSchema = StructType(
    [
        StructField("fid_i", IntegerType()),
        StructField("fid_j", IntegerType()),
        StructField("table", BinaryType()),
    ]
)


def _screening_partition(
    pdf,
    feature_cols: Sequence[str],
    target_col: str,
    n_x: Sequence[int],
    n_y: int,
):
    """Emite las celdas no nulas de crosstab(X_i, Y) de una partición."""
    y = pdf[target_col].to_numpy().astype(np.int64)
    for fid, col in enumerate(feature_cols):
        x = pdf[col].to_numpy().astype(np.int64)
        counts = np.bincount(x * n_y + y, minlength=n_x[fid] * n_y).reshape(n_x[fid], n_y)
        for a, b in zip(*np.nonzero(counts)):
            yield (fid, int(a), int(b), int(counts[a, b]))


def _joint_partition(
    pdf,
    candidate_cols: Sequence[str],
    target_col: str,
    n_codes: Sequence[int],
    n_y: int,
):
    """Emite, por par (i, j) con i < j, la tabla densa crosstab(X_i, X_j, Y)."""
    y = pdf[target_col].to_numpy().astype(np.int64)
    k = len(candidate_cols)
    xs = [pdf[col].to_numpy().astype(np.int64) for col in candidate_cols]
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = n_codes[i], n_codes[j]
            packed = (xs[i] * nj + xs[j]) * n_y + y
            counts = np.bincount(packed, minlength=ni * nj * n_y).reshape(ni, nj, n_y)
            yield (i, j, counts.tobytes())


def build_screening_tables(
    df: DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    n_x: Sequence[int],
    n_y: int,
) -> DataFrame:
    """UN pase ``mapInPandas`` + un único ``groupBy``: crosstab(X_i, Y) por feature.

    Args:
        df: DataFrame con códigos enteros (columnas de features + target).
        feature_cols: nombres de las columnas de features (en orden).
        target_col: nombre de la columna target (códigos enteros).
        n_x: nº de códigos por feature (misma longitud que ``feature_cols``).
        n_y: nº de códigos del target.

    Returns:
        DataFrame agregado con columnas ``(fid, x, y, count)``.
    """
    # mapInPandas pasa un iterador de batches; se rinde un DataFrame por batch.
    def _func(iterator):
        for pdf in iterator:
            rows = list(
                _screening_partition(pdf, feature_cols, target_col, n_x, n_y)
            )
            out = pd.DataFrame(rows, columns=["fid", "x", "y", "count"])
            yield out.astype(
                {"fid": "int64", "x": "int64", "y": "int64", "count": "int64"}
            )

    out = df.mapInPandas(_func, schema=ScreeningSchema)
    return out.groupBy("fid", "x", "y").agg(F.sum("count").alias("count"))


def build_joint_tables(
    df: DataFrame,
    candidate_cols: Sequence[str],
    target_col: str,
    n_codes: Sequence[int],
    n_y: int,
) -> DataFrame:
    """UN pase ``mapInPandas`` sobre el subsample: tablas densas (X_i, X_j, Y).

    Cada partición emite una fila por par (i, j) con la tabla densa serializada
    (int64, forma (n_i, n_j, n_y)); la suma entre particiones se hace en el
    driver (datos acotados: C(K,2) × n_y·n_i·n_j celdas).

    Returns:
        DataFrame no agregado con columnas ``(fid_i, fid_j, table)``.
    """
    # mapInPandas pasa un iterador de batches; se rinde un DataFrame por batch.
    def _func(iterator):
        for pdf in iterator:
            rows = list(
                _joint_partition(pdf, candidate_cols, target_col, n_codes, n_y)
            )
            out = pd.DataFrame(rows, columns=["fid_i", "fid_j", "table"])
            yield out.astype({"fid_i": "int64", "fid_j": "int64", "table": "object"})

    return df.mapInPandas(_func, schema=JointSchema)


def dense_from_screening(
    agg: DataFrame,
    n_x: Sequence[int],
    n_y: int,
) -> Dict[int, np.ndarray]:
    """Convierte el DataFrame agregado de screening a tablas densas.

    Returns:
        Dict ``{fid: tabla (n_x[fid], n_y) int64}`` (todas las features).
    """
    tables: Dict[int, np.ndarray] = {
        fid: np.zeros((n_x[fid], n_y), dtype=np.int64) for fid in range(len(n_x))
    }
    for r in agg.collect():
        # r["count"]: el atributo r.count chocaría con tuple.count.
        tables[r.fid][r.x, r.y] = r["count"]
    return tables


def dense_from_joints(
    rows_df: DataFrame,
    n_codes: Sequence[int],
    n_y: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Suma las tablas densas por par (i, j) emitidas por las particiones.

    Returns:
        Dict ``{(i, j): tabla (n_i, n_j, n_y) int64}`` para todos i < j.
    """
    k = len(n_codes)
    tables: Dict[Tuple[int, int], np.ndarray] = {}
    for r in rows_df.collect():
        i, j = r.fid_i, r.fid_j
        arr = np.frombuffer(r.table, dtype=np.int64).reshape(n_codes[i], n_codes[j], n_y)
        if (i, j) in tables:
            tables[i, j] += arr
        else:
            tables[i, j] = arr
    for i in range(k):
        for j in range(i + 1, k):
            tables.setdefault((i, j), np.zeros((n_codes[i], n_codes[j], n_y), dtype=np.int64))
    return tables


def cache_cell_budget(n_codes: Sequence[int], n_y: int) -> int:
    """Celdas totales de la caché de tablas conjuntas: pares + triples.

    ``C(K,2)·(n_i·n_j + n_i·n_j·n_y)`` (pares derivados de triples).
    """
    k = len(n_codes)
    total = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += n_codes[i] * n_codes[j] * (1 + n_y)
    return total


class TableCache:
    """Caché en driver de tablas de conteo densas (int64).

    Contiene:
    - ``univariate[fid]``: tabla (n_x, n_y) del screening (etapa 1).
    - ``triples[(i, j)]``: tabla (n_i, n_j, n_y) de la etapa 2.
    - ``pair(i, j)``: marginal (n_i, n_j) de la triple, cacheado.
    """

    def __init__(
        self,
        n_codes: Sequence[int],
        n_y: int,
        univariate: Dict[int, np.ndarray] | None = None,
        triples: Dict[Tuple[int, int], np.ndarray] | None = None,
    ):
        self.n_codes = list(n_codes)
        self.n_y = int(n_y)
        self.univariate = dict(univariate or {})
        self.triples = dict(triples or {})
        self._pairs: Dict[Tuple[int, int], np.ndarray] = {}

    def uni(self, fid: int) -> np.ndarray:
        """Tabla (n_x, n_y) de la feature ``fid`` vs target."""
        return self.univariate[fid]

    def triple(self, i: int, j: int) -> np.ndarray:
        """Tabla (n_i, n_j, n_y) para el par canónico (min, max)."""
        return self.triples[(min(i, j), max(i, j))]

    def pair(self, i: int, j: int) -> np.ndarray:
        """Tabla (n_i, n_j) para el par canónico (min, max), marginal sobre Y."""
        key = (min(i, j), max(i, j))
        if key not in self._pairs:
            self._pairs[key] = self.triples[key].sum(axis=2)
        return self._pairs[key]

    def cells(self) -> int:
        """Celdas totales ocupadas (pares + triples)."""
        return cache_cell_budget(self.n_codes, self.n_y)
