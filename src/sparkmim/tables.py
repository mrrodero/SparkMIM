"""Builders de tablas de conteo en pase único (``mapInPandas``) + caché en driver.

Diseño (ver docs/DESIGN.md §2):
- **Núcleo de conteo:** ``joint_counts(cols, sizes)`` — pack→bincount→reshape
  una sola vez; las tres pasadas de conteo (screening, triples, CMIM) son
  adaptadores delgados sobre él y la forma de cada tabla es ``tuple(sizes)``
  en el orden de ``cols``.
- **Etapa 1 (screening):** UN ``mapInPandas`` sobre ``df_prep`` emite las celdas
  no nulas de ``crosstab(X_i, Y)`` por feature → un único ``groupBy`` → tablas
  densas (n_x, n_y) en el driver.
- **Etapa 2 (tablas conjuntas):** UN ``mapInPandas`` sobre el subsample emite,
  por partición y por par (i, j) de candidatas, la tabla densa
  ``crosstab(X_i, X_j, Y)`` serializada → se recolectan y se suman en el
  driver → ``TableCache``. Las tablas **par** (X_i, X_j) se derivan en el
  driver como marginal sobre Y de la triple (evita duplicar el cómputo en el
  pase; mismo resultado que calcularlas aparte).
- **Etapa 3 (CMIM, por ronda):** UN ``mapInPandas`` sobre el subsample emite,
  por candidata, la conjunta ``(X_i, Y, S_m \\ {i})`` → se suma en el driver →
  ``cmim_scores``.

Regla de diseño: nada de tamaño O(n) cruza al driver; todo lo que llega es
agregado y acotado por ``max_cache_cells``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .screen import ScreenResult

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

from .info.entropy import conditional_mi

__all__ = [
    "joint_counts",
    "cmim_scores",
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

_CMIM_SCHEMA = StructType(
    [
        StructField("fid", IntegerType()),
        StructField("table", BinaryType()),
    ]
)


def joint_counts(
    cols: Sequence[np.ndarray],
    sizes: Sequence[int],
) -> np.ndarray:
    """Tabla densa de conteos conjunta de ``cols``, en el orden dado.

    Núcleo pack→bincount→reshape: empaqueta cada fila en un índice plano
    recorriendo las columnas en orden y cuenta. La forma resultante es
    ``tuple(sizes)`` en el mismo orden que ``cols``; las pasadas de conteo la
    usan tal cual (2, 3 o 4 columnas).

    Args:
        cols: arrays de códigos (misma longitud).
        sizes: nº de códigos por columna (misma longitud que ``cols``).

    Returns:
        Tabla de conteos densa (int64) con forma ``tuple(sizes)``.
    """
    packed = np.zeros(len(cols[0]), dtype=np.int64)
    for c, n in zip(cols, sizes):
        packed = packed * n + c
    return np.bincount(packed, minlength=int(np.prod(sizes))).reshape(tuple(sizes))


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
        counts = joint_counts([x, y], (n_x[fid], n_y))
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
            counts = joint_counts([xs[i], xs[j], y], (n_codes[i], n_codes[j], n_y))
            yield (i, j, counts.tobytes())


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
    condicionar en la propia candidata). La forma sale del orden de las
    columnas: ``(n_i, n_y) + tuple(n_codes[c] for c in cond)`` — con ``cond``
    vacío es la tabla univariante ``(n_i, n_y)``.
    """
    y = pdf[target_col].to_numpy().astype(np.int64)
    k = len(candidate_cols)
    xs = [pdf[col].to_numpy().astype(np.int64) for col in candidate_cols]
    for i in range(k):
        cond = [s for s in s_m if s != i]
        counts = joint_counts(
            [xs[i], y] + [xs[c] for c in cond],
            (n_codes[i], n_y) + tuple(n_codes[c] for c in cond),
        )
        yield (i, counts.tobytes())


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
        shape = (n_codes[i], n_y) + tuple(n_codes[c] for c in s_m if c != i)
        arr = np.frombuffer(r.table, dtype=np.int64).reshape(shape)
        if i in tables:
            tables[i] += arr
        else:
            tables[i] = arr

    cmis = np.zeros(k, dtype=np.float64)
    for i in range(k):
        t = tables.get(i)
        if t is None or t.sum() == 0:
            continue
        # conditional_mi cubre todos los grados: 2D (cond vacío) es MI,
        # 3D un condicionante, 4D dos.
        cmis[i] = conditional_mi(t)
    return cmis


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
    - ``triples[(i, j)]``: tabla (n_i, n_j, n_y) de la etapa 2, con clave
      canónica (min, max).

    Los accesores esconden la canonicidad y la orientación de los ejes de
    almacenamiento: el llamador pide "la tabla de x condicionada en z" y
    recibe la tabla en la orientación que esperan las funciones de entropía
    (``mutual_information``: (n_x, n_y); ``conditional_mi``: (n_x, n_y, n_z)).

    La traducción de índices globales de Feature a posiciones de Candidate
    (``0..K-1``, el espacio del seam ``InformationOracle``) vive en
    ``from_screening``, el camino del pipeline. ``__init__`` queda para
    construcción directa (tests que arman cachés sin resultado de Screening).
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

    @classmethod
    def from_screening(
        cls,
        screen_result: "ScreenResult",
        candidates: Sequence[int],
        triples: Dict[Tuple[int, int], np.ndarray],
    ) -> "TableCache":
        """Caché a partir del Screening: la traducción global→posición vive aquí.

        ``candidates`` son índices globales de Feature (el espacio del
        Screening); la caché trabaja en posiciones ``0..K-1`` (el espacio del
        seam ``InformationOracle``, ADR-0001). ``n_codes`` y ``n_y`` se derivan
        de las formas de las tablas de screening, y el cableado se valida por
        construcción: claves canónicas dentro de ``0..K-1``, todos los pares
        presentes y formas ``(n_i, n_j, n_y)`` coherentes.

        Args:
            screen_result: resultado de la etapa 1 (``tables`` por fid global).
            candidates: índices globales de las Candidates, en orden de selección.
            triples: tablas ``(n_i, n_j, n_y)`` con clave canónica en el espacio
                de posiciones de Candidate.

        Raises:
            ValueError: candidates vacío, falta una tabla de screening, una clave
                de ``triples`` que no es par canónico dentro de ``0..K-1``, un
                par ausente, o una forma que no corresponde a ``(n_i, n_j, n_y)``.
        """
        k = len(candidates)
        if k == 0:
            raise ValueError("from_screening requiere al menos una candidata")

        univariate: Dict[int, np.ndarray] = {}
        for i, fid in enumerate(candidates):
            if fid not in screen_result.tables:
                raise ValueError(f"falta la tabla de screening de la candidata {fid}")
            t = screen_result.tables[fid]
            if t.ndim != 2:
                raise ValueError(
                    f"tabla de screening de la candidata {i} con forma {tuple(t.shape)}, "
                    "se espera 2D"
                )
            univariate[i] = t

        n_codes = [int(t.shape[0]) for t in univariate.values()]
        n_y = int(univariate[0].shape[1])
        for i, t in univariate.items():
            if int(t.shape[1]) != n_y:
                raise ValueError(
                    f"tabla de screening de la candidata {i} con n_y={t.shape[1]}, "
                    f"se espera n_y={n_y}"
                )

        for key in triples:
            if not (isinstance(key, tuple) and len(key) == 2):
                raise ValueError(f"clave de triples no es un par: {key!r}")
            i, j = key
            if i >= j or j >= k:
                raise ValueError(
                    f"clave {key!r} no es un par canónico dentro de 0..{k - 1}"
                )

        expected_pairs = {(i, j) for i in range(k) for j in range(i + 1, k)}
        for key in expected_pairs:
            if key not in triples:
                raise ValueError(f"falta el par canónico {key} en triples")

        for key, t in triples.items():
            i, j = key
            expected_shape = (n_codes[i], n_codes[j], n_y)
            if tuple(t.shape) != expected_shape:
                raise ValueError(
                    f"triple {key} con forma {tuple(t.shape)}, se espera {expected_shape}"
                )

        return cls(n_codes=n_codes, n_y=n_y, univariate=univariate, triples=triples)

    def uni(self, fid: int) -> np.ndarray:
        """Tabla (n_x, n_y) de la feature ``fid`` vs target."""
        return self.univariate[fid]

    def triple(self, i: int, j: int) -> np.ndarray:
        """Tabla (n_i, n_j, n_y) para el par canónico (min, max)."""
        return self.triples[(min(i, j), max(i, j))]

    def cmi_table(self, i: int, j: int) -> np.ndarray:
        """Tabla (n_i, n_y, n_j) de ``CMI(X_i; Y | X_j)`` en la orientación
        que espera ``conditional_mi`` (x, Y, z), sin importar el orden de
        (i, j).

        La canonicidad (min, max) y la permutación de ejes viven aquí: el
        llamador recibe la tabla lista para ``conditional_mi``. Vista O(1)
        (transpose, sin copia).
        """
        t = self.triples[(min(i, j), max(i, j))]
        if i > j:
            t = t.transpose(1, 0, 2)  # (n_i, n_j, n_y)
        return t.transpose(0, 2, 1)  # (n_i, n_y, n_j)

    def pair(self, i: int, j: int) -> np.ndarray:
        """Tabla (n_i, n_j) para el par canónico (min, max), marginal sobre Y."""
        key = (min(i, j), max(i, j))
        if key not in self._pairs:
            self._pairs[key] = self.triples[key].sum(axis=2)
        return self._pairs[key]

    def cells(self) -> int:
        """Celdas totales ocupadas (pares + triples)."""
        return cache_cell_budget(self.n_codes, self.n_y)
