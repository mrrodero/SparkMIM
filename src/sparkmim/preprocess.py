"""Etapa 0 — Esquema y preprocesado (pase 0a + mapeo 0b).

Pase 0a (UN ``mapInPandas`` sobre el df original): por partición y por
columna numérica, cuantiles ``i/b`` (``pandas.quantile``) + min/max; por
columna categórica, ``value_counts()``. Emite filas compactas ``(fid, kind,
...)`` → agregación mínima → driver. Evita N llamadas a ``approxQuantile`` y
N ``groupBy`` (O(N) jobs).

Driver: cuantiles globales ponderados por ``n_part`` → fronteras de bins
(puntos medios entre valores frontera, evita bins degenerados); top-C
categorías por frecuencia + código "other".

Pase 0b (mapeo sin shuffle): cadenas ``when`` por columna aplicando bins y
códigos; missing → código dedicado (o drop). Target: binaria tal cual,
multiclase → códigos, regresión → bins cuantiles (modo histograma).

Se materializa ``df_prep`` (códigos enteros) con ``persist(MEMORY_AND_DISK)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.storagelevel import StorageLevel

from .schema import Schema, build_schema, resolve_bins

__all__ = ["PreparedData", "prepare", "resolve_specs", "apply_mapping"]

# Kinds de las filas emitidas por el pase 0a.
_KIND_QUANTILE = 1  # cuantil (fid, q_index, q_value, n_part)
_KIND_MINMAX = 2  # min/max (fid, which, value)
_KIND_CATEGORY = 3  # categoría (fid, cat_value, count)

StatsSchema = StructType(
    [
        StructField("fid", IntegerType()),
        StructField("kind", IntegerType()),
        StructField("q_index", IntegerType()),
        StructField("q_value", DoubleType()),
        StructField("cat_value", StringType()),
        StructField("count", LongType()),
        StructField("n_part", LongType()),
    ]
)


@dataclass
class PreparedData:
    """Resultado de la etapa 0."""

    df_prep: DataFrame
    schema: Schema
    n_rows: int


def _cat_to_str(value, dtype: str) -> str:
    """Convierte un valor categórico a su representación de texto (coherente
    con el cast a string de Spark en el pase 0b)."""
    if dtype == "boolean":
        return "true" if bool(value) else "false"
    if dtype == "date":
        return pd.Timestamp(value).date().isoformat()
    if dtype == "timestamp":
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _stats_partition(
    pdf: pd.DataFrame,
    columns: Sequence[Tuple[str, str, str]],  # (name, kind, dtype)
    bins: int,
    bins_target: int,
    is_target: Sequence[bool],
):
    """Emite las filas de estadísticas de una partición (pase 0a)."""
    n_part = len(pdf)
    for fid, (name, kind, dtype) in enumerate(columns):
        if kind == "numeric":
            s = pdf[name]
            if not pd.api.types.is_numeric_dtype(s):
                # Override: columna string con valores numéricos.
                s = pd.to_numeric(s, errors="coerce")
            b = bins_target if is_target[fid] else bins
            valid = s.dropna()
            if len(valid) == 0:
                continue
            vmin = float(valid.min())
            vmax = float(valid.max())
            yield (fid, _KIND_MINMAX, 0, vmin, None, 1, n_part)
            yield (fid, _KIND_MINMAX, 1, vmax, None, 1, n_part)
            if b > 1 and vmin < vmax:
                qs = valid.quantile([i / b for i in range(1, b)])
                for i, q in enumerate(qs, start=1):
                    yield (fid, _KIND_QUANTILE, i, float(q), None, 1, n_part)
        else:
            vc = pdf[name].dropna().value_counts()
            for val, c in vc.items():
                yield (fid, _KIND_CATEGORY, -1, 0.0, _cat_to_str(val, dtype), int(c), n_part)


_STATS_COLS = ["fid", "kind", "q_index", "q_value", "cat_value", "count", "n_part"]
_STATS_DTYPES = {
    "fid": "int64",
    "kind": "int64",
    "q_index": "int64",
    "q_value": "float64",
    "cat_value": "object",
    "count": "int64",
    "n_part": "int64",
}


def _stats_pass(df: DataFrame, schema: Schema, config, n_rows: int) -> DataFrame:
    """UN ``mapInPandas`` sobre el df original: estadísticas por columna.

    ``n_rows`` se usa para resolver ``bins="auto"`` antes de emitir cuantiles.
    Nota: ``mapInPandas`` pasa un *iterador* de DataFrames (batches), no una
    partición completa; la función itera y rinde un DataFrame por batch.
    """
    columns = [(f.name, f.kind, f.dtype) for f in schema.features]
    columns.append((schema.target.name, schema.target.kind, schema.target.dtype))
    is_target = [False] * len(schema.features) + [True]
    b = resolve_bins(n_rows, config.bins)

    def _func(iterator):
        for pdf in iterator:
            rows = list(_stats_partition(pdf, columns, b, config.bins_target, is_target))
            out = pd.DataFrame(rows, columns=_STATS_COLS)
            yield out.astype(_STATS_DTYPES)

    return df.mapInPandas(_func, schema=StatsSchema)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Cuantil ponderado: primer valor cuya suma de pesos alcanza q·Σw."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    idx = np.searchsorted(cw, q * cw[-1], side="left")
    return float(v[min(idx, len(v) - 1)])


def _numeric_edges(
    minmax: Dict[int, float],
    quantiles: Dict[int, List[Tuple[float, int]]],
    b: int,
) -> Tuple[tuple, int]:
    """Fronteras de bins y nº de bins a partir de min/max y cuantiles ponderados.

    ``quantiles[i]`` es una lista de pares ``(valor, n_part)`` (uno por
    batch). Devuelve ``(edges, n_bins)`` donde ``edges`` son puntos medios
    entre valores frontera consecutivos (evita bins vacíos con datos
    discretos).
    """
    # Cuantiles ponderados q1..q_{b-1} (uno por posición i/b).
    raw_q = []
    for i in range(1, b):
        if i in quantiles:
            pairs = quantiles[i]
            vals = np.array([p[0] for p in pairs], dtype=float)
            weights = np.array([p[1] for p in pairs], dtype=float)
            raw_q.append(_weighted_quantile(vals, weights, i / b))
    boundaries = sorted(set([minmax[0], minmax[1]] + raw_q))
    d = len(boundaries)
    if d <= 1:
        return (), 1
    if d <= b:
        # Datos discretos / baja cardinalidad: D valores frontera → D bins
        # (uno por valor distinto), con cortes en los puntos medios.
        edges = tuple(
            (boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(d - 1)
        )
        return edges, d
    # Datos continuos: d == b + 1 → b bins usando los cuantiles como cortes.
    return tuple(raw_q), b


def resolve_specs(
    schema: Schema,
    stats_rows: Sequence[tuple],
    config,
    n_rows: int,
) -> Schema:
    """Driver: resuelve cuantiles globales y top-C categorías; rellena specs.

    Args:
        schema: esquema con tipos (specs sin rellenar).
        stats_rows: filas ``(fid, kind, q_index, q_value, cat_value, count,
            n_part)`` del pase 0a.
        config: ``SelectorConfig``.
        n_rows: nº total de filas (para ``bins="auto"``).

    Returns:
        ``schema`` con las specs rellenadas (in-mutación).
    """
    minmax: Dict[int, Dict[int, float]] = {}
    quantiles: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}
    categories: Dict[int, Dict[str, int]] = {}

    for fid, kind, q_index, q_value, cat_value, count, n_part in stats_rows:
        if kind == _KIND_MINMAX:
            minmax.setdefault(fid, {})[q_index] = q_value
        elif kind == _KIND_QUANTILE:
            quantiles.setdefault(fid, {}).setdefault(q_index, []).append((q_value, n_part))
        elif kind == _KIND_CATEGORY:
            categories.setdefault(fid, {})[cat_value] = categories.get(fid, {}).get(cat_value, 0) + count

    all_specs = list(schema.features) + [schema.target]
    b = resolve_bins(max(n_rows, 2), config.bins)

    for fid, spec in enumerate(all_specs):
        if spec.kind == "numeric":
            b_col = config.bins_target if spec.is_target else b
            mm = minmax.get(fid, {})
            if len(mm) < 2:
                # Columna totalmente nula o vacía.
                spec.n_codes = 1
                spec.bin_edges = ()
                if config.missing == "category":
                    spec.n_codes = 2
                    spec.missing_code = 1
                continue
            edges, n_bins = _numeric_edges(mm, quantiles.get(fid, {}), b_col)
            spec.bin_edges = edges
            spec.n_codes = n_bins
            if config.missing == "category":
                spec.missing_code = n_bins
                spec.n_codes = n_bins + 1
        else:
            counts = categories.get(fid, {})
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if spec.is_target:
                # El target no se trunca: todas las clases conservan código.
                top = ordered
            else:
                top = ordered[: config.max_categories]
            spec.category_codes = {val: code for code, (val, _) in enumerate(top)}
            n_distinct = len(ordered)
            has_other = (not spec.is_target) and n_distinct > len(top)
            n_codes = len(top) + (1 if has_other else 0)
            spec.n_codes = n_codes
            if config.missing == "category":
                spec.missing_code = n_codes
                spec.n_codes = n_codes + 1
            # Guardar si hay "other" (para el fallback del mapeo).
            spec.has_other = has_other

    return schema


def _numeric_expr(col, spec, config):
    """Expresión Spark: códigos de bins para una columna numérica."""
    if spec.dtype in ("string", "varchar"):
        # Override: columna string con valores numéricos.
        col = col.cast("double")
    expr = F.lit(0)
    for i, e in enumerate(spec.bin_edges, start=1):
        expr = F.when(col >= F.lit(e), i).otherwise(expr)
    if config.missing == "category" and spec.missing_code is not None:
        expr = F.when(col.isNull(), spec.missing_code).otherwise(expr)
    return expr


def _categorical_expr(col, spec, config):
    """Expresión Spark: códigos de categorías para una columna categórica."""
    # Cast a string coherente con el pase 0a.
    if spec.dtype == "timestamp":
        cs = F.date_format(col, "yyyy-MM-dd HH:mm:ss")
    else:
        cs = col.cast("string")
    fallback = len(spec.category_codes) if spec.has_other else 0
    expr = F.lit(fallback)
    for val, code in reversed(list(spec.category_codes.items())):
        expr = F.when(cs == F.lit(val), code).otherwise(expr)
    if config.missing == "category" and spec.missing_code is not None:
        expr = F.when(cs.isNull(), spec.missing_code).otherwise(expr)
    return expr


def apply_mapping(df: DataFrame, schema: Schema, config) -> DataFrame:
    """Pase 0b: mapeo sin shuffle a códigos enteros (``df_prep``)."""
    if not schema.ready():
        raise ValueError("El esquema no está resuelto (llamar resolve_specs antes)")
    cols = list(schema.features) + [schema.target]
    out = df
    if config.missing == "drop":
        notnull = F.lit(True)
        for spec in cols:
            notnull = notnull & F.col(spec.name).isNotNull()
        out = out.filter(notnull)
    for spec in cols:
        col = F.col(spec.name)
        if spec.kind == "numeric":
            expr = _numeric_expr(col, spec, config)
        else:
            expr = _categorical_expr(col, spec, config)
        out = out.withColumn(spec.name, expr.cast("int"))
    # Conservar solo columnas necesarias (features + target).
    keep = [spec.name for spec in cols]
    return out.select(*keep)


def prepare(df: DataFrame, config) -> PreparedData:
    """Etapa 0 completa: pase 0a + resolución en driver + mapeo 0b.

    Devuelve ``PreparedData`` con ``df_prep`` persistido (MEMORY_AND_DISK).
    """
    if config.estimator != "histogram":
        raise ValueError(
            "prepare() es para el modo 'histogram'; el modo 'ksg' tiene su "
            "propio camino (ver sparkmim.info.ksg)"
        )
    schema = build_schema(df, config)
    # count() para resolver bins="auto" antes del pase 0a (sobre el df original).
    n_total = df.count()
    stats_df = _stats_pass(df, schema, config, n_total)
    stats_rows = stats_df.collect()
    resolve_specs(schema, stats_rows, config, n_total)
    df_prep = apply_mapping(df, schema, config)
    df_prep = df_prep.persist(StorageLevel.MEMORY_AND_DISK)
    # n_rows refleja las filas tras el drop (si missing="drop"); con
    # "category" coincide con n_total.
    n_rows = df_prep.count()
    return PreparedData(df_prep=df_prep, schema=schema, n_rows=n_rows)
