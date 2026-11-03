"""Etapa 0 — Esquema y preprocesado (pase 0a + mapeo 0b).

Pase 0a (UN ``mapInPandas`` sobre el df original): por batch, min/max y
cuantiles ``i/b`` de las columnas numéricas (cálculo vectorizado con
``np.partition``, interpolación lineal idéntica a ``pandas.quantile``;
columnas con NaN caen al camino pandas con ``dropna``) y ``value_counts()``
de las categóricas. Salida **compacta**: una fila ``(fid, min_v, max_v,
quantiles[])`` por columna numérica por batch y una fila ``(fid, cat_value,
count)`` por valor categórico. Evita N llamadas a ``approxQuantile`` y N
``groupBy`` (O(N) jobs) y reduce el output del UDF un factor ~b+1 respecto
al formato largo anterior (una fila por cuantil).

Driver: min/max global (mín/máx sobre los batches) + cuantiles globales
ponderados por ``n_part`` → fronteras de bins (puntos medios entre valores
frontera, evita bins degenerados); top-C categorías por frecuencia + código
"other".

Pase 0b (mapeo sin shuffle): cadenas ``when`` por columna aplicando bins y
códigos; missing → código dedicado (o drop). Target según la Task resuelta:
clasificación (binaria/multiclase) → un código por valor distinto, continua
→ bins cuantiles (estimador de histograma).

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
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.storagelevel import StorageLevel

from .schema import Schema, build_schema, resolve_bins, resolve_task

__all__ = ["PreparedData", "prepare", "resolve_specs", "apply_mapping"]

# Kinds de las filas emitidas por el pase 0a (formato compacto).
_KIND_NUMERIC = 1  # (fid, min_v, max_v, quantiles[]) — una por columna por batch
_KIND_CATEGORY = 3  # (fid, cat_value, count) — una por valor por batch

StatsSchema = StructType(
    [
        StructField("fid", IntegerType()),
        StructField("kind", IntegerType()),
        StructField("cat_value", StringType()),
        StructField("count", LongType()),
        StructField("n_part", LongType()),
        StructField("min_v", DoubleType()),
        StructField("max_v", DoubleType()),
        StructField("quantiles", ArrayType(DoubleType())),
    ]
)


@dataclass
class PreparedData:
    """Resultado de la etapa 0."""

    df_prep: DataFrame
    schema: Schema
    n_rows: int
    task: str


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


def _quantile_interp(
    col: np.ndarray, b: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuantiles ``i/b`` (``i=1..b-1``) de ``col`` sin NaNs.

    Misma semántica que ``pandas.Series.quantile`` (método ``linear``):
    índice ``q*(n-1)`` interpolando linealmente entre los ordenados
    adyacentes, calculado con ``np.partition`` (sin ordenación completa).

    Returns:
        ``(pos, vlo, vhi)``: posiciones fraccionales y los dos ordenados
        adyacentes (mismos índices, listas alineadas).
    """
    n = col.shape[0]
    pos = (n - 1) * (np.arange(1, b) / b)
    klo = np.floor(pos).astype(np.int64)
    khi = np.minimum(np.ceil(pos).astype(np.int64), n - 1)
    vlo = np.take(np.partition(col, klo), klo)
    vhi = np.take(np.partition(col, khi), khi)
    return pos, vlo, vhi


def _stats_partition(
    pdf: pd.DataFrame,
    columns: Sequence[Tuple[str, str, str]],  # (name, kind, dtype)
    bins: int,
    bins_target: int,
    is_target: Sequence[bool],
):
    """Filas compactas de estadísticas de un batch (pase 0a).

    Una fila ``_KIND_NUMERIC`` por columna numérica (``min_v``, ``max_v`` y
    cuantiles ``i/b`` en un array; ``[]`` si la columna es constante) y una
    ``_KIND_CATEGORY`` por (columna, valor). Columnas numéricas sin NaNs en
    el batch: cálculo vectorizado (``_quantile_interp``); con NaNs: camino
    pandas con ``dropna`` (misma semántica que antes).
    """
    n_part = len(pdf)
    rows: List[tuple] = []
    num_idx = [i for i, (_nm, kind, _d) in enumerate(columns) if kind == "numeric"]
    if num_idx and n_part > 0:
        X = pdf.iloc[:, num_idx].to_numpy(dtype=np.float64)  # (rows, Nn)
        n_rows = X.shape[0]
        has_nan = np.isnan(X)
        if has_nan.any():
            # Camino conservador: cualquier NaN en el batch → pandas por
            # columna (semántica ``dropna`` idéntica a la anterior).
            for fi in num_idx:
                name, kind, dtype = columns[fi]
                s = pdf[name]
                if not pd.api.types.is_numeric_dtype(s):
                    # Override: columna string con valores numéricos.
                    s = pd.to_numeric(s, errors="coerce")
                b = bins_target if is_target[fi] else bins
                valid = s.dropna()
                if len(valid) == 0:
                    continue
                vmin = float(valid.min())
                vmax = float(valid.max())
                qs: List[float] = []
                if b > 1 and vmin < vmax and len(valid) > 1:
                    qs = [float(q) for q in valid.quantile([i / b for i in range(1, b)])]
                rows.append((fi, _KIND_NUMERIC, None, None, n_part, vmin, vmax, qs))
        else:
            # Fast path: sin NaN en el batch → min/max/cuantiles vectorizados
            # por grupo de ``b`` (features vs target pueden diferir).
            vmin = X.min(axis=0)
            vmax = X.max(axis=0)
            groups: Dict[int, List[int]] = {}
            for j, fi in enumerate(num_idx):
                b_col = bins_target if is_target[fi] else bins
                if (vmin[j] < vmax[j]) and b_col > 1 and n_rows > 1:
                    groups.setdefault(b_col, []).append(j)
            quant_by_j: Dict[int, np.ndarray] = {}
            for b_col, js in groups.items():
                Xg = X[:, js]
                pos = (n_rows - 1) * (np.arange(1, b_col) / b_col)
                klo = np.floor(pos).astype(np.int64)
                khi = np.minimum(np.ceil(pos).astype(np.int64), n_rows - 1)
                Plo = np.partition(Xg, klo, axis=0)
                Phi = np.partition(Xg, khi, axis=0)
                q = Plo[klo, :] + (Phi[khi, :] - Plo[klo, :]) * (pos - klo)[:, None]
                for t, j in enumerate(js):
                    quant_by_j[j] = q[:, t]
            for j, fi in enumerate(num_idx):
                if vmin[j] < vmax[j]:
                    rows.append(
                        (
                            fi,
                            _KIND_NUMERIC,
                            None,
                            None,
                            n_part,
                            float(vmin[j]),
                            float(vmax[j]),
                            [float(v) for v in quant_by_j.get(j, [])],
                        )
                    )
                else:
                    rows.append(
                        (
                            fi,
                            _KIND_NUMERIC,
                            None,
                            None,
                            n_part,
                            float(vmin[j]),
                            float(vmax[j]),
                            [],
                        )
                    )
    for i, (name, kind, dtype) in enumerate(columns):
        if kind != "numeric":
            vc = pdf[name].dropna().value_counts()
            for val, c in vc.items():
                rows.append(
                    (i, _KIND_CATEGORY, _cat_to_str(val, dtype), int(c), n_part, None, None, None)
                )
    return rows


_STATS_COLS = ["fid", "kind", "cat_value", "count", "n_part", "min_v", "max_v", "quantiles"]
_STATS_DTYPES = {
    "fid": "int64",
    "kind": "int64",
    "cat_value": "object",
    "count": "int64",
    "n_part": "int64",
    "min_v": "float64",
    "max_v": "float64",
    "quantiles": "object",
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
            rows = _stats_partition(pdf, columns, b, config.bins_target, is_target)
            out = pd.DataFrame(rows, columns=_STATS_COLS)
            for col, dt in _STATS_DTYPES.items():
                if dt == "object":
                    continue
                if dt.startswith("int"):
                    # None en columnas int (p. ej. ``count`` de filas numéricas).
                    out[col] = out[col].fillna(0).astype(dt)
                else:
                    out[col] = out[col].astype(dt)
            yield out

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
        stats_rows: filas compactas del pase 0a:
            ``(fid, kind, cat_value, count, n_part, min_v, max_v,
            quantiles[])``.
        config: ``SelectorConfig``.
        n_rows: nº total de filas (para ``bins="auto"``).

    Returns:
        ``schema`` con las specs rellenadas (in-mutación).
    """
    minmax: Dict[int, Dict[int, float]] = {}
    quantiles: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}
    categories: Dict[int, Dict[str, int]] = {}

    for fid, kind, cat_value, count, n_part, min_v, max_v, quant_vals in stats_rows:
        if kind == _KIND_NUMERIC:
            if min_v is None or max_v is None:
                continue
            mm = minmax.setdefault(fid, {})
            # Mín/máx GLOBAL: el pase 0a emite min/max por batch.
            mm[0] = min_v if 0 not in mm else min(mm[0], min_v)
            mm[1] = max_v if 1 not in mm else max(mm[1], max_v)
            if quant_vals:
                for i, qv in enumerate(quant_vals, start=1):
                    quantiles.setdefault(fid, {}).setdefault(i, []).append((qv, n_part))
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
    """Pase 0b: mapeo sin shuffle a códigos enteros (``df_prep``).

    El mapeo se emite como **un único ``select``** (un solo ``Project``
    de N+1 campos), no como N ``withColumn`` encadenados: cada
    ``withColumn`` apila un ``Project`` completo de N+1 campos sobre el
    anterior, de modo que (a) la fila entera se re-materializa en cada
    uno de los N niveles (tráfico de memoria O(N²) por fila), (b) el plan
    crece en O(N²) nodos (análisis/optimización/serialización caros; en
    Spark Connect el plan se envía por RPC en cada acción) y (c) cada
    nivel apilado impide que la proyección entera se codegen en un solo
    stage.
    """
    if not schema.ready():
        raise ValueError("El esquema no está resuelto (llamar resolve_specs antes)")
    cols = list(schema.features) + [schema.target]
    out = df
    if config.missing == "drop":
        notnull = F.lit(True)
        for spec in cols:
            notnull = notnull & F.col(spec.name).isNotNull()
        out = out.filter(notnull)
    # Un único Project con todas las columnas mapeadas (conserva el orden
    # features + target y los nombres originales vía alias).
    exprs = []
    for spec in cols:
        col = F.col(spec.name)
        if spec.kind == "numeric":
            expr = _numeric_expr(col, spec, config)
        else:
            expr = _categorical_expr(col, spec, config)
        exprs.append(expr.cast("int").alias(spec.name))
    return out.select(*exprs)


def prepare(df: DataFrame, config) -> PreparedData:
    """Etapa 0 completa: pase 0a + resolución en driver + mapeo 0b.

    Devuelve ``PreparedData`` con ``df_prep`` persistido (MEMORY_AND_DISK).
    """
    if config.estimator != "histogram":
        raise ValueError(
            "prepare() es para el modo 'histogram'; el modo 'ksg' tiene su "
            "propio camino (ver sparkmim.info.ksg)"
        )
    # La Task se resuelve una vez aquí (regla compartida, schema.resolve_task)
    # y se lleva en el PreparedData y en el SelectorModel.
    task = resolve_task(df, config.target, config.task)
    schema = build_schema(df, config, task)
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
    return PreparedData(df_prep=df_prep, schema=schema, n_rows=n_rows, task=task)
