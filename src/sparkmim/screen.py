"""Etapa 1 — Screening univariante (un pase sobre ``df_prep``) + significancia.

Flujo (PLAN_IMPLEMENTACION.md §6, Etapa 1):

1. ``build_screening_tables`` (UN ``mapInPandas``) + ``dense_from_screening``
   → N tablas densas ``(n_x, n_y)`` en el driver.
2. Driver: ``MI(X_i; Y)`` por feature (nats).
3. p-valor por feature:
   - ``"chi2"`` (por defecto): O(1), ``G² = 2·n·MI ~ χ²``.
   - ``"permutation"``: test de permutación distribuido (UN ``mapInPandas``
     sobre un subsample de ≤ ``permutation_rows`` filas; ``B = n_permutations``
     permutaciones de y por partición, semilla fija).
   - ``None``: sin filtro de significancia.
4. Control FDR de Benjamini-Hochberg (``fdr_q``).
5. Top ``screen_top_k`` por MI entre las significativas → conjunto C de
   candidatas (|C| ≤ K).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    IntegerType,
    StructField,
    StructType,
)

from .config import SelectorConfig
from .info.entropy import mutual_information
from .schema import Schema
from .significance import bh_fdr, chi2_pvalue, permutation_pvalue
from .tables import build_screening_tables, dense_from_screening

__all__ = ["ScreenResult", "screen"]

_PermSchema = StructType(
    [
        StructField("fid", IntegerType()),
        StructField("b", IntegerType()),
        StructField("table", BinaryType()),
    ]
)


@dataclass
class ScreenResult:
    """Resultado de la etapa 1 (screening).

    - ``mi``: ``MI(X_i; Y)`` por feature (nats), misma longitud que features.
    - ``pvalues``: p-valor por feature (1.0 si no se computó, p.e. ``None``).
    - ``significant``: máscara BH-FDR por feature (todas True si ``None``).
    - ``candidates``: índices de features candidatas (top-K por MI entre las
      significativas).
    - ``tables``: tablas densas ``(n_x, n_y)`` por feature (reutilizables).
    - ``n_rows``: filas de ``df_prep``.
    """

    mi: np.ndarray
    pvalues: np.ndarray
    significant: np.ndarray
    candidates: List[int]
    tables: Dict[int, np.ndarray] = field(default_factory=dict)
    n_rows: int = 0


def _permutation_pvalues(
    df_prep: DataFrame,
    candidate_cols: Sequence[str],
    target_col: str,
    n_x: Sequence[int],
    n_y: int,
    n_b: int,
    n_rows_sub: int,
    seed: int,
) -> np.ndarray:
    """Test de permutación distribuido: UN ``mapInPandas`` sobre el subsample.

    Por partición y por candidata ``i`` y por permutación ``b`` (0..B-1) se
    emite la tabla densa ``crosstab(X_i, y_perm_b)`` serializada; ``b = -1`` es
    el observado (sin permutar). En el driver se suman las particiones por
    ``(i, b)`` → MI observado y MI nulos → p-valor empírico.

    Las permutaciones son por partición (semilla ``seed + b``), una nula válida
    y estándar para el test distribuido.
    """
    k = len(candidate_cols)
    n_total = int(df_prep.count())
    if n_total <= n_rows_sub:
        sub = df_prep
    else:
        fraction = n_rows_sub / n_total
        sub = df_prep.sample(withReplacement=False, fraction=fraction, seed=seed)

    def _func(iterator):
        for pdf in iterator:
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

    rows_df = sub.mapInPandas(_func, schema=_PermSchema)

    # Acumular las tablas por (i, b) en el driver.
    acc_obs = [np.zeros((n_x[i], n_y), dtype=np.int64) for i in range(k)]
    acc_null = [
        [np.zeros((n_x[i], n_y), dtype=np.int64) for _ in range(n_b)] for i in range(k)
    ]
    for r in rows_df.collect():
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


def screen(df_prep: DataFrame, schema: Schema, config: SelectorConfig) -> ScreenResult:
    """Ejecuta la etapa 1 (screening) y devuelve el conjunto de candidatas.

    Args:
        df_prep: DataFrame preprocesado (códigos enteros: features + target).
        schema: ``Schema`` resuelto (``n_codes`` rellenados).
        config: ``SelectorConfig`` (``significance``, ``fdr_q``, ``screen_top_k``,
            ``n_permutations``, ``permutation_rows``, ``seed``).

    Returns:
        ``ScreenResult`` con MI, p-valores, máscara FDR, candidatas y tablas.
    """
    feature_cols = schema.feature_names()
    target_col = schema.target.name
    n_x = schema.n_codes()
    n_y = schema.target.n_codes
    n_features = len(feature_cols)

    # 1. Tablas de screening (un pase) + MI por feature.
    agg = build_screening_tables(df_prep, feature_cols, target_col, n_x, n_y)
    tables = dense_from_screening(agg, n_x, n_y)
    mi = np.array([mutual_information(tables[fid]) for fid in range(n_features)])

    # 2. p-valores + 3. FDR.
    if config.significance is None:
        pvalues = np.ones(n_features)
        significant = np.ones(n_features, dtype=bool)
    elif config.significance == "chi2":
        pvalues = np.array(
            [chi2_pvalue(tables[fid], mi[fid]) for fid in range(n_features)]
        )
        significant = bh_fdr(pvalues, config.fdr_q)
    elif config.significance == "permutation":
        # Pre-filtro: top-K por MI (el test de permutación es caro; solo se
        # aplica a las candidatas con más MI).
        k = min(config.screen_top_k, n_features)
        top_idx = np.argsort(-mi, kind="stable")[:k]
        pvalues = np.ones(n_features)
        significant = np.zeros(n_features, dtype=bool)
        if k > 0:
            cand_cols = [feature_cols[i] for i in top_idx]
            cand_nx = [n_x[i] for i in top_idx]
            pvals_k = _permutation_pvalues(
                df_prep,
                cand_cols,
                target_col,
                cand_nx,
                n_y,
                config.n_permutations,
                config.permutation_rows,
                config.seed,
            )
            sig_k = bh_fdr(pvals_k, config.fdr_q)
            for pos, i in enumerate(top_idx):
                pvalues[i] = pvals_k[pos]
                significant[i] = sig_k[pos]
    else:
        raise ValueError(
            f"significance debe ser 'chi2', 'permutation' o None, no {config.significance!r}"
        )

    # 4. Top screen_top_k por MI entre las significativas.
    if significant.any():
        sig_idx = np.where(significant)[0]
        order = sig_idx[np.argsort(-mi[sig_idx], kind="stable")]
        candidates = [int(i) for i in order[: config.screen_top_k]]
    else:
        candidates = []

    # Cada fila aporta a una celda de la tabla de la feature 0 => suma = nº de filas.
    n_rows = int(tables[0].sum()) if n_features > 0 else 0
    return ScreenResult(
        mi=mi,
        pvalues=pvalues,
        significant=significant,
        candidates=candidates,
        tables=tables,
        n_rows=n_rows,
    )
