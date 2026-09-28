"""Etapa 1 — Screening univariante (un pase sobre ``df_prep``).

Flujo (docs/DESIGN.md §2, Etapa 1):

1. ``build_screening_tables`` (UN ``mapInPandas``) + ``dense_from_screening``
   → N tablas densas ``(n_x, n_y)`` en el driver.
2. Driver: ``MI(X_i; Y)`` por feature (nats).
3. **Significancia** a través de la costura ``SignificanceTest``
   (``significance.py``): adaptadores χ² / permutación / none + control FDR.
4. Top ``screen_top_k`` por MI entre las significativas → conjunto C de
   candidatas (|C| ≤ K).

Este módulo solo compone: tablas → MI → significancia → top-K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from pyspark.sql import DataFrame

from .config import SelectorConfig
from .info.entropy import mutual_information
from .schema import Schema
from .significance import SignificanceInput, make_significance_test
from .tables import build_screening_tables, dense_from_screening

__all__ = ["ScreenResult", "screen", "select_candidates"]


@dataclass
class ScreenResult:
    """Resultado de la etapa 1 (screening).

    Attributes:
        mi: ``MI(X_i; Y)`` por feature (nats), longitud N.
        pvalues: p-valor de significancia por feature (longitud N).
        significant: máscara booleana FDR por feature (longitud N).
        candidates: índices de las features candidatas (top-K por MI entre
            las significativas), |C| ≤ ``screen_top_k``.
        tables: tablas de screening densas ``(n_x, n_y)`` por feature (fid);
            se reutilizan en la etapa 2 para no recomputarlas.
        n_rows: nº de filas del df preprocesado (suma de la tabla de la
            feature 0).
    """

    mi: np.ndarray
    pvalues: np.ndarray
    significant: np.ndarray
    candidates: List[int]
    tables: Dict[int, np.ndarray] = field(default_factory=dict)
    n_rows: int = 0


def select_candidates(
    mi: np.ndarray, significant: np.ndarray, k: int
) -> List[int]:
    """Top-``k`` por MI entre las features significativas (estable en empates).

    La política de corte de la etapa 1; pura (sin Spark).

    Args:
        mi: ``MI(X_i; Y)`` por feature (nats), longitud N.
        significant: máscara booleana FDR por feature (longitud N).
        k: tamaño máximo del conjunto de candidatas (``screen_top_k``).

    Returns:
        Índices de las candidatas, en orden descendente de MI.
    """
    if not significant.any():
        return []
    sig_idx = np.where(significant)[0]
    order = sig_idx[np.argsort(-mi[sig_idx], kind="stable")]
    return [int(i) for i in order[:k]]


def screen(
    df_prep: DataFrame, schema: Schema, config: SelectorConfig
) -> ScreenResult:
    """Ejecuta la etapa 1 (screening) y devuelve el conjunto de candidatas.

    Args:
        df_prep: DataFrame preprocesado (códigos enteros: features + target).
        schema: ``Schema`` resuelto (``n_codes`` rellenados).
        config: ``SelectorConfig``.

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

    # 2. Significancia (costura: adaptadores χ² / permutación / none) + FDR.
    test = make_significance_test(config)
    sig = test.test(
        SignificanceInput(
            tables=tables,
            mi=mi,
            df_prep=df_prep,
            feature_cols=feature_cols,
            target_col=target_col,
            n_x=n_x,
            n_y=n_y,
        )
    )

    # 3. Top screen_top_k por MI entre las significativas.
    candidates = select_candidates(mi, sig.significant, config.screen_top_k)

    # Cada fila aporta a una celda de la tabla de la feature 0 => suma = nº de filas.
    n_rows = int(tables[0].sum()) if n_features > 0 else 0
    return ScreenResult(
        mi=mi,
        pvalues=sig.pvalues,
        significant=sig.significant,
        candidates=candidates,
        tables=tables,
        n_rows=n_rows,
    )
