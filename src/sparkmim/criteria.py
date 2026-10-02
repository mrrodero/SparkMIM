"""Criterios informacionales greedy (Hito 4, etapa 3).

Criterios rápidos (aritmética pura sobre el oráculo de información, sin
acceso a la estructura de datos subyacente):

- ``mrmr``: ``MI(X;Y) − (1/|S|)·Σ_{Xi∈S} MI(X;Xi)``   [``mi_pair``]
- ``mim``:  ``MI(X;Y) − Σ_{Xi∈S} MI(X;Xi)``             [``mi_pair``]
- ``jmi``:  ``Σ_{Xi∈S} CMI(X;Y|Xi)``                     [``cmi_single``]
- ``jmim``: ``min_{Xi∈S} CMI(X;Y|Xi)``                   [``cmi_single``] (default)

``cmim`` es el único que requiere un pase ``mapInPandas`` extra por ronda;
vive en ``tables.py`` (``cmim_scores``). Aproximación documentada (CMIM exacto
inabordable → acotado por diseño).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .oracles import InformationOracle

__all__ = ["CRITERIA", "criterion_score"]

CRITERIA = ("mrmr", "mim", "jmi", "jmim", "cmim")


def criterion_score(
    x: int,
    S: Sequence[int],
    oracle: "InformationOracle",
    mi_xy: np.ndarray,
    criterion: str,
) -> float:
    """Puntaje del criterio rápido para la candidata ``x`` dado el conjunto ``S``.

    Aritmética pura sobre el oráculo de información: no toca la estructura
    de datos subyacente (tablas conjuntas, arrays crudos).

    Args:
        x: índice de la candidata (0..K-1), ``x ∉ S``.
        S: lista de índices ya seleccionados (0..K-1).
        oracle: oráculo de información (``InformationOracle``).
        mi_xy: array ``MI(X_i; Y)`` por candidata.
        criterion: ``"mrmr" | "mim" | "jmi" | "jmim"``.

    Returns:
        Puntaje en nats. Con ``S`` vacío (ronda 1) devuelve ``MI(X;Y)``.
    """
    if criterion == "mrmr":
        if not S:
            return float(mi_xy[x])
        redundancy = float(np.mean([oracle.mi_pair(x, s) for s in S]))
        return float(mi_xy[x] - redundancy)
    if criterion == "mim":
        if not S:
            return float(mi_xy[x])
        redundancy = float(sum(oracle.mi_pair(x, s) for s in S))
        return float(mi_xy[x] - redundancy)
    if criterion == "jmi":
        if not S:
            return float(mi_xy[x])
        return float(sum(oracle.cmi_single(x, s) for s in S))
    if criterion == "jmim":
        if not S:
            return float(mi_xy[x])
        return float(min(oracle.cmi_single(x, s) for s in S))
    raise ValueError(f"criterio no soportado por criterion_score: {criterion!r}")
