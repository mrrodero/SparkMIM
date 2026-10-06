"""Selección greedy detrás del oráculo de información (etapa 3).

Un único bucle greedy para ambos estimadores (histograma, KSG): solo consulta
la interfaz ``InformationOracle`` en **posiciones de candidata** (0..K-1).
Esto elimina por construcción la clase de bugs de mezclar índices globales de
feature con posiciones de candidata (ADR-0001).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

from .criteria import CRITERIA, criterion_score

if TYPE_CHECKING:
    from .oracles import InformationOracle

__all__ = ["SelectionResult", "greedy_select"]


@dataclass
class SelectionResult:
    """Resultado de la selección greedy.

    Attributes:
        selected: posiciones de candidata (0..K-1) en orden de selección.
        scores: puntaje del criterio en el momento de la selección (nats).
    """

    selected: List[int]
    scores: List[float]


def greedy_select(
    oracle: "InformationOracle",
    criterion: str,
    cmim_approx: str,
    min_score: float,
    max_features: int,
    cmim_m: int,
) -> SelectionResult:
    """Bucle greedy único sobre el oráculo.

    Args:
        oracle: oráculo de información (posiciones de candidata 0..K-1).
        criterion: uno de ``CRITERIA``.
        cmim_approx: ``"max_min"`` → cmim se aproxima como jmim (criterios
            rápidos sobre el oráculo); cualquier otro valor → pase CMIM por
            ronda (``cmi_set_all``).
        min_score: umbral de parada (nats).
        max_features: tope de features a seleccionar.
        cmim_m: tamaño de ``S_m`` (top-m por MI univariante, fijo).

    Returns:
        ``SelectionResult`` con posiciones y puntajes.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion debe ser uno de {CRITERIA}")

    mi = oracle.mi_all()
    K = len(mi)

    use_cmim_pass = criterion == "cmim" and cmim_approx != "max_min"
    crit = "jmim" if (criterion == "cmim" and not use_cmim_pass) else criterion
    m = min(cmim_m, K)
    s_m = [int(i) for i in np.argsort(-mi, kind="stable")[:m]] if use_cmim_pass else []

    selected: List[int] = []
    scores: List[float] = []

    while len(selected) < max_features and len(selected) < K:
        if not selected:
            # Ronda 1: argmax MI univariante.
            best = int(np.argmax(mi))
            best_score = float(mi[best])
        elif use_cmim_pass:
            # CMIM: CMI(X_i; Y | S_m \ {i}) para todas las candidatas (un pase).
            cmis = oracle.cmi_set_all(s_m)
            best = -1
            best_score = -np.inf
            for x in range(K):
                if x in selected:
                    continue
                if cmis[x] > best_score:
                    best_score = float(cmis[x])
                    best = x
        else:
            # Criterios rápidos sobre el oráculo.
            best = -1
            best_score = -np.inf
            for x in range(K):
                if x in selected:
                    continue
                score = criterion_score(x, selected, oracle, mi, crit)
                if score > best_score:
                    best_score = score
                    best = x
        if best_score < min_score:
            break
        selected.append(best)
        scores.append(float(best_score))

    return SelectionResult(selected=selected, scores=scores)
