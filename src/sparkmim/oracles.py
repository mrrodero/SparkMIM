"""Oráculos de información: la interfaz por la que el greedy consulta MI/CMI.

El bucle greedy (``selection.greedy_select``) solo ve esta interfaz y nunca
toca la estructura de datos subyacente (tablas conjuntas, arrays crudos).
Todos los índices son **posiciones de candidata** (0..K-1), nunca índices
globales de feature.

Dos adaptadores (docs/DESIGN.md §2, ADR-0001):

- ``HistogramOracle``: MI/CMI desde las tablas conjuntas (estimator
  histograma). ``mi_all``/``mi_pair``/``cmi_single`` responden O(1) desde el
  caché del driver; ``cmi_set_all`` ejecuta el pase CMIM distribuido por
  ronda (``cmim_scores``) sobre el subsample.
- ``KsgOracle``: MI/CMI por kNN (KSG) sobre el bloque de candidatas en el
  driver; todas las consultas son numpy puro, con caché por consulta.
"""

from __future__ import annotations

from typing import Dict, Protocol, Sequence, Tuple

import numpy as np

from .info.entropy import conditional_mi, mutual_information
from .info.ksg import ksg_cmi, ksg_mi
from .tables import TableCache, cmim_scores

__all__ = ["InformationOracle", "HistogramOracle", "KsgOracle"]


class InformationOracle(Protocol):
    """Interfaz de consulta de MI/CMI en posiciones de candidata (0..K-1).

    El bucle greedy solo usa estas cuatro consultas. Cada adaptador decide
    su coste: el histograma responde desde el caché de tablas conjuntas;
    KSG responde con kNN sobre el subsample en el driver.
    """

    def mi_all(self) -> np.ndarray:
        """MI(X_i; Y) para todas las candidatas (array de longitud K)."""
        ...

    def mi_pair(self, i: int, j: int) -> float:
        """MI(X_i; X_j) (simétrica)."""
        ...

    def cmi_single(self, i: int, j: int) -> float:
        """CMI(X_i; Y | X_j)."""
        ...

    def cmi_set_all(self, s_m: Sequence[int]) -> np.ndarray:
        """CMI(X_i; Y | S_m \\ {i}) para todas las candidatas en un pase."""
        ...


class HistogramOracle:
    """Oráculo sobre tablas conjuntas (estimator histograma).

    ``mi_all``/``mi_pair``/``cmi_single`` responden desde ``cache``
    (O(1) por consulta, sin tocar Spark). ``cmi_set_all`` es el único método
    que toca Spark: ejecuta el pase CMIM distribuido (``cmim_scores``) sobre
    el subsample ``sub`` con ``S_m`` = top-m por MI univariante.
    """

    def __init__(
        self,
        cache: TableCache,
        sub,
        candidate_cols: Sequence[str],
        target_col: str,
        candidate_n_codes: Sequence[int],
        n_y: int,
    ) -> None:
        self._cache = cache
        self._sub = sub
        self._candidate_cols = list(candidate_cols)
        self._target_col = target_col
        self._candidate_n_codes = list(candidate_n_codes)
        self._n_y = int(n_y)
        self._mi_all: np.ndarray | None = None

    def mi_all(self) -> np.ndarray:
        if self._mi_all is None:
            self._mi_all = np.array(
                [mutual_information(self._cache.uni(i)) for i in range(len(self._candidate_cols))],
                dtype=np.float64,
            )
        return self._mi_all

    def mi_pair(self, i: int, j: int) -> float:
        # Simétrica: la orientación del par canónico no importa.
        return float(mutual_information(self._cache.pair(i, j)))

    def cmi_single(self, i: int, j: int) -> float:
        # La caché entrega la triple en la orientación (n_i, n_y, n_j) que
        # espera conditional_mi; canonicidad y ejes viven en TableCache.
        return float(conditional_mi(self._cache.cmi_table(i, j)))

    def cmi_set_all(self, s_m: Sequence[int]) -> np.ndarray:
        return cmim_scores(
            self._sub,
            self._candidate_cols,
            self._target_col,
            self._candidate_n_codes,
            self._n_y,
            list(s_m),
        )


class KsgOracle:
    """Oráculo por kNN (KSG) sobre el bloque de candidatas en el driver.

    ``X`` es el bloque de candidatas (n_sub × K, en orden de candidata) y
    ``y`` la columna objetivo; todas las consultas son numpy puro con caché
    por consulta (``mi`` por candidata, ``pair`` por par canónico, ``cmi``
    por par orientado y ``cmi_set`` por (candidata, condicionantes)).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, k: int = 10) -> None:
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._k = int(k)
        self._mi: Dict[int, float] = {}
        self._pair: Dict[Tuple[int, int], float] = {}
        self._cmi: Dict[Tuple[int, int], float] = {}
        self._cmi_set: Dict[Tuple[int, Tuple[int, ...]], float] = {}

    def _mi_at(self, i: int) -> float:
        if i not in self._mi:
            self._mi[i] = float(ksg_mi(self._X[:, i], self._y, self._k))
        return self._mi[i]

    def mi_all(self) -> np.ndarray:
        return np.array([self._mi_at(i) for i in range(self._X.shape[1])], dtype=np.float64)

    def mi_pair(self, i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in self._pair:
            self._pair[key] = float(ksg_mi(self._X[:, key[0]], self._X[:, key[1]], self._k))
        return self._pair[key]

    def cmi_single(self, i: int, j: int) -> float:
        key = (i, j)
        if key not in self._cmi:
            self._cmi[key] = float(ksg_cmi(self._X[:, i], self._y, self._X[:, j], self._k))
        return self._cmi[key]

    def cmi_set_all(self, s_m: Sequence[int]) -> np.ndarray:
        """CMI(X_i; Y | S_m \\ {i}) para todas las candidatas (numpy puro).

        Con ``m ≤ 2`` (por diseño), ``cond`` tiene 0, 1 o 2 elementos; el
        caso vacío lo cubre el guard de ``ksg_cmi`` (la identidad degenera en
        MI univariante), así que no hay ramificación por tamaño de ``cond``.
        """
        out = np.empty(self._X.shape[1], dtype=np.float64)
        for i in range(self._X.shape[1]):
            cond = tuple(s for s in s_m if s != i)
            key = (i, cond)
            if key not in self._cmi_set:
                self._cmi_set[key] = float(
                    ksg_cmi(self._X[:, i], self._y, self._X[:, list(cond)], self._k)
                )
            out[i] = self._cmi_set[key]
        return out
