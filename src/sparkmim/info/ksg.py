"""Estimador KSG (Kraskov, Stögbauer & Grassberger, 2004) para MI y CMI.

Modo opcional (``estimator="ksg"``) para variables continuas donde el binning
histórico no es aceptable. Usa ``cKDTree`` de scipy; el cómputo es en el driver
sobre un subsample (≤ ``ksg_subsample`` filas), documentado como trade-off
(único modo con kNN global en driver).

Identidad para CMI: ``CMI(X;Y|Z) = MI(XZ;Y) − MI(Z;Y)`` (|Z| acotado).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

__all__ = ["ksg_mi", "ksg_cmi"]


def _as_2d(a) -> np.ndarray:
    """Convierte a (n, d): 1D -> (n, 1), 2D invariante."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def ksg_mi(a, b, k: int = 10) -> float:
    """MI(X;Y) entre ``a`` (n×d_a) y ``b`` (n×d_b) por KSG.

    Estimador (Kraskov et al. 2004):
        I(X;Y) = (1/n)·Σ_i [ψ(k) + ln(n) − ψ(nx_i) − ψ(ny_i)]
    donde ``nx_i``/``ny_i`` son los recuentos de vecinos (incluido el propio
    punto) dentro de la esfera de radio ``ε_i`` (distancia al k-ésimo vecino en
    el espacio conjunto). Devuelve float en nats, acotado a 0 (la estimación
    puede dar negativos pequeños por ruido de muestreo).
    """
    a = _as_2d(a)
    b = _as_2d(b)
    n = a.shape[0]
    if n <= k + 1:
        return 0.0
    joint = np.hstack([a, b])
    tree_joint = cKDTree(joint)
    # Distancia al k-ésimo vecino en el espacio conjunto (el punto propio es el
    # vecino 0, por eso se pide k+1 y se toma el índice k).
    dist, _ = tree_joint.query(joint, k=k + 1)
    eps = dist[:, k]
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        nx[i] = len(tree_a.query_ball_point(a[i], eps[i]))
        ny[i] = len(tree_b.query_ball_point(b[i], eps[i]))
    mi = float(np.mean(digamma(k) + np.log(n) - digamma(nx) - digamma(ny)))
    return max(0.0, mi)


def ksg_cmi(x, y, z, k: int = 10) -> float:
    """CMI(X;Y|Z) = MI(XZ;Y) − MI(Z;Y), con ``z`` (n×d_z).

    Devuelve float en nats (puede ser ligeramente negativo por ruido; no se
    acota, ya que en el greedy un CMI ≤ 0 significa "no aporta información").
    """
    x = _as_2d(x)
    y = _as_2d(y)
    z = _as_2d(z)
    xz = np.hstack([x, z])
    return ksg_mi(xz, y, k) - ksg_mi(z, y, k)
