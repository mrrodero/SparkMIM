"""Estimadores de entropía, información mutua y CMI desde tablas de conteo.

Todas las funciones operan sobre tablas de conteos enteras (int64) y devuelven
valores en **nats** (log natural), vectorizadas con numpy.

Convenciones numéricas:
- ``0 · log 0 = 0`` (los celdas vacías no contribuyen).
- CMI puede salir ligeramente negativa por error de coma flotante (~1e-16);
  no se recorta aquí para no ocultar errores de entrada.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "entropy_from_counts",
    "joint_entropy_from_counts",
    "mutual_information",
    "conditional_mi",
    "conditional_mi_multi",
]


def _log_nonzero(p: np.ndarray) -> np.ndarray:
    """log(p) con -inf donde p == 0 (sin warnings)."""
    out = np.full(p.shape, -np.inf, dtype=np.float64)
    np.log(p, out=out, where=p > 0)
    return out


def entropy_from_counts(counts: np.ndarray) -> float:
    """Entropía ``H(X) = -Σ p(x) log p(x)`` desde un vector de conteos 1D.

    Args:
        counts: conteos no negativos (cualquier forma; se aplana).

    Returns:
        Entropía en nats. 0.0 si la tabla está vacía.
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    log_p = _log_nonzero(p)
    terms = p * log_p
    # 0 · log 0 = 0: se descartan los términos no finitos (p == 0).
    return float(-np.where(np.isfinite(terms), terms, 0.0).sum())


def joint_entropy_from_counts(table: np.ndarray) -> float:
    """Entropía conjunta ``H(X1, ..., Xd)`` desde una tabla de conteos de d dims.

    Args:
        table: tabla de conteos con una dimensión por variable.

    Returns:
        Entropía conjunta en nats.
    """
    return entropy_from_counts(table)


def mutual_information(table_xy: np.ndarray) -> float:
    """Información mutua ``MI(X;Y) = Σ p(x,y) log [p(x,y) / (p(x)p(y))]``.

    Equivalente a ``H(X) + H(Y) - H(X,Y)``; se calcula directamente de la
    tabla conjunta por estabilidad numérica.

    Args:
        table_xy: tabla de conteos conjunta (n_x, n_y).

    Returns:
        MI en nats (≥ 0 salvo error numérico).
    """
    table_xy = np.asarray(table_xy, dtype=np.float64)
    if table_xy.ndim != 2:
        raise ValueError(f"table_xy debe ser 2D, tiene {table_xy.ndim} dims")
    n = table_xy.sum()
    if n == 0:
        return 0.0
    pxy = table_xy / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    terms = pxy * (_log_nonzero(pxy) - _log_nonzero(px)[:, None] - _log_nonzero(py)[None, :])
    return float(np.where(np.isfinite(terms), terms, 0.0).sum())


def conditional_mi(table_xyz: np.ndarray) -> float:
    """CMI ``CMI(X;Y|Z) = Σ p(x,y,z) log [p(x,y,z)p(z) / (p(x,z)p(y,z))]``.

    Args:
        table_xyz: tabla de conteos conjunta (n_x, n_y, n_z).

    Returns:
        CMI en nats (≥ 0 salvo error numérico).
    """
    table_xyz = np.asarray(table_xyz, dtype=np.float64)
    if table_xyz.ndim != 3:
        raise ValueError(f"table_xyz debe ser 3D, tiene {table_xyz.ndim} dims")
    n = table_xyz.sum()
    if n == 0:
        return 0.0
    pxyz = table_xyz / n
    pz = pxyz.sum(axis=(0, 1))
    pxz = pxyz.sum(axis=1)
    pyz = pxyz.sum(axis=0)
    terms = pxyz * (
        _log_nonzero(pxyz)
        + _log_nonzero(pz)[None, None, :]
        - _log_nonzero(pxz)[:, None, :]
        - _log_nonzero(pyz)[None, :, :]
    )
    return float(np.where(np.isfinite(terms), terms, 0.0).sum())


def conditional_mi_multi(table: np.ndarray) -> float:
    """CMI con varios condicionantes: ``CMI(X;Y|Z1,...,Zk)``.

    Args:
        table: tabla de conteos con ejes ``(X, Y, Z1, ..., Zk)`` (dim ≥ 3).
            Para ``k=1`` es equivalente a :func:`conditional_mi`.

    Returns:
        CMI en nats (≥ 0 salvo error numérico).
    """
    table = np.asarray(table, dtype=np.float64)
    if table.ndim < 3:
        raise ValueError(f"table debe tener ≥ 3 dims, tiene {table.ndim}")
    n = table.sum()
    if n == 0:
        return 0.0
    p = table / n
    pz = p.sum(axis=(0, 1))   # (Z1, ..., Zk)
    pxz = p.sum(axis=1)       # (X, Z1, ..., Zk)
    pyz = p.sum(axis=0)       # (Y, Z1, ..., Zk)
    # Broadcasting: pz gana 2 ejes al frente (X, Y); pxz gana 1 eje tras X;
    # pyz gana 1 eje al frente (X). Idéntico al caso 3D (k=1).
    terms = p * (
        _log_nonzero(p)
        + _log_nonzero(pz)[None, None]
        - _log_nonzero(pxz)[:, None]
        - _log_nonzero(pyz)[None, :]
    )
    return float(np.where(np.isfinite(terms), terms, 0.0).sum())
