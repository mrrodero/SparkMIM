"""Generador de datos sintéticos con estructura plantada (Hito 7).

Genera ``n`` filas y ``N`` features:
- ``n_informative`` features informativas (determinan el target).
- ``n_redundant`` features redundantes (copias ruidosas de las informativas).
- El resto: ruido independiente.

Target: clasificación binaria o regresión.

Uso:
    from synthetic import generate
    data = generate(n=10**5, n_features=200, n_informative=10, n_redundant=5)
    # data: {nombre: array(n,)} incluyendo "y".
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["generate"]


def generate(
    n: int,
    n_features: int,
    n_informative: int = 10,
    n_redundant: int = 5,
    seed: int = 0,
    task: str = "classification",
) -> Dict[str, np.ndarray]:
    """Genera datos sintéticos con estructura plantada.

    Args:
        n: nº de filas.
        n_features: nº total de features (N).
        n_informative: nº de features informativas.
        n_redundant: nº de features redundantes (copias de las informativas).
        seed: semilla.
        task: ``"classification"`` (binaria) o ``"regression"``.

    Returns:
        dict ``{nombre: array(n,)}`` con las features ``x0..``, ``r0..``,
        ``n0..`` y el target ``y``.
    """
    if n_informative + n_redundant > n_features:
        raise ValueError(
            f"n_informative + n_redundant ({n_informative + n_redundant}) "
            f"> n_features ({n_features})"
        )
    n_noise = n_features - n_informative - n_redundant
    rng = np.random.default_rng(seed)

    data: Dict[str, np.ndarray] = {}
    # Informativas.
    for i in range(n_informative):
        data[f"x{i}"] = rng.normal(size=n)
    # Redundantes (copias ruidosas de las informativas).
    for i in range(n_redundant):
        src = i % n_informative
        data[f"r{i}"] = data[f"x{src}"] + 0.01 * rng.normal(size=n)
    # Ruido independiente.
    for i in range(n_noise):
        data[f"n{i}"] = rng.normal(size=n)

    # Target: función de las informativas + ruido.
    informative = np.stack([data[f"x{i}"] for i in range(n_informative)], axis=1)
    score = informative.sum(axis=1)
    if task == "classification":
        y = (score + 0.5 * rng.normal(size=n) > 0).astype(int)
    elif task == "regression":
        y = score + 0.1 * rng.normal(size=n)
    else:
        raise ValueError(f"task debe ser 'classification' o 'regression' (recibido {task!r})")
    data["y"] = y
    return data
