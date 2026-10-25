"""Generador de datos sintéticos del benchmark (autónomo).

Versión autónoma (sin dependencia de ``tests/planted.py``) de la estructura
clásica de benchmark, empaquetada en el wheel para poder usarse en
Databricks sin clonar el repo ni manipular ``sys.path``.

Genera ``n`` filas y ``N`` features:
- ``n_informative`` features informativas (determinan el target).
- ``n_redundant`` features redundantes (copias ruidosas de las informativas).
- El resto: ruido independiente.

Target: clasificación binaria o regresión.

**Paridad bit a bit**: el orden de extracción del RNG es idéntico al
generador compartido ``tests/planted.py`` (``make_planted``) con la misma
estructura: primero las features en orden (``x0..``, ``r0..``, ``n0..``) y
luego el ruido del target. Mismos parámetros y semilla producen exactamente
los mismos valores que la versión local; se comprueba en
``tests/test_benchmark_synthetic.py``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["generate"]

_COPY_NOISE = 0.01
_CLASSIFICATION_NOISE = 0.5
_REGRESSION_NOISE = 0.1
_THRESHOLD = 0.0


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
        ``n0..`` y el target ``y`` (int en clasificación, float en
        regresión).
    """
    if n <= 0:
        raise ValueError("n debe ser positivo")
    if n_informative + n_redundant > n_features:
        raise ValueError(
            f"n_informative + n_redundant ({n_informative + n_redundant}) "
            f"> n_features ({n_features})"
        )
    if task not in ("classification", "regression"):
        raise ValueError(f"task debe ser 'classification' o 'regression' (recibido {task!r})")

    rng = np.random.default_rng(seed)
    data: Dict[str, np.ndarray] = {}

    # Features en el mismo orden que la estructura compartida: x0..
    # (informativas), r0.. (copias ruidosas) y n0.. (ruido independiente).
    for i in range(n_informative):
        data[f"x{i}"] = rng.normal(size=n)
    for i in range(n_redundant):
        base = data[f"x{i % n_informative}"]
        data[f"r{i}"] = base + _COPY_NOISE * rng.normal(size=n)
    n_noise = n_features - n_informative - n_redundant
    for i in range(n_noise):
        data[f"n{i}"] = rng.normal(size=n)

    # Target lineal: suma de las informativas + ruido; la clasificación es
    # el umbral en 0.
    score = sum(data[f"x{i}"] for i in range(n_informative))
    if task == "classification":
        score = score + _CLASSIFICATION_NOISE * rng.normal(size=n)
        data["y"] = (score > _THRESHOLD).astype(int)
    else:
        score = score + _REGRESSION_NOISE * rng.normal(size=n)
        data["y"] = score
    return data
