"""Generador de datos sintéticos con estructura plantada (Hito 7).

Adaptador fino sobre el generador compartido ``tests/planted.py``: construye
la estructura clásica de benchmark y delega la generación, de modo que tests
y benchmarks comparten una única implementación de la estructura plantada.

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

import sys
from pathlib import Path
from typing import Dict

import numpy as np

# El generador compartido vive en tests/ (sin paquete instalado).
_TESTS = str(Path(__file__).resolve().parent.parent / "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from planted import Feature, Target, make_planted  # noqa: E402

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
    if task not in ("classification", "regression"):
        raise ValueError(f"task debe ser 'classification' o 'regression' (recibido {task!r})")
    features = [Feature(f"x{i}", "informative") for i in range(n_informative)]
    features += [
        Feature(f"r{i}", "redundant", copy_of=f"x{i % n_informative}", copy_noise=0.01)
        for i in range(n_redundant)
    ]
    n_noise = n_features - n_informative - n_redundant
    features += [Feature(f"n{i}", "independent") for i in range(n_noise)]
    if task == "classification":
        target = Target(kind="linear", noise=0.5, thresholds=(0.0,))
    else:
        target = Target(kind="linear", noise=0.1)
    return make_planted(n, seed, features, target)
