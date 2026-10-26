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

from typing import Dict, Iterator, Optional

import numpy as np

__all__ = ["generate", "generate_chunks"]

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
    _validate_params(n, n_features, n_informative, n_redundant, task)
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


def _validate_params(
    n: int, n_features: int, n_informative: int, n_redundant: int, task: str
) -> None:
    """Validación común de parámetros (misma semántica que ``generate``)."""
    if n <= 0:
        raise ValueError("n debe ser positivo")
    if n_informative + n_redundant > n_features:
        raise ValueError(
            f"n_informative + n_redundant ({n_informative + n_redundant}) "
            f"> n_features ({n_features})"
        )
    if task not in ("classification", "regression"):
        raise ValueError(
            f"task debe ser 'classification' o 'regression' (recibido {task!r})"
        )


def _feature_names(n_features: int, n_informative: int, n_redundant: int) -> list:
    """Nombres de features en el orden del generador: ``x0..``, ``r0..``, ``n0..``."""
    names = [f"x{i}" for i in range(n_informative)]
    names += [f"r{i}" for i in range(n_redundant)]
    names += [f"n{i}" for i in range(n_features - n_informative - n_redundant)]
    return names


def generate_chunks(
    n: int,
    n_features: int,
    n_informative: int = 10,
    n_redundant: int = 5,
    seed: int = 0,
    task: str = "classification",
    chunk_rows: Optional[int] = None,
) -> Iterator[Dict[str, np.ndarray]]:
    """Misma estructura plantada que :func:`generate`, producida en
    **chunks de filas** para acotar la memoria del driver a
    ``O(chunk_rows × N)`` en lugar de ``O(n × N)``.

    Cada chunk es un dict ``{nombre: array(chunk,)}`` con las mismas
    columnas y orden que :func:`generate` (features ``x0..``, ``r0..``,
    ``n0..`` y el target ``y``); al concatenar los chunks se obtienen las
    ``n`` filas completas.

    RNG
    ---
    Cada columna tiene su propio ``Generator`` derivado de un único
    ``SeedSequence(seed)`` (``spawn``: i-ésima columna → i-ésimo spawn,
    último spawn para el ruido del target). La secuencia de extracción
    *dentro de cada columna* es la de generarla completa, por lo que los
    valores **no dependen de ``chunk_rows``** (ver
    ``test_generate_chunks_invariant_to_chunk_size``). Los valores sí
    difieren de :func:`generate`, que usa un único stream compartido entre
    columnas: ambas funciones son deterministas y reproducibles, pero no
    son bit-idénticas entre sí.

    El target se calcula **por chunk** (suma de las informativas + ruido
    del target, umbral 0 en clasificación): la estructura plantada es la
    misma que en :func:`generate`.
    """
    _validate_params(n, n_features, n_informative, n_redundant, task)
    if chunk_rows is None or chunk_rows <= 0:
        chunk_rows = n

    names = _feature_names(n_features, n_informative, n_redundant)
    # Spawn determinista por columna, en el orden del generador, más uno
    # para el ruido del target.
    spawns = np.random.SeedSequence(seed).spawn(n_features + 1)
    rngs = [np.random.default_rng(s) for s in spawns]

    def _draw(i: int, m: int) -> np.ndarray:
        """``m`` valores de la columna ``i`` (stream por columna)."""
        return rngs[i].normal(size=m)

    for start in range(0, n, chunk_rows):
        m = min(chunk_rows, n - start)
        chunk: Dict[str, np.ndarray] = {}
        # Informativas: x0..
        for i in range(n_informative):
            chunk[f"x{i}"] = _draw(i, m)
        # Redundantes: copia ruidosa de la informativa base (chunk a chunk
        # sobre la misma base que en ``generate``).
        for i in range(n_redundant):
            base = chunk[f"x{i % n_informative}"]
            chunk[f"r{i}"] = base + _COPY_NOISE * _draw(n_informative + i, m)
        # Ruido independiente: n0..
        n_noise = n_features - n_informative - n_redundant
        for i in range(n_noise):
            chunk[f"n{i}"] = _draw(n_informative + n_redundant + i, m)
        # Target por chunk (misma fórmula que ``generate``).
        score = sum(chunk[f"x{i}"] for i in range(n_informative))
        if task == "classification":
            score = score + _CLASSIFICATION_NOISE * rngs[n_features].normal(size=m)
            chunk["y"] = (score > _THRESHOLD).astype(int)
        else:
            score = score + _REGRESSION_NOISE * rngs[n_features].normal(size=m)
            chunk["y"] = score
        yield chunk
