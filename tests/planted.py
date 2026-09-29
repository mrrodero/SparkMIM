"""Generador compartido de estructuras plantadas (tests y benchmarks).

Una "estructura plantada" declara, para cada feature, su rol respecto al
target y cómo se genera el target:

Roles (``Feature.role``):

- ``"informative"``: el target se genera a partir de las informativas.
- ``"independent"``: ruido independiente del target.
- ``"redundant"``: copia de otra feature (exacta si ``copy_noise = 0``; con
  ruido gaussiano aditivo si ``copy_noise > 0``).
- ``"correlated"``: copia ruidosa del target (coincide con él con
  probabilidad ``agreement``); solo con target ``"flip"``.

Generación del target (``Target.kind``):

- ``"and"``: ``y = x0 & x1 & ...`` (binaria); cada valor se invierte con
  probabilidad ``noise``.
- ``"linear"``: ``y = suma(informativas) + noise * N(0, 1)``; si
  ``thresholds`` no está vacío, ``y = sum(int(y > t) for t in thresholds)``
  (clasificación con ``len(thresholds) + 1`` clases).
- ``"flip"``: ``y`` bernoulli independiente; las features correladas lo
  copian.

El orden de extracción del RNG es determinista y documentado:

- ``flip``: primero el target, luego las features en orden.
- ``and``/``linear``: primero las features en orden, luego el ruido del
  target (si ``noise > 0``).

Esto permite que cada escenario de test reproduzca *bit a bit* los datos de
los generadores inline originales (misma semilla → mismos datos).

Uso::

    from planted import Feature, Target, make_planted, to_spark_df

    data = make_planted(
        n=4000,
        seed=0,
        features=[
            Feature("x0", "informative", kind="bernoulli"),
            Feature("x1", "informative", kind="bernoulli"),
            Feature("x2", "independent", kind="bernoulli"),
            Feature("x3", "redundant", copy_of="x0"),
        ],
        target=Target(kind="and", noise=0.1),
    )
    df = to_spark_df(spark, data, as_strings=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

__all__ = ["Feature", "Target", "make_planted", "to_spark_df"]

_ROLES = ("informative", "independent", "redundant", "correlated")
_KINDS = ("normal", "bernoulli")
_TARGET_KINDS = ("and", "linear", "flip")
_TARGET_NAME = "y"


@dataclass(frozen=True)
class Feature:
    """Rol de una feature dentro de una estructura plantada.

    - ``kind``: distribución de la feature (``"normal"`` o ``"bernoulli"``);
      aplica a informativas e independientes.
    - ``copy_of``: nombre de la feature copiada por una ``"redundant"``
      (debe aparecer antes en la lista).
    - ``copy_noise``: desviación típica del ruido aditivo de la copia
      (``0`` = copia exacta, sin extracción del RNG).
    - ``agreement``: probabilidad de que una ``"correlated"`` coincida con
      el target.
    """

    name: str
    role: str
    kind: str = "normal"
    copy_of: str | None = None
    copy_noise: float = 0.0
    agreement: float | None = None


@dataclass(frozen=True)
class Target:
    """Cómo se genera el target.

    - ``"and"``: AND lógico de las informativas; ``noise`` es la
      probabilidad de invertir cada valor.
    - ``"linear"``: suma de las informativas + ``noise * N(0, 1)``;
      ``thresholds`` lo convierte en clasificación (una clase por umbral).
    - ``"flip"``: target bernoulli independiente.
    """

    kind: str
    noise: float = 0.0
    thresholds: tuple[float, ...] = ()


def _draw(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    if kind == "bernoulli":
        return rng.integers(0, 2, size=n)
    if kind == "normal":
        return rng.normal(size=n)
    raise ValueError(f"tipo de feature desconocido: {kind!r}")


def _validate(features: Sequence[Feature], target: Target) -> None:
    """Valida la estructura plantada (errores con mensaje claro)."""
    names = [f.name for f in features]
    if len(set(names)) != len(names):
        raise ValueError("nombres de feature duplicados")
    if _TARGET_NAME in names:
        raise ValueError(f"el nombre {_TARGET_NAME!r} está reservado para el target")
    if target.kind not in _TARGET_KINDS:
        raise ValueError(f"tipo de target desconocido: {target.kind!r}")
    if target.noise < 0:
        raise ValueError("target.noise debe ser >= 0")
    informative = [f for f in features if f.role == "informative"]
    if target.kind in ("and", "linear") and not informative:
        raise ValueError(
            f"el target {target.kind!r} requiere al menos una feature informativa"
        )
    if target.kind == "and" and any(f.kind != "bernoulli" for f in informative):
        raise ValueError("el target 'and' requiere features informativas bernoulli")
    if target.kind == "flip" and informative:
        raise ValueError("el target 'flip' no admite features informativas")
    for f in features:
        if f.role not in _ROLES:
            raise ValueError(f"rol desconocido: {f.role!r}")
        if f.kind not in _KINDS:
            raise ValueError(f"tipo desconocido: {f.kind!r}")
        if f.role == "redundant":
            if f.copy_of is None or f.copy_of not in names:
                raise ValueError(
                    f"la feature redundante {f.name!r} necesita copy_of existente"
                )
            if names.index(f.copy_of) >= names.index(f.name):
                raise ValueError(
                    f"la feature redundante {f.name!r} debe copiar a una feature anterior"
                )
            if f.copy_noise < 0:
                raise ValueError(
                    f"la feature redundante {f.name!r} necesita copy_noise >= 0"
                )
        if f.role == "correlated":
            if target.kind != "flip":
                raise ValueError(
                    f"la feature correlada {f.name!r} requiere un target 'flip'"
                )
            if f.agreement is None or not (0.0 < f.agreement < 1.0):
                raise ValueError(
                    f"la feature correlada {f.name!r} necesita agreement en (0, 1)"
                )


def make_planted(
    n: int,
    seed: int,
    features: Sequence[Feature],
    target: Target,
) -> Dict[str, np.ndarray]:
    """Genera una estructura plantada como dict de arrays numpy.

    El dict contiene las features (en el orden dado) y el target ``"y"``.
    El orden de extracción del RNG es determinista (ver docstring del
    módulo): ``flip`` extrae primero el target; ``and``/``linear`` extraen
    primero las features y luego el ruido del target.
    """
    if n <= 0:
        raise ValueError("n debe ser positivo")
    _validate(features, target)
    rng = np.random.default_rng(seed)
    data: Dict[str, np.ndarray] = {}

    def _redundant(f: Feature) -> np.ndarray:
        base = data[f.copy_of]
        if f.copy_noise > 0:
            return base + f.copy_noise * rng.normal(size=n)
        return base

    if target.kind == "flip":
        data[_TARGET_NAME] = rng.integers(0, 2, size=n)
        for f in features:
            if f.role == "correlated":
                mask = rng.random(n) < f.agreement
                data[f.name] = np.where(mask, data[_TARGET_NAME], 1 - data[_TARGET_NAME])
            elif f.role == "redundant":
                data[f.name] = _redundant(f)
            else:
                data[f.name] = _draw(rng, n, f.kind)
    else:
        for f in features:
            if f.role == "redundant":
                data[f.name] = _redundant(f)
            elif f.role in ("informative", "independent"):
                data[f.name] = _draw(rng, n, f.kind)
            else:  # "correlated" ya descartada por _validate aquí.
                raise ValueError(f"rol {f.role!r} no válido con el target {target.kind!r}")
        informative = [data[f.name] for f in features if f.role == "informative"]
        if target.kind == "and":
            y = np.logical_and.reduce(informative).astype(int)
            if target.noise > 0:
                mask = rng.random(n) < target.noise
                y = np.where(mask, 1 - y, y)
            data[_TARGET_NAME] = y
        else:  # "linear"
            score = sum(informative)
            if target.noise > 0:
                score = score + target.noise * rng.normal(size=n)
            if target.thresholds:
                data[_TARGET_NAME] = sum((score > t).astype(int) for t in target.thresholds)
            else:
                data[_TARGET_NAME] = score
    return data


def to_spark_df(
    spark,
    data: Dict[str, np.ndarray],
    target: str = _TARGET_NAME,
    as_strings: bool = False,
):
    """Construye un DataFrame de Spark a partir de los arrays de ``make_planted``.

    Orden de columnas: las features (en el orden que aparecen en ``data``)
    seguidas del target. Spark infiere los tipos desde los escalares Python
    (int → long, float → double, str → string). Con ``as_strings`` todos los
    valores se convierten a cadenas (columnas categóricas).
    """
    cols = [c for c in data if c != target]
    arrays = [data[c] for c in cols] + [data[target]]
    if as_strings:
        rows = [tuple(str(v) for v in row) for row in zip(*arrays)]
    else:
        rows = [tuple(v.item() for v in row) for row in zip(*arrays)]
    return spark.createDataFrame(rows, cols + [target])
