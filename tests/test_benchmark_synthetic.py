"""Paridad exacta del generador autónomo (``sparkmim.benchmarks``) vs ``tests/planted.py``.

El wheel no puede importar ``tests/``, así que ``sparkmim.benchmarks.synthetic``
reimplementa la estructura clásica de benchmark; estas pruebas garantizan que
produzca **los mismos valores bit a bit** que la referencia compartida
(``make_planted``) para clasificación y regresión, y que conserve los mismos
errores de validación que el generador anterior.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from planted import Feature, Target, make_planted  # noqa: E402

from sparkmim.benchmarks import generate, generate_chunks


def _reference(n: int, n_features: int, n_informative: int, n_redundant: int,
               seed: int, task: str) -> dict:
    """La estructura clásica de benchmark a través del generador compartido."""
    features = [Feature(f"x{i}", "informative") for i in range(n_informative)]
    features += [
        Feature(f"r{i}", "redundant", copy_of=f"x{i % n_informative}", copy_noise=0.01)
        for i in range(n_redundant)
    ]
    features += [
        Feature(f"n{i}", "independent")
        for i in range(n_features - n_informative - n_redundant)
    ]
    if task == "classification":
        target = Target(kind="linear", noise=0.5, thresholds=(0.0,))
    else:
        target = Target(kind="linear", noise=0.1)
    return make_planted(n, seed, features, target)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_generate_matches_planted_bitwise(task, seed):
    n, nf, ni, nr = 3000, 40, 10, 5
    got = generate(n=n, n_features=nf, n_informative=ni, n_redundant=nr, seed=seed, task=task)
    ref = _reference(n, nf, ni, nr, seed, task)
    assert list(got) == list(ref)  # mismo orden de columnas
    for key in ref:
        assert got[key].dtype == ref[key].dtype, key
        # Iguala exacta (bit a bit), no aproximada: el orden del RNG debe
        # coincidir con el generador compartido.
        np.testing.assert_array_equal(got[key], ref[key])


def test_generate_classification_target_binary():
    got = generate(n=500, n_features=30, seed=1, task="classification")
    assert set(np.unique(got["y"])) <= {0, 1}


def test_generate_validation():
    with pytest.raises(ValueError, match="n debe ser positivo"):
        generate(n=0, n_features=10)
    with pytest.raises(ValueError, match="n_informative"):
        generate(n=10, n_features=5, n_informative=4, n_redundant=2)
    with pytest.raises(ValueError, match="task"):
        generate(n=10, n_features=15, task="other")


def _concat_chunks(**kwargs) -> dict:
    """Concatena los chunks de ``generate_chunks`` en un dict de arrays completos."""
    arrays: dict = {}
    for chunk in generate_chunks(**kwargs):
        for key, arr in chunk.items():
            arrays.setdefault(key, []).append(arr)
    return {key: np.concatenate(v) for key, v in arrays.items()}


def test_generate_chunks_shapes_and_columns():
    n, nf, ni, nr = 1000, 20, 5, 3
    data = _concat_chunks(n=n, n_features=nf, n_informative=ni, n_redundant=nr, seed=1)
    expected_cols = [f"x{i}" for i in range(ni)] + [f"r{i}" for i in range(nr)] + [
        f"n{i}" for i in range(nf - ni - nr)
    ]
    assert list(data) == expected_cols + ["y"]
    for key, arr in data.items():
        assert arr.shape == (n,), key


def test_generate_chunks_invariant_to_chunk_size():
    """Los valores no dependen de ``chunk_rows`` (stream por columna)."""
    kw = dict(n=500, n_features=15, n_informative=5, n_redundant=2, seed=7)
    full = _concat_chunks(chunk_rows=500, **kw)
    split = _concat_chunks(chunk_rows=77, **kw)
    for key in full:
        np.testing.assert_array_equal(full[key], split[key], err_msg=key)


def test_generate_chunks_deterministic():
    kw = dict(n=300, n_features=15, n_informative=5, n_redundant=2, seed=99)
    a = _concat_chunks(chunk_rows=101, **kw)
    b = _concat_chunks(chunk_rows=101, **kw)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_generate_chunks_planted_structure():
    """La estructura plantada se conserva: las informativas correlacionan
    con el target, el ruido independiente no."""
    n, nf, ni, nr = 4000, 25, 8, 4
    data = _concat_chunks(
        n=n, n_features=nf, n_informative=ni, n_redundant=nr, seed=5, task="classification"
    )
    assert set(np.unique(data["y"])) <= {0, 1}
    y = data["y"].astype(float)
    for i in range(ni):
        assert np.corrcoef(data[f"x{i}"], y)[0, 1] > 0.2, f"x{i} no informativa"
    for i in range(nf - ni - nr):
        assert abs(np.corrcoef(data[f"n{i}"], y)[0, 1]) < 0.1, f"n{i} debería ser ruido"


def test_generate_chunks_validation():
    with pytest.raises(ValueError, match="n debe ser positivo"):
        list(generate_chunks(n=0, n_features=10))
    with pytest.raises(ValueError, match="n_informative"):
        list(generate_chunks(n=10, n_features=5, n_informative=4, n_redundant=2))
    with pytest.raises(ValueError, match="task"):
        list(generate_chunks(n=10, n_features=15, task="other"))
