"""Tests unitarios de entropía/MI/CMI con valores analíticos (nats)."""

import numpy as np
import pytest

from sparkmim.info.entropy import (
    conditional_mi,
    entropy_from_counts,
    mutual_information,
)


def test_entropy_uniform():
    # H(uniforme sobre 4) = ln 4
    assert entropy_from_counts(np.array([1, 1, 1, 1])) == pytest.approx(np.log(4))


def test_entropy_nonuniform():
    # p = [0.9, 0.05, 0.05]
    p = np.array([0.9, 0.05, 0.05])
    expected = float(-(p * np.log(p)).sum())
    assert entropy_from_counts(np.array([90, 5, 5])) == pytest.approx(expected)


def test_entropy_zero_and_empty():
    # Los celdas vacías no contribuyen ni producen NaN: p = [0.5, 0.5, 0].
    assert entropy_from_counts(np.array([5, 5, 0])) == pytest.approx(
        float(-(0.5 * np.log(0.5) + 0.5 * np.log(0.5)))
    )
    assert entropy_from_counts(np.array([0, 0, 0])) == 0.0


def test_mi_xx_equals_entropy():
    # MI(X;X) = H(X); tabla diagonal.
    counts = np.array([5, 3, 2])
    table = np.zeros((3, 3), dtype=np.int64)
    for i, c in enumerate(counts):
        table[i, i] = c
    assert mutual_information(table) == pytest.approx(entropy_from_counts(counts))


def test_mi_independent_is_zero():
    # tabla = producto externo de marginales -> MI = 0
    px = np.array([60, 40])
    py = np.array([50, 30, 20])
    table = np.outer(px, py)
    assert mutual_information(table) == pytest.approx(0.0, abs=1e-12)


def test_mi_2x2_hand():
    # [[10, 20], [30, 40]] -> MI = 0.1 ln(5/6) + 0.2 ln(10/9)
    #                              + 0.3 ln(15/14) + 0.4 ln(20/21)
    table = np.array([[10, 20], [30, 40]], dtype=np.int64)
    expected = (
        0.1 * np.log(5 / 6)
        + 0.2 * np.log(10 / 9)
        + 0.3 * np.log(15 / 14)
        + 0.4 * np.log(20 / 21)
    )
    assert mutual_information(table) == pytest.approx(expected, abs=1e-12)


def test_mi_matches_entropy_identity():
    # MI = H(X) + H(Y) - H(X,Y) para una tabla general.
    table = np.array([[10, 20], [30, 40]], dtype=np.int64)
    px = table.sum(axis=1)
    py = table.sum(axis=0)
    expected = (
        entropy_from_counts(px) + entropy_from_counts(py) - entropy_from_counts(table)
    )
    assert mutual_information(table) == pytest.approx(expected, abs=1e-12)


def test_cmi_zero_when_z_determines_xy():
    # Z determina X e Y -> CMI = 0.
    table = np.zeros((2, 2, 2), dtype=np.int64)
    table[0, 0, 0] = 10
    table[1, 1, 1] = 10
    assert conditional_mi(table) == pytest.approx(0.0, abs=1e-12)


def test_cmi_equals_mi_when_z_independent():
    # Z independiente de (X,Y) -> CMI(X;Y|Z) = MI(X;Y).
    base = np.array([[10, 20], [30, 40]], dtype=np.int64)
    table = np.stack([base, base], axis=2)  # ambas rebanadas z iguales
    assert conditional_mi(table) == pytest.approx(
        mutual_information(base), abs=1e-12
    )


def test_cmi_positive_when_z_shares_info():
    # Sanity: CMI no negativa en un caso con dependencia real.
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, 2000)
    x = ((z + rng.integers(0, 2, 2000)) % 2)
    y = ((z + rng.integers(0, 2, 2000)) % 2)
    table = np.zeros((2, 2, 2), dtype=np.int64)
    for a, b, c in zip(x, y, z):
        table[a, b, c] += 1
    assert conditional_mi(table) > 0
