"""Tests de significancia (Hito 3): χ², BH-FDR y test de permutación.

- χ² y BH-FDR: tests puros (numpy/scipy), sin Spark.
- Permutación: test distribuido (``mapInPandas``) con Spark. Independencia →
  no significativo; dependencia → p < 0.05.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession
from scipy import stats

from sparkmim.info.entropy import mutual_information
from sparkmim.screen import _permutation_pvalues
from sparkmim.significance import bh_fdr, chi2_pvalue, permutation_pvalue


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("sparkmim-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


# --- χ² ---


def test_chi2_independent():
    # Tabla con independencia exacta => MI = 0, G² = 0, p = 1.
    table = np.array([[10, 10], [10, 10]], dtype=np.int64)
    mi = mutual_information(table)
    assert mi == pytest.approx(0.0, abs=1e-12)
    assert chi2_pvalue(table, mi) == pytest.approx(1.0)


def test_chi2_dependent():
    # Correlación perfecta => MI = log(2), G² = 2·200·log(2), df = 1.
    table = np.array([[100, 0], [0, 100]], dtype=np.int64)
    mi = mutual_information(table)
    assert mi == pytest.approx(np.log(2), abs=1e-12)
    p = chi2_pvalue(table, mi)
    assert p < 0.05
    assert p == pytest.approx(stats.chi2.sf(2 * 200 * np.log(2), 1))


def test_chi2_empty_and_degenerate():
    # Tabla vacía => p = 1.
    assert chi2_pvalue(np.zeros((2, 2), dtype=np.int64), 0.0) == 1.0
    # df no positivo (una dimensión) => p = 1.
    assert chi2_pvalue(np.array([[5, 5]], dtype=np.int64), 0.0) == 1.0


# --- BH-FDR ---


def test_bh_fdr_all_significant():
    # m=5, umbrales (1..5/5)·0.05 = [0.01, 0.02, 0.03, 0.04, 0.05].
    pvals = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    assert bh_fdr(pvals, q=0.05).all()


def test_bh_fdr_none_significant():
    pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    assert not bh_fdr(pvals, q=0.05).any()


def test_bh_fdr_partial():
    # m=4, q=0.05, umbrales [0.0125, 0.025, 0.0375, 0.05].
    # Solo el p más pequeño (0.01 <= 0.0125) pasa.
    pvals = np.array([0.01, 0.03, 0.04, 0.9])
    assert bh_fdr(pvals, q=0.05).tolist() == [True, False, False, False]


def test_bh_fdr_empty():
    assert bh_fdr(np.array([]), q=0.05).size == 0


# --- permutation_pvalue (función pura) ---


def test_permutation_pvalue_helper():
    # Todos los nulos < observado => p = 1/(B+1).
    assert permutation_pvalue(1.0, np.array([0.1, 0.2, 0.3])) == pytest.approx(1 / 4)
    # Todos los nulos >= observado => p = (1+B)/(B+1) = 1.
    assert permutation_pvalue(0.1, np.array([0.5, 0.6, 0.7])) == pytest.approx(1.0)
    # Sin permutaciones => p = 1.
    assert permutation_pvalue(0.5, np.array([])) == 1.0


# --- Test de permutación distribuido (Spark) ---


def _make_perm_df(spark, n, dependent, seed):
    """x con 4 categorías; y binaria dependiente (y = x % 2) o independiente."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=n)
    y = (x % 2) if dependent else rng.integers(0, 2, size=n)
    rows = [(int(a), int(b)) for a, b in zip(x, y)]
    return spark.createDataFrame(rows, ["x", "y"])


def test_permutation_independent_not_significant(spark):
    df = _make_perm_df(spark, n=2000, dependent=False, seed=1)
    pvals = _permutation_pvalues(
        df, ["x"], "y", [4], 2, n_b=100, n_rows_sub=2000, seed=42
    )
    # Independencia => p alto (no significativo).
    assert pvals[0] > 0.05


def test_permutation_dependent_significant(spark):
    df = _make_perm_df(spark, n=2000, dependent=True, seed=2)
    pvals = _permutation_pvalues(
        df, ["x"], "y", [4], 2, n_b=100, n_rows_sub=2000, seed=42
    )
    # Dependencia fuerte => p bajo (significativo).
    assert pvals[0] < 0.05
