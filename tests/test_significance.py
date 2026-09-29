"""Tests de significancia (Hito 3): costura ``SignificanceTest`` + adaptadores.

- Helpers puros (χ², p-valor de permutación, BH-FDR): sin Spark.
- Adaptadores ``Chi2Test`` y ``NoTest``: puros (sin Spark).
- Adaptador ``PermutationTest``: test distribuido (``mapInPandas``) con Spark.
  Independencia → no significativo; dependencia → p < 0.05 y significativo.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession
from scipy import stats

from sparkmim.config import SelectorConfig
from sparkmim.info.entropy import mutual_information
from sparkmim.significance import (
    Chi2Test,
    NoTest,
    PermutationTest,
    SignificanceInput,
    bh_fdr,
    chi2_pvalue,
    make_significance_test,
    permutation_pvalue,
)
from planted import Feature, Target, make_planted, to_spark_df


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


# --- Helpers puros (χ², permutación, BH-FDR) ---


def test_chi2_independent():
    # Tabla 2x2 uniforme: MI = 0 => p ≈ 1.
    table = np.array([[10, 10], [10, 10]], dtype=np.int64)
    mi = mutual_information(table)
    assert mi == pytest.approx(0.0, abs=1e-12)
    assert chi2_pvalue(table, mi) == pytest.approx(1.0)


def test_chi2_dependent():
    # Correlación perfecta: MI = log 2, G² = 2·n·MI.
    table = np.array([[100, 0], [0, 100]], dtype=np.int64)
    mi = mutual_information(table)
    assert mi == pytest.approx(np.log(2))
    g2 = 2 * 200 * mi
    expected = stats.chi2.sf(g2, 1)
    assert chi2_pvalue(table, mi) == pytest.approx(expected)
    assert chi2_pvalue(table, mi) < 0.05


def test_chi2_empty_and_degenerate():
    # Tabla vacía => p = 1.
    assert chi2_pvalue(np.zeros((2, 2), dtype=np.int64), 0.0) == 1.0
    # df = 0 (una fila o una columna) => p = 1.
    assert chi2_pvalue(np.array([[5, 5]], dtype=np.int64), 0.0) == 1.0


def test_bh_fdr_all_significant():
    p = np.array([0.001, 0.002, 0.003])
    assert bh_fdr(p, q=0.05).all()


def test_bh_fdr_none_significant():
    p = np.array([0.5, 0.6, 0.7])
    assert not bh_fdr(p, q=0.05).any()


def test_bh_fdr_partial():
    # Solo el p más pequeño sobrevive: p_(1) = 0.01 <= (1/4)·0.05 = 0.0125,
    # p_(2) = 0.03 > (2/4)·0.05 = 0.025.
    p = np.array([0.01, 0.03, 0.04, 0.9])
    assert bh_fdr(p, q=0.05).tolist() == [True, False, False, False]


def test_bh_fdr_empty():
    assert bh_fdr(np.array([]), q=0.05).size == 0


def test_permutation_pvalue_helper():
    # 1 de 3 nulas >= observada => (1+1)/(3+1) = 0.5.
    assert permutation_pvalue(0.5, np.array([0.1, 0.6, 0.2])) == pytest.approx(0.5)
    # Ninguna nula >= observada => 1/(B+1).
    assert permutation_pvalue(0.9, np.array([0.1, 0.2, 0.3])) == pytest.approx(1 / 4)
    # B = 0 => 1.0.
    assert permutation_pvalue(0.5, np.array([])) == 1.0


# --- Adaptadores (costura SignificanceTest) ---


def test_chi2_adapter():
    # x0: correlación perfecta (significativa); x1: independencia exacta.
    t_dep = np.array([[100, 0], [0, 100]], dtype=np.int64)
    t_indep = np.array([[10, 10], [10, 10]], dtype=np.int64)
    tables = {0: t_dep, 1: t_indep}
    mi = np.array([mutual_information(t_dep), mutual_information(t_indep)])
    result = Chi2Test(q=0.05).test(SignificanceInput(tables=tables, mi=mi))
    assert result.pvalues[0] < 0.05
    assert result.pvalues[1] == pytest.approx(1.0)
    assert result.significant.tolist() == [True, False]


def test_no_test_adapter():
    tables = {0: np.array([[5, 5], [5, 5]], dtype=np.int64)}
    mi = np.array([0.0])
    result = NoTest().test(SignificanceInput(tables=tables, mi=mi))
    assert result.pvalues == pytest.approx(np.array([1.0]))
    assert result.significant.all()


def test_factory_adapters():
    assert isinstance(
        make_significance_test(SelectorConfig(target="y", significance="chi2")),
        Chi2Test,
    )
    assert isinstance(
        make_significance_test(SelectorConfig(target="y", significance="permutation")),
        PermutationTest,
    )
    assert isinstance(
        make_significance_test(SelectorConfig(target="y", significance=None)),
        NoTest,
    )


def _make_perm_df(spark, n, seed):
    """3 features binarias + target binario.

    - x_dep: 90% correlada con y (fuerte).
    - x_indep: independiente de y (ruido).
    - x_noise: independiente de y (ruido; la de menor MI).
    """
    data = make_planted(
        n,
        seed,
        [
            Feature("x_dep", "correlated", agreement=0.9),
            Feature("x_indep", "independent", kind="bernoulli"),
            Feature("x_noise", "independent", kind="bernoulli"),
        ],
        Target(kind="flip"),
    )
    return to_spark_df(spark, data)


def test_permutation_adapter(spark):
    df = _make_perm_df(spark, n=2000, seed=1)
    # Las tablas no las usa el adaptador (las recomputa desde df_prep); se
    # pasan por uniformidad de la interfaz.
    tables = {
        i: np.array([[500, 500], [500, 500]], dtype=np.int64) for i in range(3)
    }
    # MI plantada: x_dep > x_indep > x_noise (el pre-filtro usa este orden).
    mi = np.array([0.5, 0.01, 0.005])
    test = PermutationTest(
        q=0.05, n_permutations=100, permutation_rows=2000, screen_top_k=2, seed=42
    )
    result = test.test(
        SignificanceInput(
            tables=tables,
            mi=mi,
            df_prep=df,
            feature_cols=["x_dep", "x_indep", "x_noise"],
            target_col="y",
            n_x=[2, 2, 2],
            n_y=2,
        )
    )
    # La dependiente (top-1 por MI) es significativa.
    assert result.pvalues[0] < 0.05
    assert result.significant[0]
    # La independiente (top-2) no lo es.
    assert not result.significant[1]
    # La de menor MI no se testa (pre-filtro top-K): p = 1, no significativa.
    assert result.pvalues[2] == pytest.approx(1.0)
    assert not result.significant[2]
