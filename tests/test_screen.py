"""Tests end-to-end de la etapa 1 (screening): MI, significancia, FDR, top-K.

Dataset sintético con estructura plantada: x0 fuertemente dependiente de y,
x1 moderadamente, x2 independiente. Verifica el ranking por MI, la
significancia (χ² y permutación) y la selección de candidatas.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim.config import SelectorConfig
from sparkmim.schema import FeatureSpec, Schema
from sparkmim.screen import screen


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


def _make_screen_df(spark, n=2000, seed=0):
    """3 features binarias + target binario (códigos enteros).

    - x0: 80% correlada con y (fuerte).
    - x1: 70% correlada con y (moderada).
    - x2: independiente de y (ruido).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x0 = np.where(rng.random(n) < 0.8, y, 1 - y)
    x1 = np.where(rng.random(n) < 0.7, y, 1 - y)
    x2 = rng.integers(0, 2, size=n)
    rows = [(int(a), int(b), int(c), int(d)) for a, b, c, d in zip(x0, x1, x2, y)]
    return spark.createDataFrame(rows, ["x0", "x1", "x2", "y"])


def _make_schema():
    features = [
        FeatureSpec(name=f"x{i}", kind="categorical", n_codes=2) for i in range(3)
    ]
    target = FeatureSpec(name="y", kind="categorical", n_codes=2, is_target=True)
    return Schema(features=features, target=target)


def test_screen_chi2_ranks_dependent_first(spark):
    df = _make_screen_df(spark, n=2000, seed=0)
    schema = _make_schema()
    config = SelectorConfig(
        target="y", significance="chi2", screen_top_k=3, max_features=2
    )
    result = screen(df, schema, config)
    # Ranking por MI: x0 > x1 > x2.
    assert result.mi[0] > result.mi[1] > result.mi[2]
    # Las dependientes (x0, x1) son significativas.
    assert result.pvalues[0] < 0.05
    assert result.pvalues[1] < 0.05
    # Candidatas: x0 es la primera y x1 está presente.
    assert result.candidates[0] == 0
    assert 1 in result.candidates
    # Las tablas suman al nº de filas.
    assert all(t.sum() == result.n_rows for t in result.tables.values())


def test_screen_no_significance(spark):
    df = _make_screen_df(spark, n=2000, seed=0)
    schema = _make_schema()
    config = SelectorConfig(
        target="y", significance=None, screen_top_k=3, max_features=2
    )
    result = screen(df, schema, config)
    # Sin filtro: todas "significativas", candidatas = top-3 por MI.
    assert result.significant.all()
    assert len(result.candidates) == 3
    assert result.candidates[0] == 0


def test_screen_permutation(spark):
    df = _make_screen_df(spark, n=2000, seed=0)
    schema = _make_schema()
    config = SelectorConfig(
        target="y",
        significance="permutation",
        screen_top_k=3,
        max_features=2,
        n_permutations=100,
        permutation_rows=2000,
    )
    result = screen(df, schema, config)
    # x0 (fuerte) es significativa y entra en candidatas.
    assert result.significant[0]
    assert 0 in result.candidates
    # x0 sigue siendo la de mayor MI.
    assert result.mi[0] > result.mi[1]


def test_screen_invalid_significance():
    # La config rechaza un modo de significancia no válido (en __post_init__).
    with pytest.raises(ValueError, match="significance"):
        SelectorConfig(target="y", significance="bogus")
