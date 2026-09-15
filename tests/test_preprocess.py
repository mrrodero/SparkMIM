"""Tests de la etapa 0 (preprocesado): bins, missing, cardinalidad, target."""

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim.config import SelectorConfig
from sparkmim.preprocess import prepare


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("sparkmim-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    yield session
    session.stop()


def _cfg(**kw):
    base = dict(target="y", seed=42)
    base.update(kw)
    return SelectorConfig(**base)


def test_numeric_binning_frequencies(spark):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 10_000)
    y = rng.integers(0, 2, 10_000)
    df = spark.createDataFrame(list(zip(x.tolist(), y.tolist())), "x: double, y: long")
    prep = prepare(df, _cfg(bins=10, missing="drop"))
    spec = prep.schema.features[0]
    assert spec.n_codes == 10
    codes = [r[0] for r in prep.df_prep.select("x").collect()]
    counts = np.bincount(codes, minlength=10)
    # Bins cuantiles de uniforme: cada bin ≈ 1000 ± 10%.
    assert all(900 <= c <= 1100 for c in counts), counts


def test_missing_category_code(spark):
    rng = np.random.default_rng(0)
    n = 5_000
    x = rng.uniform(0, 1, n)
    mask = rng.random(n) < 0.1
    x_list = [None if m else float(v) for v, m in zip(x, mask)]
    y = rng.integers(0, 2, n)
    df = spark.createDataFrame(list(zip(x_list, y.tolist())), "x: double, y: long")
    prep = prepare(df, _cfg(bins=10, missing="category"))
    spec = prep.schema.features[0]
    assert spec.missing_code == 10
    assert spec.n_codes == 11
    rows = prep.df_prep.select("x").collect()
    codes = [r[0] for r in rows]
    # El nº de códigos de missing coincide con el nº de nulos.
    assert codes.count(10) == int(mask.sum())
    # Los no-nulos caen en bins 0..9 coherentes con las fronteras.
    edges = spec.bin_edges
    # Solo los valores NO enmascarados (los nulos son el código 10).
    expected = sorted(
        sum(1 for e in edges if v >= e) for v, m in zip(x, mask) if not m
    )
    actual = sorted(c for c in codes if c != 10)
    assert expected == actual


def test_missing_drop(spark):
    rng = np.random.default_rng(0)
    n = 5_000
    x = rng.uniform(0, 1, n)
    mask = rng.random(n) < 0.1
    x_list = [None if m else float(v) for v, m in zip(x, mask)]
    y = rng.integers(0, 2, n)
    df = spark.createDataFrame(list(zip(x_list, y.tolist())), "x: double, y: long")
    prep = prepare(df, _cfg(bins=10, missing="drop"))
    assert prep.n_rows == n - int(mask.sum())
    codes = [r[0] for r in prep.df_prep.select("x").collect()]
    assert all(0 <= c <= 9 for c in codes)


def test_categorical_cardinality_cap(spark):
    rng = np.random.default_rng(0)
    n = 10_000
    weights = 1.0 / (np.arange(100) + 1)
    x = rng.choice(100, size=n, p=weights / weights.sum())
    y = rng.integers(0, 2, n)
    df = spark.createDataFrame(
        [(str(a), int(b)) for a, b in zip(x, y)], "x: string, y: long"
    )
    prep = prepare(df, _cfg(max_categories=50, missing="drop"))
    spec = prep.schema.features[0]
    assert spec.n_codes == 51  # 50 top + "other"
    assert spec.has_other
    codes = [r[0] for r in prep.df_prep.select("x").collect()]
    # La categoría más frecuente ("0") recibe el código 0.
    idx0 = [i for i, v in enumerate(x) if v == 0]
    assert all(codes[i] == 0 for i in idx0)
    # Las categorías raras (fuera del top-50) reciben el código "other" (50).
    idx99 = [i for i, v in enumerate(x) if v == 99]
    assert idx99, "la categoría 99 debe aparecer en los datos"
    assert all(codes[i] == 50 for i in idx99)


def test_categorical_no_cap_when_small(spark):
    # Frecuencias deterministas: a (1500) > b (1000) > c (500).
    n = 3_000
    x = (["a"] * 1500) + (["b"] * 1000) + (["c"] * 500)
    y = [i % 2 for i in range(n)]
    df = spark.createDataFrame(list(zip(x, y)), "x: string, y: long")
    prep = prepare(df, _cfg(max_categories=50, missing="drop"))
    spec = prep.schema.features[0]
    assert spec.n_codes == 3
    assert not spec.has_other
    assert spec.category_codes == {"a": 0, "b": 1, "c": 2}


def test_regression_target_binned(spark):
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    df = spark.createDataFrame(list(zip(x.tolist(), y.tolist())), "x: double, y: double")
    prep = prepare(df, _cfg(bins=10, bins_target=10, missing="drop"))
    spec = prep.schema.target
    assert spec.n_codes == 10
    codes = [r[0] for r in prep.df_prep.select("y").collect()]
    counts = np.bincount(codes, minlength=10)
    # Bins cuantiles de normal: cada bin ≈ 1000 ± 20%.
    assert all(800 <= c <= 1200 for c in counts), counts


def test_binary_target_two_codes(spark):
    rng = np.random.default_rng(0)
    n = 5_000
    x = rng.uniform(0, 1, n)
    y = rng.integers(0, 2, n)
    df = spark.createDataFrame(list(zip(x.tolist(), y.tolist())), "x: double, y: long")
    prep = prepare(df, _cfg(bins=10, missing="drop"))
    spec = prep.schema.target
    assert spec.n_codes == 2
    codes = [r[0] for r in prep.df_prep.select("y").collect()]
    # Códigos 0/1 que conservan el valor original.
    assert sorted(codes) == sorted(y.tolist())


def test_multiclass_target(spark):
    rng = np.random.default_rng(0)
    n = 9_000
    x = rng.uniform(0, 1, n)
    y = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    df = spark.createDataFrame(list(zip(x.tolist(), y.tolist())), "x: double, y: string")
    prep = prepare(df, _cfg(bins=10, missing="drop"))
    spec = prep.schema.target
    assert spec.n_codes == 3
    # Orden por frecuencia: a (0.5) → 0, b (0.3) → 1, c (0.2) → 2.
    assert spec.category_codes == {"a": 0, "b": 1, "c": 2}
    codes = [r[0] for r in prep.df_prep.select("y").collect()]
    assert sorted(codes) == sorted(spec.category_codes[v] for v in y)


def test_bins_auto(spark):
    rng = np.random.default_rng(0)
    n = 100_000
    x = rng.uniform(0, 1, n)
    y = rng.integers(0, 2, n)
    df = spark.createDataFrame(list(zip(x.tolist(), y.tolist())), "x: double, y: long")
    prep = prepare(df, _cfg(bins="auto", missing="drop"))
    assert prep.schema.features[0].n_codes == 17  # round(log2 1e5) = 17


def test_constant_feature_one_bin(spark):
    n = 1_000
    df = spark.createDataFrame(
        [(5.0, int(i % 2)) for i in range(n)], "x: double, y: long"
    )
    prep = prepare(df, _cfg(bins=10, missing="drop"))
    spec = prep.schema.features[0]
    assert spec.n_codes == 1
    assert spec.bin_edges == ()
    codes = [r[0] for r in prep.df_prep.select("x").collect()]
    assert set(codes) == {0}
