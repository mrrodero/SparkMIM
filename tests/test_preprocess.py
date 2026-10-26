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


# --- Etapa 0: representación del target según la Task ---


def test_int_target_15_classes_auto_multiclass(spark):
    """Target entero de 15 clases con task ``auto`` → multiclase: 15 códigos,
    uno por valor distinto."""
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.uniform(0, 1, n)
    y = rng.integers(0, 15, n)
    df = spark.createDataFrame(list(zip(x.tolist(), y.tolist())), "x: double, y: long")
    prep = prepare(df, _cfg(bins=10, missing="drop"))
    assert prep.task == "classifier_multiclass"
    spec = prep.schema.target
    assert spec.kind == "categorical"
    assert spec.n_codes == 15
    codes = [r[0] for r in prep.df_prep.select("y").collect()]
    # Los códigos renumeran las clases (en orden de frecuencia); el mapeo
    # es coherente fila a fila y usa los 15 códigos.
    assert set(codes) == set(range(15))
    assert codes == [spec.category_codes[str(v)] for v in y.tolist()]


def test_float_target_declared_multiclass(spark):
    """Target float de 5 niveles declarado ``classifier_multiclass`` → 5
    códigos, uno por valor distinto."""
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.uniform(0, 1, n)
    y = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n)
    df = spark.createDataFrame(
        list(zip(x.tolist(), y.tolist())), "x: double, y: double"
    )
    prep = prepare(
        df, _cfg(bins=10, task="classifier_multiclass", missing="drop")
    )
    assert prep.task == "classifier_multiclass"
    spec = prep.schema.target
    assert spec.kind == "categorical"
    assert spec.n_codes == 5
    codes = [r[0] for r in prep.df_prep.select("y").collect()]
    # Mapeo coherente fila a fila (las claves del mapa son strings) y
    # se usan los 5 códigos.
    assert set(codes) == set(range(5))
    assert codes == [spec.category_codes[str(v)] for v in y.tolist()]


def test_int_target_declared_continuous(spark):
    """Target entero declarado ``continuous`` → binning cuantil (10 bins)."""
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.uniform(0, 1, n)
    y = rng.integers(0, 100, n)
    df = spark.createDataFrame(list(zip(x.tolist(), y.tolist())), "x: double, y: long")
    prep = prepare(
        df, _cfg(bins=10, bins_target=10, task="continuous", missing="drop")
    )
    assert prep.task == "continuous"
    spec = prep.schema.target
    assert spec.kind == "numeric"
    assert spec.n_codes == 10


# ---------------------------------------------------------------------------
# Pase 0a compacto: equivalencia bit a bit de _stats_partition con pandas
# ---------------------------------------------------------------------------


def _numeric_rows(pdf, n_cols, bins, bins_target=None, is_target=None):
    """Filas numéricas de ``_stats_partition`` como dict {col: (min, max, qs)}."""
    from sparkmim.preprocess import _KIND_NUMERIC, _stats_partition

    cols = [(f"c{i}", "numeric", "float64") for i in range(n_cols)]
    target = is_target is not None
    if target:
        cols.append(("y", "numeric", "float64"))
    it = ([False] * n_cols) + ([True] if target else [])
    rows = _stats_partition(pdf, cols, bins, bins_target or bins, it)
    out = {}
    for r in rows:
        if r[1] == _KIND_NUMERIC:
            idx = r[0]
            name = cols[idx][0]
            out[name] = (r[5], r[6], list(r[7]))
    return out


def test_stats_partition_bitwise_vs_pandas():
    """Fast path (sin NaN): min/max y cuantiles idénticos a pandas, bit a bit."""
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 5_000
    a = rng.normal(size=n)
    b = rng.exponential(2.0, size=n)
    pdf = pd.DataFrame({"c0": a, "c1": b})
    got = _numeric_rows(pdf, 2, bins=13)
    for name, arr in (("c0", a), ("c1", b)):
        mn, mx, qs = got[name]
        assert mn == float(pd.Series(arr).min())
        assert mx == float(pd.Series(arr).max())
        ref = [float(q) for q in pd.Series(arr).quantile([i / 13 for i in range(1, 13)])]
        np.testing.assert_array_equal(np.asarray(qs), np.asarray(ref))


def test_stats_partition_constant_column_empty_quantiles():
    import pandas as pd

    pdf = pd.DataFrame({"c0": np.full(100, 3.5), "c1": np.arange(100, dtype=float)})
    got = _numeric_rows(pdf, 2, bins=10)
    assert got["c0"][2] == []
    assert got["c0"][0] == got["c0"][1] == 3.5
    assert len(got["c1"][2]) == 9


def test_stats_partition_nan_fallback_matches_dropna():
    """Con NaNs: fallback pandas, cuantiles sobre ``dropna`` (bit a bit)."""
    import pandas as pd

    rng = np.random.default_rng(11)
    n = 2_000
    a = rng.normal(size=n)
    a[[10, 500, 1500]] = np.nan
    pdf = pd.DataFrame({"c0": a})
    got = _numeric_rows(pdf, 1, bins=10)
    valid = pd.Series(a).dropna()
    mn, mx, qs = got["c0"]
    assert mn == float(valid.min())
    assert mx == float(valid.max())
    ref = [float(q) for q in valid.quantile([i / 10 for i in range(1, 10)])]
    np.testing.assert_array_equal(np.asarray(qs), np.asarray(ref))


def test_stats_partition_all_nan_column_emits_no_row():
    import pandas as pd

    pdf = pd.DataFrame({"c0": np.full(50, np.nan)})
    got = _numeric_rows(pdf, 1, bins=10)
    assert "c0" not in got


def test_stats_partition_target_uses_bins_target():
    """El target usa ``bins_target`` (aqui 4), las features, ``bins`` (8)."""
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 1_000
    pdf = pd.DataFrame({"c0": rng.normal(size=n), "y": rng.normal(size=n)})
    from sparkmim.preprocess import _KIND_NUMERIC, _stats_partition

    rows = _stats_partition(
        pdf,
        [("c0", "numeric", "float64"), ("y", "numeric", "float64")],
        bins=8,
        bins_target=4,
        is_target=[False, True],
    )
    by_fid = {r[0]: r for r in rows if r[1] == _KIND_NUMERIC}
    assert len(by_fid[0][7]) == 7  # c0: 8 bins → 7 cuantiles
    assert len(by_fid[1][7]) == 3  # y: 4 bins → 3 cuantiles
