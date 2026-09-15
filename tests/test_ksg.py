"""Tests del modo KSG (Hito 5): estimador KSG + integración e2e.

- Unitarios: ``ksg_mi`` (independiente ≈ 0, dependiente > independiente) y
  ``ksg_cmi`` (identidad ``MI(XZ;Y) − MI(Z;Y)``).
- E2E: selector KSG sobre continuas (x0/x1 informativas, x2 independiente,
  x3 redundante con x0) → selecciona {x0, x1} y descarta x2.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim import InfoSelector, JMIMSelector, ksg_cmi, ksg_mi


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("sparkmim-test-ksg")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


# --- Unitarios: ksg_mi ---


def test_ksg_mi_independent_near_zero():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    mi = ksg_mi(x, y, k=10)
    assert mi < 0.05  # ≈ 0 (ruido de muestreo).


def test_ksg_mi_dependent_greater_than_independent():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    y_dep = x + 0.1 * rng.normal(size=n)  # dependiente.
    y_ind = rng.normal(size=n)            # independiente.
    mi_dep = ksg_mi(x, y_dep, k=10)
    mi_ind = ksg_mi(x, y_ind, k=10)
    assert mi_dep > mi_ind
    assert mi_dep > 0.1


def test_ksg_mi_multidimensional():
    # MI entre (x1, x2) y y, donde y depende de ambas.
    rng = np.random.default_rng(2)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 + x2 + 0.1 * rng.normal(size=n)
    mi_joint = ksg_mi(np.column_stack([x1, x2]), y, k=10)
    mi_single = ksg_mi(x1, y, k=10)
    # La conjunta (x1, x2) captura más información que x1 sola.
    assert mi_joint > mi_single


# --- Unitarios: ksg_cmi ---


def test_ksg_cmi_identity():
    # CMI(X;Y|Z) = MI(XZ;Y) − MI(Z;Y).
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = x + z + 0.1 * rng.normal(size=n)
    cmi = ksg_cmi(x, y, z, k=10)
    mi_xz_y = ksg_mi(np.column_stack([x, z]), y, k=10)
    mi_z_y = ksg_mi(z, y, k=10)
    assert abs(cmi - (mi_xz_y - mi_z_y)) < 1e-9


def test_ksg_cmi_zero_when_conditioned_on_self():
    # CMI(X;Y|X) = MI(X,X;Y) − MI(X;Y) = MI(X;Y) − MI(X;Y) = 0.
    rng = np.random.default_rng(4)
    n = 3000
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    cmi = ksg_cmi(x, y, x, k=10)
    # (X,X) es degenerado: la identidad da 0, con ruido de estimación KSG.
    assert abs(cmi) < 1e-3


# --- E2E: selector KSG ---


def _make_df(spark, n, seed):
    """x0, x1 informativas; x2 independiente; x3 redundante con x0 (exacta)."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x0.copy()  # redundante exacta.
    y = x0 + x1 + 0.1 * rng.normal(size=n)
    rows = [
        (float(a), float(b), float(c), float(d), float(e))
        for a, b, c, d, e in zip(x0, x1, x2, x3, y)
    ]
    return spark.createDataFrame(rows, ["x0", "x1", "x2", "x3", "y"])


@pytest.fixture(scope="module")
def df(spark):
    return _make_df(spark, n=3000, seed=0)


def test_ksg_selects_informative(df):
    model = InfoSelector(
        target="y",
        criterion="jmim",
        estimator="ksg",
        ksg_k=10,
        max_features=5,
        screen_top_k=10,
        seed=42,
    ).fit(df)
    assert "x0" in model.selected_features
    assert "x1" in model.selected_features
    assert "x2" not in model.selected_features
    # x3 (redundante) no es necesaria: JMIM la descarta (CMI ≈ 0 dado x0).
    assert "x3" not in model.selected_features
    assert len(model.scores_) == len(model.selected_features)


def test_ksg_mrmr_selects_informative(df):
    model = InfoSelector(
        target="y",
        criterion="mrmr",
        estimator="ksg",
        ksg_k=10,
        max_features=5,
        screen_top_k=10,
        seed=42,
    ).fit(df)
    assert "x0" in model.selected_features
    assert "x1" in model.selected_features
    assert "x2" not in model.selected_features


def test_ksg_transform(df):
    model = InfoSelector(
        target="y",
        criterion="jmim",
        estimator="ksg",
        ksg_k=10,
        max_features=5,
        screen_top_k=10,
        seed=42,
    ).fit(df)
    out = model.transform(df)
    out_cols = set(out.columns)
    assert "y" in out_cols
    for f in model.selected_features:
        assert f in out_cols
    assert len(out.columns) == len(model.selected_features) + 1
