"""Tests e2e de selección (Hito 4): criterios greedy + selector + model.

Estructura plantada:
- ``x0``, ``x1``: informativas (``y = x0 AND x1`` con ruido 10%).
- ``x2``: independiente de ``y``.
- ``x3``: casi redundante con ``x0`` (``x0`` con ruido 5%).

Con ``significance="chi2"`` (default), el screening descarta ``x2`` (MI≈0)
antes del greedy; el greedy descarta ``x3`` (redundante). Se verifica que el
resultado sea ``{x0, x1}`` para cada criterio.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim import (
    CMIMSelector,
    InfoSelector,
    JMIMSelector,
    MIMSelector,
    MRMRSelector,
)


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


def _make_df(spark, n, seed):
    """x0, x1 informativas; x2 independiente; x3 redundante con x0 (exacta)."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, size=n)
    x1 = rng.integers(0, 2, size=n)
    x2 = rng.integers(0, 2, size=n)
    x3 = x0.copy()  # redundante exacta: CMI(x3; y | x0) = 0.
    y = (x0 & x1).astype(int)
    noise = rng.random(n) < 0.1
    y = np.where(noise, 1 - y, y)
    rows = [
        (str(a), str(b), str(c), str(d), str(e))
        for a, b, c, d, e in zip(x0, x1, x2, x3, y)
    ]
    return spark.createDataFrame(rows, ["x0", "x1", "x2", "x3", "y"])


@pytest.fixture(scope="module")
def df(spark):
    return _make_df(spark, n=4000, seed=0)


def _fit(selector_cls, df, **kwargs):
    kwargs.setdefault("target", "y")
    kwargs.setdefault("max_features", 5)
    kwargs.setdefault("screen_top_k", 10)
    kwargs.setdefault("significance", "chi2")
    kwargs.setdefault("seed", 42)
    return selector_cls(**kwargs).fit(df)


def _assert_selects_x0_x1(model):
    assert "x0" in model.selected_features
    assert "x1" in model.selected_features
    assert "x2" not in model.selected_features
    assert "x3" not in model.selected_features
    assert len(model.selected_features) <= 5
    assert len(model.scores_) == len(model.selected_features)


# --- JMIM (default) ---


def test_jmim_selects_informative(df):
    _assert_selects_x0_x1(_fit(JMIMSelector, df))


def test_jmim_ranking(df):
    model = _fit(JMIMSelector, df)
    ranked = [name for name, _ in model.ranking_]
    # x0 y x1 presentes; x3 (redundante, MI menor por el ruido de y) por
    # debajo de x0 en el ranking por MI.
    assert "x0" in ranked
    assert "x1" in ranked
    assert "x3" in ranked
    assert ranked.index("x3") > ranked.index("x0")


# --- mRMR / mIM / JMI ---


def test_mrmr_selects_informative(df):
    _assert_selects_x0_x1(_fit(MRMRSelector, df))


def test_mim_selects_informative(df):
    _assert_selects_x0_x1(_fit(MIMSelector, df))


def test_jmi_selects_informative(df):
    # JMI suma CMI sobre todas las ya elegidas: penaliza menos la redundancia
    # que JMIM, de modo que x3 (copia de x0) no se descarta. Por eso solo se
    # exige que x0 y x1 entren y x2 (independiente) no entre.
    model = _fit(InfoSelector, df, criterion="jmi")
    assert "x0" in model.selected_features
    assert "x1" in model.selected_features
    assert "x2" not in model.selected_features


# --- CMIM ---


def test_cmim_selects_informative(df):
    _assert_selects_x0_x1(_fit(CMIMSelector, df))


def test_cmim_approx_max_min(df):
    # cmim_approx="max_min" usa JMIM como aproximación ultrarrápida.
    _assert_selects_x0_x1(_fit(CMIMSelector, df, cmim_approx="max_min"))


# --- Parada por min_score sin filtro de significancia ---


def test_min_score_stops_without_significance(df):
    # Sin filtro FDR: el greedy debe parar por min_score (x2 y x3 tienen
    # CMI de ruido ~1e-4, por debajo de 1e-3).
    model = _fit(JMIMSelector, df, significance=None, min_score=1e-3)
    _assert_selects_x0_x1(model)


# --- transform ---


def test_transform(df):
    model = _fit(JMIMSelector, df)
    out = model.transform(df)
    out_cols = set(out.columns)
    assert "y" in out_cols
    for f in model.selected_features:
        assert f in out_cols
    # Solo features seleccionadas + target.
    assert len(out.columns) == len(model.selected_features) + 1


# --- criterio inválido ---


def test_invalid_criterion_raises():
    with pytest.raises(ValueError):
        InfoSelector(target="y", criterion="bogus")
