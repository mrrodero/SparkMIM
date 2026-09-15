"""Tests de evaluación (Hito 6): métricas + report() con GBT spark.ml.

- Unitarios: ``auc_binary``, ``auc_macro``, ``r2_score``.
- E2E: ``report()`` con GBT en df sintético — AUC(seleccionadas) ≥
  AUC(todas) − ε; curva de eficiencia no decreciente.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim import (
    JMIMSelector,
    auc_binary,
    auc_macro,
    r2_score,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("sparkmim-test-eval")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


# --- Unitarios: métricas ---


def test_auc_binary_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_binary(y_true, y_prob) == pytest.approx(1.0)


def test_auc_binary_random():
    rng = np.random.default_rng(0)
    n = 2000
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.random(n)
    assert auc_binary(y_true, y_prob) == pytest.approx(0.5, abs=0.05)


def test_auc_binary_inverted():
    # Probabilidades invertidas → AUC ≈ 0.
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    assert auc_binary(y_true, y_prob) == pytest.approx(0.0)


def test_auc_macro_binary_matches_binary():
    # Multiclase con 2 clases: macro AUC = AUC binaria.
    y_true = np.array([0, 0, 1, 1])
    prob1 = np.array([0.1, 0.2, 0.8, 0.9])
    prob_matrix = np.array([[1 - p for p in prob1], prob1])  # (2, n)
    assert auc_macro(y_true, prob_matrix) == pytest.approx(
        auc_binary(y_true, prob1)
    )


def test_r2_score_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert r2_score(y_true, y_pred) == pytest.approx(1.0)


def test_r2_score_mean_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # media.
    assert r2_score(y_true, y_pred) == pytest.approx(0.0)


# --- E2E: report() con GBT ---


def _make_df(spark, n, seed):
    """x0, x1 informativas; x2..x4 ruido; y binaria."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x4 = rng.normal(size=n)
    logit = x0 + x1 + 0.5 * rng.normal(size=n)
    y = (logit > 0).astype(int)
    rows = [
        (float(a), float(b), float(c), float(d), float(e), int(f))
        for a, b, c, d, e, f in zip(x0, x1, x2, x3, x4, y)
    ]
    return spark.createDataFrame(rows, ["x0", "x1", "x2", "x3", "x4", "y"])


@pytest.fixture(scope="module")
def df(spark):
    return _make_df(spark, n=4000, seed=0)


def test_report_gbt(df):
    model = JMIMSelector(
        target="y",
        max_features=5,
        screen_top_k=10,
        significance="chi2",
        seed=42,
    ).fit(df)
    rep = model.report(df, model="gbt")
    # AUC(seleccionadas) ≥ AUC(todas) − ε.
    assert rep.metric_selected >= rep.metric_all - 0.05
    # AUC razonable (las features informativas capturan la señal).
    assert rep.metric_selected > 0.7
    # Curva de eficiencia no decreciente (con tolerancia).
    curve = rep.efficiency_curve
    assert len(curve) == len(model.ranking_)
    for (k1, m1), (k2, m2) in zip(curve, curve[1:]):
        assert m2 >= m1 - 0.02, f"curva decreciente en k={k2}: {m2} < {m1}"
    # El modelo y la tarea son correctos.
    assert rep.model_name == "gbt"
    assert rep.task == "classification"


def test_report_invalid_model(df):
    model = JMIMSelector(
        target="y",
        max_features=5,
        screen_top_k=10,
        significance="chi2",
        seed=42,
    ).fit(df)
    with pytest.raises(ValueError):
        model.report(df, model="bogus")
