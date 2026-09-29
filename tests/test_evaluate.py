"""Tests de evaluación (Hito 6): métricas + costura ``ModelFactory``.

- Unitarios: ``auc_binary``, ``auc_macro``, ``r2_score``, ``detect_task`` y
  ``make_model_factory``.
- Costura: ``TrivialFactory`` (adaptador trivial determinista, sin Spark ML)
  ejercita el cableado de la evaluación: clasificación AUC = 0.5 (empates),
  multiclase macro = 0.5 y regresión R² = 0.0 (la ruta de regresión, antes
  sin test).
- E2E: ``report()`` con GBT en df sintético — AUC(seleccionadas) ≥
  AUC(todas) − ε; curva de eficiencia no decreciente.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sparkmim import (
    GbtFactory,
    JMIMSelector,
    LightgbmFactory,
    XgboostFactory,
    auc_binary,
    auc_macro,
    detect_task,
    efficiency_curve,
    make_model_factory,
    r2_score,
    report,
    train_and_evaluate,
)
from planted import Feature, Target, make_planted, to_spark_df


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


# --- Unitarios: fábrica de modelos y detección de tarea ---


def test_make_model_factory_names():
    # Cada nombre de backend resuelve a su adaptador (sin importar extras).
    assert make_model_factory("gbt").name == "gbt"
    assert make_model_factory("xgboost").name == "xgboost"
    assert make_model_factory("lightgbm").name == "lightgbm"
    assert isinstance(make_model_factory("gbt"), GbtFactory)
    assert isinstance(make_model_factory("xgboost"), XgboostFactory)
    assert isinstance(make_model_factory("lightgbm"), LightgbmFactory)


def test_make_model_factory_bogus():
    # Nombre desconocido → ValueError (mensaje estable).
    with pytest.raises(ValueError, match="model_name debe ser"):
        make_model_factory("bogus")


def test_detect_task_string(spark):
    df = spark.createDataFrame([("a",), ("b",), ("a",)], ["y"])
    assert detect_task(df, "y") == "classification"


def test_detect_task_double(spark):
    df = spark.createDataFrame([(1.0,), (2.0,), (3.0,)], ["y"])
    assert detect_task(df, "y") == "regression"


def test_detect_task_int_few_classes(spark):
    # Entero con ≤ 20 valores distintos → clasificación.
    df = spark.createDataFrame([(1,), (2,), (3,)], ["y"])
    assert detect_task(df, "y") == "classification"


def test_detect_task_int_many_classes(spark):
    # Entero con > 20 valores distintos → regresión.
    df = spark.createDataFrame([(i,) for i in range(25)], ["y"])
    assert detect_task(df, "y") == "regression"


# --- Costura: adaptador trivial determinista (sin Spark ML) ---


class TrivialFactory:
    """Adaptador trivial determinista para tests (sin entrenamiento real).

    - Clasificación: ``probability`` = [0.5] * n_classes → AUC = 0.5
      (todos los empates; ``n_classes`` debe coincidir con el nº de clases
      porque ``auc_macro`` indexa la matriz por clase).
    - Regresión: ``prediction`` = media del target → R² = 0.0.
    """

    name = "trivial"

    def __init__(self, n_classes: int = 2):
        self._n_classes = n_classes

    def train(self, df_vec, task, label_col):
        if task == "classification":
            probs = F.array(*[F.lit(0.5) for _ in range(self._n_classes)])
            return df_vec.withColumn("probability", probs)
        mean = df_vec.select(F.mean(label_col)).first()[0]
        return df_vec.withColumn("prediction", F.lit(float(mean)))


_FEATS = [
    Feature("x0", "informative"),
    Feature("x1", "informative"),
    Feature("x2", "independent"),
    Feature("x3", "independent"),
    Feature("x4", "independent"),
]


def _make_df(spark, n, seed):
    """x0, x1 informativas; x2..x4 ruido; y binaria.

    ``y = (x0 + x1 + 0.5·N(0, 1) > 0)`` (ver ``planted.make_planted``).
    """
    data = make_planted(
        n, seed, _FEATS, Target(kind="linear", noise=0.5, thresholds=(0.0,))
    )
    return to_spark_df(spark, data)


def _make_df_multiclass(spark, n, seed):
    """x0, x1 informativas; x2..x4 ruido; y con 3 clases (0, 1, 2).

    ``y = int(score > 0) + int(score > 0.5)`` (ver ``planted.make_planted``).
    """
    data = make_planted(
        n, seed, _FEATS, Target(kind="linear", noise=0.5, thresholds=(0.0, 0.5))
    )
    return to_spark_df(spark, data)


def _make_df_reg(spark, n, seed):
    """x0, x1 informativas; x2..x4 ruido; y continua (regresión).

    ``y = x0 + x1 + 0.5·N(0, 1)`` (ver ``planted.make_planted``).
    """
    data = make_planted(n, seed, _FEATS, Target(kind="linear", noise=0.5))
    return to_spark_df(spark, data)


@pytest.fixture(scope="module")
def df(spark):
    return _make_df(spark, n=4000, seed=0)


@pytest.fixture(scope="module")
def df_multiclass(spark):
    return _make_df_multiclass(spark, n=2000, seed=1)


@pytest.fixture(scope="module")
def df_reg(spark):
    return _make_df_reg(spark, n=2000, seed=2)


def test_train_and_evaluate_trivial_classification(df):
    # Cableado de clasificación: codificación + ensamblaje + AUC (empates).
    metric = train_and_evaluate(df, ["x0", "x1"], "y", TrivialFactory())
    assert metric == pytest.approx(0.5)


def test_train_and_evaluate_trivial_multiclass(df_multiclass):
    # Multiclase: macro AUC one-vs-rest con 0.5 por clase.
    metric = train_and_evaluate(
        df_multiclass, ["x0", "x1"], "y", TrivialFactory(n_classes=3)
    )
    assert metric == pytest.approx(0.5)


def test_train_and_evaluate_trivial_regression(df_reg):
    # Ruta de regresión (antes sin test): predicción = media → R² = 0.0.
    metric = train_and_evaluate(df_reg, ["x0", "x1"], "y", TrivialFactory())
    assert metric == pytest.approx(0.0)


def test_efficiency_curve_trivial_classification(df):
    # Curva con la costura: una métrica por prefijo, todas 0.5.
    curve = efficiency_curve(df, ["x0", "x1", "x2"], "y", TrivialFactory())
    assert [k for k, _ in curve] == [1, 2, 3]
    assert all(m == pytest.approx(0.5) for _, m in curve)


def test_efficiency_curve_trivial_regression(df_reg):
    # Curva de regresión con la costura: R² = 0.0 en cada prefijo.
    curve = efficiency_curve(df_reg, ["x0", "x1", "x2"], "y", TrivialFactory())
    assert [k for k, _ in curve] == [1, 2, 3]
    assert all(m == pytest.approx(0.0) for _, m in curve)


def test_report_trivial_classification(df):
    # report() con una ModelFactory (no un nombre): forma y métricas.
    rep = report(
        df,
        selected_features=["x0", "x1"],
        ranking=["x0", "x1", "x2"],
        target_col="y",
        model=TrivialFactory(),
    )
    assert rep.model_name == "trivial"
    assert rep.task == "classification"
    assert rep.metric_selected == pytest.approx(0.5)
    assert rep.metric_all == pytest.approx(0.5)
    assert len(rep.efficiency_curve) == 3
    assert rep.selected_features == ["x0", "x1"]


def test_report_trivial_regression(df_reg):
    # report() de regresión con la costura: R² = 0.0 en todo.
    rep = report(
        df_reg,
        selected_features=["x0", "x1"],
        ranking=["x0", "x1", "x2"],
        target_col="y",
        model=TrivialFactory(),
    )
    assert rep.model_name == "trivial"
    assert rep.task == "regression"
    assert rep.metric_selected == pytest.approx(0.0)
    assert rep.metric_all == pytest.approx(0.0)
    assert all(m == pytest.approx(0.0) for _, m in rep.efficiency_curve)


# --- E2E: report() con GBT ---


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
