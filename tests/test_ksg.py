"""Tests del estimador KSG (Hito 5): estimador KSG + integración e2e.

- Unitarios: ``ksg_mi`` (independiente ≈ 0, dependiente > independiente) y
  ``ksg_cmi`` (identidad ``MI(XZ;Y) − MI(Z;Y)``).
- E2E: selector KSG sobre continuas (x0/x1 informativas, x2 independiente,
  x3 redundante con x0) → selecciona {x0, x1} y descarta x2.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sparkmim import InfoSelector, JMIMSelector, ksg_cmi, ksg_mi
from planted import Feature, Target, make_planted, to_spark_df


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
    """x0, x1 informativas; x2 independiente; x3 redundante con x0 (exacta).

    ``y = x0 + x1 + 0.1·N(0, 1)`` (ver ``planted.make_planted``).
    """
    data = make_planted(
        n,
        seed,
        [
            Feature("x0", "informative"),
            Feature("x1", "informative"),
            Feature("x2", "independent"),
            Feature("x3", "redundant", copy_of="x0"),
        ],
        Target(kind="linear", noise=0.1),
    )
    return to_spark_df(spark, data)


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


# --- Regresión: índice global vs posición en candidatas (estimador KSG) ---
#
# El greedy trabaja en posiciones de candidata (0..K-1). El código antiguo
# indexaba la lista ``candidates`` (longitud K) con el índice global de
# feature (0..N-1): ``IndexError`` cuando ``screen_top_k < N`` y la mejor
# feature queda fuera de las primeras K posiciones, y feature/condicionamiento
# equivocados en silencio en los demás casos.


def _make_df_wide(spark, n, seed, weights):
    """6 features; ``weights[i]`` es el coeficiente de ``x_i`` en ``y``.

    Se queda inline (no usa ``planted``): el único draw de matriz
    ``rng.normal((n, 6))`` no encaja en el modelo de draws por feature del
    generador compartido.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    y = sum(w * x[:, i] for i, w in enumerate(weights)) + 0.1 * rng.normal(size=n)
    rows = [tuple(map(float, r)) + (float(v),) for r, v in zip(x, y)]
    return spark.createDataFrame(rows, ["x0", "x1", "x2", "x3", "x4", "x5", "y"])


@pytest.fixture(scope="module")
def df_best_x5(spark):
    """x5 (índice global 5) es la mejor; x3 segunda; x0..x4 independientes."""
    return _make_df_wide(spark, n=3000, seed=7, weights=[0.0, 0.0, 0.0, 1.0, 0.0, 2.0])


@pytest.fixture(scope="module")
def df_order_x1_x2_x0(spark):
    """Orden de MI x1 > x2 > x0 (la mejor no está en el índice global 0)."""
    return _make_df_wide(spark, n=3000, seed=8, weights=[0.5, 1.6, 1.4, 0.0, 0.0, 0.0])


def test_ksg_jmim_best_outside_first_k_positions(df_best_x5):
    """screen_top_k=3 < N=6 y la mejor en el índice global 5: antes → IndexError."""
    model = InfoSelector(
        target="y",
        criterion="jmim",
        estimator="ksg",
        ksg_k=10,
        max_features=2,
        screen_top_k=3,
        seed=42,
    ).fit(df_best_x5)
    assert model.selected_features[0] == "x5"


def test_ksg_cmim_best_outside_first_k_positions(df_best_x5):
    """Igual con cmim exacta: la línea de ``s_m`` también indexaba con el global."""
    model = InfoSelector(
        target="y",
        criterion="cmim",
        estimator="ksg",
        ksg_k=10,
        max_features=2,
        screen_top_k=3,
        seed=42,
    ).fit(df_best_x5)
    assert model.selected_features[0] == "x5"


def test_ksg_jmim_silent_wrong_first_pick(df_order_x1_x2_x0):
    """Mejor en el índice global 1 < K, pero orden de MI ≠ identidad:
    antes elegía x2 (``candidates[1]``) en lugar de x1."""
    model = InfoSelector(
        target="y",
        criterion="jmim",
        estimator="ksg",
        ksg_k=10,
        max_features=3,
        screen_top_k=3,
        seed=42,
    ).fit(df_order_x1_x2_x0)
    assert model.selected_features[0] == "x1"
    assert model.selected_features[1] == "x2"


def test_ksg_cmim_silent_correct_s_m(df_order_x1_x2_x0):
    """``s_m`` correcto = {x1, x2}; el código antiguo daba {x2, x0}."""
    model = InfoSelector(
        target="y",
        criterion="cmim",
        estimator="ksg",
        ksg_k=10,
        max_features=2,
        screen_top_k=3,
        seed=42,
    ).fit(df_order_x1_x2_x0)
    assert model.selected_features[0] == "x1"
    assert model.selected_features[1] == "x2"


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


# --- Modo KSG: columnas no numéricas → error claro en fit ---


def test_ksg_rejects_non_numeric_feature(spark):
    """Feature string con KSG: ValueError claro (no crash en to_numpy)."""
    rows = [(1.0, "a", 2.0), (2.0, "b", 3.0), (3.0, "a", 4.0)]
    df = spark.createDataFrame(rows, ["x0", "cat", "y"])
    with pytest.raises(ValueError) as excinfo:
        InfoSelector(target="y", estimator="ksg", ksg_k=10).fit(df)
    assert "cat" in str(excinfo.value)
    assert "numéricas" in str(excinfo.value)


def test_ksg_rejects_non_numeric_target_continuous(spark):
    """Target string declarado ``continuous`` con KSG: ValueError claro
    nombrando la columna."""
    rows = [(1.0, "ok"), (2.0, "ok"), (3.0, "ok")]
    df = spark.createDataFrame(rows, ["x0", "y"])
    with pytest.raises(ValueError) as excinfo:
        InfoSelector(target="y", task="continuous", estimator="ksg", ksg_k=10).fit(df)
    assert "y" in str(excinfo.value)
    assert "numérico" in str(excinfo.value)


def test_ksg_auto_unsupported_target_dtype(spark):
    """Task ``auto`` con un dtype de target no reconocido (array): ValueError
    claro nombrando la columna."""
    from pyspark.sql.types import (
        ArrayType,
        DoubleType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("x0", DoubleType()),
            StructField("y", ArrayType(DoubleType())),
        ]
    )
    df = spark.createDataFrame([(1.0, [1.0]), (2.0, [2.0])], schema)
    with pytest.raises(ValueError) as excinfo:
        InfoSelector(target="y", estimator="ksg", ksg_k=10).fit(df)
    assert "y" in str(excinfo.value)
    assert "no reconoce" in str(excinfo.value)


# --- KSG: clasificación multiclase (target entero, codificación lossless) ---


_KSG_MC_THRESHOLDS = tuple(round(-2.0 + i * 4 / 14, 4) for i in range(14))


def _make_df_ksg_multiclass(spark, n, seed):
    """x0, x1 informativas; x2 independiente; y con 15 clases enteras (0..14).

    ``y = sum(int(score > t) for t in thresholds)`` (ver ``planted``).
    """
    data = make_planted(
        n,
        seed,
        [
            Feature("x0", "informative"),
            Feature("x1", "informative"),
            Feature("x2", "independent"),
        ],
        Target(kind="linear", noise=0.1, thresholds=_KSG_MC_THRESHOLDS),
    )
    return to_spark_df(spark, data)


@pytest.fixture(scope="module")
def df_ksg_multiclass(spark):
    return _make_df_ksg_multiclass(spark, n=3000, seed=3)


def test_ksg_multiclass_resolves_task(df_ksg_multiclass):
    """KSG con task ``auto`` y target entero de 15 clases → multiclase."""
    model = InfoSelector(
        target="y",
        criterion="jmim",
        estimator="ksg",
        ksg_k=10,
        max_features=5,
        screen_top_k=10,
        seed=42,
    ).fit(df_ksg_multiclass)
    assert model.task == "classifier_multiclass"
    assert "x0" in model.selected_features
    assert "x1" in model.selected_features
    assert "x2" not in model.selected_features


class TrivialFactory:
    """Adaptador trivial determinista para tests (sin entrenamiento real).

    - Clasificación: ``probability`` = [0.5] * n_classes → AUC = 0.5
      (todos los empates).
    - Continua: ``prediction`` = media del target → R² = 0.0.
    """

    name = "trivial"

    def __init__(self, n_classes: int = 2):
        self._n_classes = n_classes

    def train(self, df_vec, task, label_col):
        if task in ("classifier_binary", "classifier_multiclass"):
            probs = F.array(*[F.lit(0.5) for _ in range(self._n_classes)])
            return df_vec.withColumn("probability", probs)
        mean = df_vec.select(F.mean(label_col)).first()[0]
        return df_vec.withColumn("prediction", F.lit(float(mean)))


def test_ksg_multiclass_report(df_ksg_multiclass):
    """report() tras un fit KSG multiclase: macro AUC = 0.5 (empates) y la
    task resuelta viaja con el modelo."""
    model = InfoSelector(
        target="y",
        criterion="jmim",
        estimator="ksg",
        ksg_k=10,
        max_features=5,
        screen_top_k=10,
        seed=42,
    ).fit(df_ksg_multiclass)
    rep = model.report(df_ksg_multiclass, model=TrivialFactory(n_classes=15))
    assert rep.task == "classifier_multiclass"
    assert rep.metric_selected == pytest.approx(0.5)
    assert rep.metric_all == pytest.approx(0.5)
