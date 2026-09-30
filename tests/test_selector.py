"""Tests e2e de selección (Hito 4): criterios greedy + selector + model.

Estructura plantada:
- ``x0``, ``x1``: informativas (``y = x0 AND x1`` con ruido 10%).
- ``x2``: independiente de ``y``.
- ``x3``: redundante exacta con ``x0``.

Con ``significance="chi2"`` (default), el screening descarta ``x2`` (MI≈0)
antes del greedy; el greedy descarta ``x3`` (redundante). Se verifica que el
resultado sea ``{x0, x1}`` para cada criterio.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sparkmim import (
    CMIMSelector,
    InfoSelector,
    JMIMSelector,
    MIMSelector,
    MRMRSelector,
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


def _make_df(spark, n, seed):
    """x0, x1 informativas; x2 independiente; x3 redundante con x0 (exacta).

    ``y = x0 AND x1`` con 10% de ruido (ver ``planted.make_planted``).
    """
    data = make_planted(
        n,
        seed,
        [
            Feature("x0", "informative", kind="bernoulli"),
            Feature("x1", "informative", kind="bernoulli"),
            Feature("x2", "independent", kind="bernoulli"),
            Feature("x3", "redundant", copy_of="x0"),
        ],
        Target(kind="and", noise=0.1),
    )
    return to_spark_df(spark, data, as_strings=True)


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


# --- Task: fit + report (histograma, target entero) ---


def _make_df_int_target(spark, n, seed, n_classes):
    """x0, x1 informativas; x2 independiente; y entero con ``n_classes`` clases.

    ``y = sum(int(score > t) for t in thresholds)`` (ver ``planted``).
    """
    thresholds = tuple(
        round(-2.0 + i * 4 / (n_classes - 1), 4) for i in range(n_classes - 1)
    )
    data = make_planted(
        n,
        seed,
        [
            Feature("x0", "informative"),
            Feature("x1", "informative"),
            Feature("x2", "independent"),
        ],
        Target(kind="linear", noise=0.1, thresholds=thresholds),
    )
    return to_spark_df(spark, data)


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


def test_report_multiclass_int_target(spark):
    """Target entero de 15 clases (auto → multiclase) + report: macro AUC = 0.5."""
    df = _make_df_int_target(spark, n=4000, seed=5, n_classes=15)
    model = JMIMSelector(
        target="y", max_features=5, screen_top_k=10, significance="chi2", seed=42
    ).fit(df)
    assert model.task == "classifier_multiclass"
    rep = model.report(df, model=TrivialFactory(n_classes=15))
    assert rep.task == "classifier_multiclass"
    assert rep.metric_selected == pytest.approx(0.5)


def test_report_binary_int_target_explicit(spark):
    """Target entero de 2 clases declarado ``classifier_binary`` → AUC binaria = 0.5."""
    df = _make_df_int_target(spark, n=4000, seed=6, n_classes=2)
    model = JMIMSelector(
        target="y",
        task="classifier_binary",
        max_features=5,
        screen_top_k=10,
        significance="chi2",
        seed=42,
    ).fit(df)
    assert model.task == "classifier_binary"
    rep = model.report(df, model=TrivialFactory(n_classes=2))
    assert rep.task == "classifier_binary"
    assert rep.metric_selected == pytest.approx(0.5)


def test_report_continuous_declared_on_int(spark):
    """Target entero de 25 clases declarado ``continuous`` → R² = 0.0 (media)."""
    df = _make_df_int_target(spark, n=4000, seed=7, n_classes=25)
    model = JMIMSelector(
        target="y",
        task="continuous",
        max_features=5,
        screen_top_k=10,
        significance="chi2",
        seed=42,
    ).fit(df)
    assert model.task == "continuous"
    rep = model.report(df, model=TrivialFactory())
    assert rep.task == "continuous"
    assert rep.metric_selected == pytest.approx(0.0)
