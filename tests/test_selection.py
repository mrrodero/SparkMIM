"""Tests de la costura de selección (etapa 3): ``greedy_select`` + oráculos.

- Unitarios de ``greedy_select`` con un oráculo falso (sin Spark): ronda 1,
  criterios rápidos (mrmr/mim/jmi/jmim), ``min_score``, ``max_features``,
  pase CMIM (``S_m``), ``cmim``+``max_min`` y criterio inválido.
- Unitarios de ``KsgOracle`` (numpy puro, semillas fijas).
- Invariante de equivalencia (Spark): ``HistogramOracle.cmi_set_all`` ≡
  ``conditional_mi`` directo sobre el mismo subsample (conteos exactos).
"""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim.config import SelectorConfig
from sparkmim.info.entropy import conditional_mi, conditional_mi_multi
from sparkmim.info.ksg import ksg_cmi
from sparkmim.oracles import HistogramOracle, KsgOracle
from sparkmim.preprocess import prepare
from sparkmim.screen import screen
from sparkmim.selection import greedy_select
from sparkmim.tables import TableCache, build_joint_tables, dense_from_joints


# ----------------------------------------------------------------------
# Oráculo falso para los unitarios de greedy_select.
# ----------------------------------------------------------------------


class FakeOracle:
    """Oráculo falso: MI/CMI desde dicts, con registro de llamadas."""

    def __init__(self, mi, mi_pair=None, cmi_single=None, cmi_set=None):
        self.mi = np.asarray(mi, dtype=float)
        # Los dicts llevan prefijo `_`: no deben sombrear los métodos del protocolo.
        self._mi_pair = dict(mi_pair or {})
        self._cmi_single = dict(cmi_single or {})
        self._cmi_set = dict(cmi_set or {})
        self.mi_all_calls = 0
        self.cmi_set_calls = []

    def mi_all(self):
        self.mi_all_calls += 1
        return self.mi.copy()

    def mi_pair(self, i, j):
        return float(self._mi_pair.get((min(i, j), max(i, j)), 0.0))

    def cmi_single(self, i, j):
        return float(self._cmi_single.get((i, j), 0.0))

    def cmi_set_all(self, s_m):
        self.cmi_set_calls.append(list(s_m))
        return np.array([float(self._cmi_set.get(i, 0.0)) for i in range(len(self.mi))])


# ----------------------------------------------------------------------
# Unitarios: greedy_select (oráculo falso, sin Spark).
# ----------------------------------------------------------------------


def test_greedy_round1_argmax():
    oracle = FakeOracle(mi=[0.1, 0.5, 0.3])
    result = greedy_select(oracle, "jmim", "max_min", 1e-4, 1, 2)
    assert result.selected == [1]
    assert result.scores == [0.5]


def test_greedy_jmim_uses_cmi_single():
    # Ronda 1: argmax MI = 0. Ronda 2: min CMI(X_i;Y|X0): x1 → 0.2, x2 → 0.4.
    oracle = FakeOracle(
        mi=[1.0, 0.9, 0.8],
        cmi_single={(1, 0): 0.2, (2, 0): 0.4},
    )
    result = greedy_select(oracle, "jmim", "max_min", 1e-4, 2, 2)
    assert result.selected == [0, 2]
    assert result.scores == [1.0, 0.4]


def test_greedy_mrmr_and_mim_use_mi_pair():
    # Ronda 2: mrmr x1 → 0.9 − 0.5 = 0.4; x2 → 0.8 − 0.1 = 0.7 → x2.
    # mim: mismos valores (|S| = 1) → x2.
    oracle = FakeOracle(
        mi=[1.0, 0.9, 0.8],
        mi_pair={(0, 1): 0.5, (0, 2): 0.1},
    )
    for crit in ("mrmr", "mim"):
        result = greedy_select(oracle, crit, "max_min", 1e-4, 2, 2)
        assert result.selected == [0, 2]
        assert result.scores == pytest.approx([1.0, 0.7])


def test_greedy_jmi_sums_cmi():
    # S = [0, 1] tras rondas 1-2. Ronda 3:
    # jmi x2 → 0.1 + 0.4 = 0.5; x3 → 0.2 + 0.15 = 0.35 → x2.
    oracle = FakeOracle(
        mi=[1.0, 0.9, 0.8, 0.7],
        cmi_single={
            (1, 0): 0.5,
            (2, 0): 0.1,
            (2, 1): 0.4,
            (3, 0): 0.2,
            (3, 1): 0.15,
        },
    )
    result = greedy_select(oracle, "jmi", "max_min", 1e-4, 3, 2)
    assert result.selected == [0, 1, 2]
    assert result.scores == [1.0, 0.5, 0.5]


def test_greedy_min_score_stops():
    # Ronda 1: 1.0 ≥ 0.95 → selecciona 0. Ronda 2: mejor 0.4 < 0.95 → para.
    oracle = FakeOracle(
        mi=[1.0, 0.9, 0.8],
        cmi_single={(1, 0): 0.2, (2, 0): 0.4},
    )
    result = greedy_select(oracle, "jmim", "max_min", 0.95, 5, 2)
    assert result.selected == [0]
    assert result.scores == [1.0]


def test_greedy_max_features_stops():
    oracle = FakeOracle(
        mi=[1.0, 0.9, 0.8, 0.7],
        mi_pair={(0, 1): 0.1, (0, 2): 0.2, (0, 3): 0.3},
    )
    result = greedy_select(oracle, "mrmr", "max_min", 1e-4, 2, 2)
    assert result.selected == [0, 1]
    assert len(result.scores) == 2


def test_greedy_cmim_pass_s_m():
    # cmim + pase distribuido: S_m = top-2 por MI = [0, 1] (fijo).
    # Ronda 1: 0. Ronda 2: cmi_set → x2 (0.6). Ronda 3: x1 (0.5).
    oracle = FakeOracle(
        mi=[1.0, 0.9, 0.8, 0.7],
        cmi_set={1: 0.5, 2: 0.6, 3: 0.4},
    )
    result = greedy_select(oracle, "cmim", "exact", 1e-4, 3, 2)
    assert result.selected == [0, 2, 1]
    assert result.scores == [1.0, 0.6, 0.5]
    # S_m fijo: el pase se consulta con [0, 1] en cada ronda.
    assert oracle.cmi_set_calls == [[0, 1], [0, 1]]


def test_greedy_cmim_max_min_routes_to_jmim():
    # cmim + max_min → jmim (criterios rápidos, sin pase CMIM).
    oracle = FakeOracle(
        mi=[1.0, 0.9, 0.8],
        cmi_single={(1, 0): 0.2, (2, 0): 0.4},
    )
    result = greedy_select(oracle, "cmim", "max_min", 1e-4, 2, 2)
    assert result.selected == [0, 2]
    assert oracle.cmi_set_calls == []


def test_greedy_invalid_criterion_raises():
    oracle = FakeOracle(mi=[1.0, 0.9])
    with pytest.raises(ValueError):
        greedy_select(oracle, "inexistente", "max_min", 1e-4, 2, 2)


def test_greedy_min_score_above_all_mi_selects_nothing():
    oracle = FakeOracle(mi=[0.5, 0.4])
    result = greedy_select(oracle, "jmim", "max_min", 1.0, 5, 2)
    assert result.selected == []
    assert result.scores == []


# ----------------------------------------------------------------------
# Unitarios: KsgOracle (numpy puro, semillas fijas).
# ----------------------------------------------------------------------


def test_ksg_oracle_mi_all_caching():
    rng = np.random.default_rng(0)
    n = 300
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    y = x0 + 0.1 * rng.normal(size=n)
    oracle = KsgOracle(np.column_stack([x0, x1]), y, k=10)
    mi = oracle.mi_all()
    assert mi[0] > mi[1]  # x0 informativa, x1 independiente.
    # mi_all cacheado: la segunda llamada devuelve los mismos valores.
    assert np.array_equal(oracle.mi_all(), mi)


def test_ksg_oracle_cmi_set_all_single_condition():
    rng = np.random.default_rng(1)
    n = 500
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    y = x0 + 0.1 * rng.normal(size=n)
    oracle = KsgOracle(np.column_stack([x0, x1]), y, k=10)
    out = oracle.cmi_set_all([0])
    # i=0: cond vacía → MI(X0;Y).
    assert out[0] == pytest.approx(oracle.mi_all()[0])
    # i=1: CMI(X1;Y|X0).
    assert out[1] == pytest.approx(oracle.cmi_single(1, 0))


def test_ksg_oracle_cmi_set_all_two_conditions():
    rng = np.random.default_rng(2)
    n = 500
    X = rng.normal(size=(n, 4))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.normal(size=n)
    oracle = KsgOracle(X, y, k=10)
    out = oracle.cmi_set_all([0, 1])
    # i=0: cond = (1,) → CMI(X0;Y|X1).
    assert out[0] == pytest.approx(oracle.cmi_single(0, 1))
    # i=1: cond = (0,) → CMI(X1;Y|X0).
    assert out[1] == pytest.approx(oracle.cmi_single(1, 0))
    # i=2, i=3: cond = (0, 1) → CMI con dos condicionantes (z de n×2).
    assert out[2] == pytest.approx(ksg_cmi(X[:, 2], y, X[:, [0, 1]], k=10))
    assert out[3] == pytest.approx(ksg_cmi(X[:, 3], y, X[:, [0, 1]], k=10))


def test_ksg_oracle_mi_pair_symmetric():
    rng = np.random.default_rng(3)
    n = 300
    X = rng.normal(size=(n, 2))
    oracle = KsgOracle(X, rng.normal(size=n), k=10)
    assert oracle.mi_pair(0, 1) == pytest.approx(oracle.mi_pair(1, 0))


# ----------------------------------------------------------------------
# Invariante de equivalencia (Spark): HistogramOracle.cmi_set_all ≡
# conditional_mi directo sobre el mismo subsample.
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("sparkmim-test-selection")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


def _make_df_eq(spark, n, seed):
    """x0/x1 informativas (y = (x0+2·x1) mod 4 ⇒ MI(x0;y)=MI(x1;y)=1 nat),
    x2/x3 ruido independiente (MI ≈ 0)."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 4, size=n)
    x1 = rng.integers(0, 4, size=n)
    x2 = rng.integers(0, 4, size=n)
    x3 = rng.integers(0, 4, size=n)
    y = (x0 + 2 * x1) % 4
    rows = list(zip(x0.tolist(), x1.tolist(), x2.tolist(), x3.tolist(), y.tolist()))
    return spark.createDataFrame(rows, ["x0", "x1", "x2", "x3", "y"])


@pytest.fixture(scope="module")
def eq_df(spark):
    return _make_df_eq(spark, n=2000, seed=42)


def test_histogram_oracle_cmi_set_matches_conditional_mi(eq_df):
    """El pase CMIM del oráculo (mapInPandas) da los mismos valores que el
    cálculo directo sobre los mismos conteos: ``conditional_mi`` sobre la
    triple del caché (1 condicionante) y ``conditional_mi_multi`` sobre la
    conjunta 4-vía reconstruida en numpy (2 condicionantes)."""
    config = SelectorConfig(
        target="y", screen_top_k=4, max_features=4, subsample=2000, seed=42,
        significance=None,
    )
    prepared = prepare(eq_df, config)
    schema = prepared.schema
    screen_result = screen(prepared.df_prep, schema, config)
    candidates = list(screen_result.candidates)
    K = len(candidates)
    assert K == 4

    feature_names = schema.feature_names()
    candidate_cols = [feature_names[i] for i in candidates]
    candidate_n_codes = [schema.n_codes()[i] for i in candidates]
    n_y = schema.target.n_codes

    # n = 2000 ≤ subsample → sin muestreo: las mismas filas.
    sub = prepared.df_prep
    rows_df = build_joint_tables(sub, candidate_cols, "y", candidate_n_codes, n_y)
    triples = dense_from_joints(rows_df, candidate_n_codes, n_y)
    univariate = {i: screen_result.tables[candidates[i]] for i in range(K)}
    cache = TableCache(
        n_codes=candidate_n_codes,
        n_y=n_y,
        univariate=univariate,
        triples=triples,
    )
    oracle = HistogramOracle(cache, sub, candidate_cols, "y", candidate_n_codes, n_y)

    # S_m = top-2 por MI (posiciones 0, 1: candidatas ordenadas por MI desc).
    mi = oracle.mi_all()
    s_m = [int(i) for i in np.argsort(-mi, kind="stable")[:2]]
    assert set(s_m) == {0, 1}

    out = oracle.cmi_set_all(s_m)

    # Cálculo directo de referencia:
    # - cond vacío → MI univariante.
    # - 1 condicionante → CMI desde la triple del caché.
    # - 2 condicionantes → conjunta 4-vía (n_i, n_y, n_c0, n_c1) reconstruida
    #   en numpy (mismas filas, mismos conteos que el pase mapInPandas).
    sub_pdf = sub.toPandas()
    y_codes = sub_pdf["y"].to_numpy().astype(np.int64)
    xs = {col: sub_pdf[col].to_numpy().astype(np.int64) for col in candidate_cols}

    for i in range(K):
        cond = [s for s in s_m if s != i]
        if len(cond) == 0:
            expected = mi[i]
        elif len(cond) == 1:
            j = cond[0]
            # La caché entrega la triple en la orientación de conditional_mi.
            expected = conditional_mi(cache.cmi_table(i, j))
        else:
            c0, c1 = cond
            packed = (
                (xs[candidate_cols[i]] * n_y + y_codes) * candidate_n_codes[c0]
                + xs[candidate_cols[c0]]
            ) * candidate_n_codes[c1] + xs[candidate_cols[c1]]
            t4 = np.bincount(
                packed,
                minlength=(
                    candidate_n_codes[i] * n_y
                    * candidate_n_codes[c0] * candidate_n_codes[c1]
                ),
            ).reshape(
                candidate_n_codes[i], n_y, candidate_n_codes[c0], candidate_n_codes[c1]
            )
            expected = conditional_mi_multi(t4)
        assert out[i] == pytest.approx(expected, abs=1e-9)
