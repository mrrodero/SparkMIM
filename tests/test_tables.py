"""Tests de los builders de tablas en pase único y de TableCache.

Compara el pase ``mapInPandas`` contra un cómputo de referencia directo
(numpy) sobre un DataFrame pequeño.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim.criteria import criterion_score
from sparkmim.info.entropy import conditional_mi, mutual_information
from sparkmim.oracles import HistogramOracle
from sparkmim.screen import ScreenResult
from sparkmim.tables import (
    TableCache,
    build_joint_tables,
    build_screening_tables,
    dense_from_joints,
    dense_from_screening,
    joint_counts,
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


# Datos deterministas: 24 filas, 3 features + target (códigos enteros).
ROWS = [
    (0, 0, 0, 0),
    (0, 0, 1, 1),
    (0, 1, 2, 0),
    (1, 0, 3, 1),
    (1, 1, 0, 0),
    (1, 1, 1, 1),
    (2, 0, 2, 1),
    (2, 1, 3, 0),
    (2, 1, 0, 1),
    (0, 1, 1, 0),
    (1, 0, 2, 1),
    (2, 0, 0, 0),
    (0, 0, 3, 1),
    (1, 1, 2, 0),
    (2, 1, 1, 1),
    (0, 1, 0, 0),
    (1, 0, 1, 1),
    (2, 0, 3, 0),
    (0, 0, 2, 1),
    (1, 1, 3, 0),
    (2, 1, 0, 1),
    (0, 1, 3, 0),
    (1, 0, 0, 1),
    (2, 0, 1, 0),
]
FEAT_COLS = ["x0", "x1", "x2"]
N_X = [3, 2, 4]
N_Y = 2


def _make_df(spark):
    # Lista de nombres: createDataFrame infiere los tipos (int -> long).
    return spark.createDataFrame(ROWS, FEAT_COLS + ["y"])


def _ref_uni(x, y, n_x, n_y):
    c = np.zeros((n_x, n_y), dtype=np.int64)
    for a, b in zip(x, y):
        c[a, b] += 1
    return c


def _ref_triple(xi, xj, y, ni, nj, ny):
    c = np.zeros((ni, nj, ny), dtype=np.int64)
    for a, b, z in zip(xi, xj, y):
        c[a, b, z] += 1
    return c


def _ref_counts(cols, sizes):
    c = np.zeros(tuple(sizes), dtype=np.int64)
    for row in zip(*cols):
        c[row] += 1
    return c


# --- El núcleo joint_counts, contra conteos plantados (2, 3 y 4 columnas) ---


def test_joint_counts_2_columns_matches_planted_table():
    cols = [np.array([r[i] for r in ROWS]) for i in range(3)]
    y = np.array([r[3] for r in ROWS])
    for fid in range(3):
        np.testing.assert_array_equal(
            joint_counts([cols[fid], y], (N_X[fid], N_Y)),
            _ref_counts([cols[fid], y], (N_X[fid], N_Y)),
        )


def test_joint_counts_3_columns_matches_planted_table():
    cols = [np.array([r[i] for r in ROWS]) for i in range(3)]
    y = np.array([r[3] for r in ROWS])
    for i in range(3):
        for j in range(i + 1, 3):
            sizes = (N_X[i], N_X[j], N_Y)
            np.testing.assert_array_equal(
                joint_counts([cols[i], cols[j], y], sizes),
                _ref_counts([cols[i], cols[j], y], sizes),
            )


def test_joint_counts_4_columns_matches_planted_table():
    # (X_i, Y, Z1, Z2): la forma sale del orden de las columnas.
    rng = np.random.default_rng(7)
    n = 500
    sizes = (3, 2, 4, 5)
    cols = [rng.integers(0, s, size=n) for s in sizes]
    np.testing.assert_array_equal(joint_counts(cols, sizes), _ref_counts(cols, sizes))


def test_joint_counts_shape_follows_column_order():
    # El mismo par de columnas en otro orden da la tabla transpuesta.
    x = np.array([r[0] for r in ROWS])
    y = np.array([r[3] for r in ROWS])
    np.testing.assert_array_equal(
        joint_counts([y, x], (N_Y, N_X[0])),
        joint_counts([x, y], (N_X[0], N_Y)).T,
    )


def test_screening_matches_reference(spark):
    df = _make_df(spark)
    agg = build_screening_tables(df, FEAT_COLS, "y", N_X, N_Y)
    tables = dense_from_screening(agg, N_X, N_Y)
    xs = [np.array([r[i] for r in ROWS]) for i in range(3)]
    ys = np.array([r[3] for r in ROWS])
    for fid in range(3):
        expected = _ref_uni(xs[fid], ys, N_X[fid], N_Y)
        np.testing.assert_array_equal(tables[fid], expected)
    # Las sumas de cada tabla deben ser el nº de filas.
    assert all(t.sum() == len(ROWS) for t in tables.values())


def test_screening_sparse_only_nonzero(spark):
    df = _make_df(spark)
    agg = build_screening_tables(df, FEAT_COLS, "y", N_X, N_Y)
    rows = agg.collect()
    # r["count"]: el atributo r.count chocaría con tuple.count.
    assert all(r["count"] > 0 for r in rows)
    # Cada (fid, x, y) aparece una única vez (agregado).
    keys = [(r.fid, r.x, r.y) for r in rows]
    assert len(keys) == len(set(keys))


def test_joint_tables_match_reference(spark):
    df = _make_df(spark)
    rows_df = build_joint_tables(df, FEAT_COLS, "y", N_X, N_Y)
    tables = dense_from_joints(rows_df, N_X, N_Y)
    xs = [np.array([r[i] for r in ROWS]) for i in range(3)]
    ys = np.array([r[3] for r in ROWS])
    for i in range(3):
        for j in range(i + 1, 3):
            expected = _ref_triple(xs[i], xs[j], ys, N_X[i], N_X[j], N_Y)
            np.testing.assert_array_equal(tables[(i, j)], expected)
    # Todas las tablas presentes.
    assert set(tables.keys()) == {(0, 1), (0, 2), (1, 2)}


def test_joint_partition_sum_consistency(spark):
    # La suma de cada triple debe ser el nº de filas (todas las celdas cuentan).
    df = _make_df(spark)
    rows_df = build_joint_tables(df, FEAT_COLS, "y", N_X, N_Y)
    tables = dense_from_joints(rows_df, N_X, N_Y)
    for key, t in tables.items():
        assert t.sum() == len(ROWS)


def test_tablecache_pair_is_marginal(spark):
    df = _make_df(spark)
    rows_df = build_joint_tables(df, FEAT_COLS, "y", N_X, N_Y)
    tables = dense_from_joints(rows_df, N_X, N_Y)
    cache = TableCache(N_X, N_Y, triples=tables)
    xs = [np.array([r[i] for r in ROWS]) for i in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            # Marginal sobre Y de la triple = crosstab(Xi, Xj).
            expected = _ref_uni(xs[i], xs[j], N_X[i], N_X[j])
            np.testing.assert_array_equal(cache.pair(i, j), expected)
            # Simetría de acceso.
            np.testing.assert_array_equal(cache.triple(j, i), cache.triple(i, j))


def test_tablecache_cells(spark):
    df = _make_df(spark)
    rows_df = build_joint_tables(df, FEAT_COLS, "y", N_X, N_Y)
    tables = dense_from_joints(rows_df, N_X, N_Y)
    cache = TableCache(N_X, N_Y, triples=tables)
    # pares: 3*2 + 3*4 + 2*4 = 6+12+8 = 26; triples: 26 * 2 = 52; total 78.
    assert cache.cells() == (3 * 2 + 3 * 4 + 2 * 4) * (1 + N_Y)


def test_tablecache_cmi_table_orientation():
    # Triples canónicas (n_min, n_max, n_y) manuales: sin Spark.
    # cmi_table debe devolver (n_i, n_y, n_j) para ambos órdenes de (i, j).
    n_codes = [3, 2, 2]
    n_y = 2
    triples = {
        (0, 1): np.arange(3 * 2 * 2, dtype=np.int64).reshape(3, 2, 2) + 1,
        (0, 2): np.arange(3 * 2 * 2, dtype=np.int64).reshape(3, 2, 2) + 7,
        (1, 2): np.arange(2 * 2 * 2, dtype=np.int64).reshape(2, 2, 2) + 13,
    }
    cache = TableCache(n_codes, n_y, triples=triples)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            canon = triples[(min(i, j), max(i, j))]
            expected = np.zeros((n_codes[i], n_y, n_codes[j]), dtype=np.int64)
            for a in range(n_codes[i]):
                for b in range(n_y):
                    for c in range(n_codes[j]):
                        # expected[a, b, c] = count(X_i=a, Y=b, X_j=c).
                        expected[a, b, c] = canon[a, c, b] if i < j else canon[c, a, b]
            assert cache.cmi_table(i, j).shape == (n_codes[i], n_y, n_codes[j])
            np.testing.assert_array_equal(cache.cmi_table(i, j), expected)


def test_criterion_score_on_handbuilt_cache():
    # criterion_score sobre una caché manual: sin Spark ni mapInPandas.
    n_codes = [2, 3, 2]
    n_y = 2
    rng = np.random.default_rng(7)
    triples = {
        (i, j): rng.integers(0, 9, size=(n_codes[i], n_codes[j], n_y)).astype(np.int64)
        for i in range(3)
        for j in range(i + 1, 3)
    }
    univariate = {
        i: rng.integers(0, 9, size=(n_codes[i], n_y)).astype(np.int64) for i in range(3)
    }
    cache = TableCache(n_codes, n_y, univariate=univariate, triples=triples)
    oracle = HistogramOracle(cache, None, ["a", "b", "c"], "y", n_codes, n_y)
    mi = oracle.mi_all()
    # Referencia directa: funciones de entropía sobre los accesores de la caché.
    for x in range(3):
        assert mi[x] == pytest.approx(mutual_information(cache.uni(x)), abs=1e-12)
        for s in range(3):
            if x == s:
                continue
            assert oracle.cmi_single(x, s) == pytest.approx(
                conditional_mi(cache.cmi_table(x, s)), abs=1e-12
            )
    # jmim con S = [0, 1] para x = 2: min_s CMI(x; Y | s).
    score = criterion_score(2, [0, 1], oracle, mi, "jmim")
    assert score == pytest.approx(
        min(conditional_mi(cache.cmi_table(2, s)) for s in (0, 1)), abs=1e-12
    )
    # mim con S = [1]: MI(x; Y) - MI(x; s).
    score = criterion_score(2, [1], oracle, mi, "mim")
    assert score == pytest.approx(
        mutual_information(cache.uni(2)) - mutual_information(cache.pair(2, 1)),
        abs=1e-12,
    )
    # Ronda 1 (S vacío): MI univariante, cualquiera sea el criterio.
    assert criterion_score(0, [], oracle, mi, "jmim") == pytest.approx(mi[0], abs=1e-12)


# --- La factory: traducción global→posición y validación del cableado ---


def _screen_result(candidates, tables):
    """ScreenResult mínimo para la factory (sin Spark, sin significancia)."""
    return ScreenResult(
        mi=np.zeros(len(tables)),
        pvalues=np.ones(len(tables)),
        significant=np.ones(len(tables), dtype=bool),
        candidates=list(candidates),
        tables=dict(tables),
        n_rows=int(next(iter(tables.values())).sum()),
    )


# Candidatas globales [3, 0, 4] → posiciones de Candidate 0, 1, 2.
_F_FIDS = [3, 0, 4]
_F_TABLES = {
    3: np.full((3, 2), 2, dtype=np.int64),
    0: np.full((2, 2), 1, dtype=np.int64),
    4: np.full((4, 2), 3, dtype=np.int64),
}
_F_TRIPLES = {
    (0, 1): np.full((3, 2, 2), 5, dtype=np.int64),
    (0, 2): np.full((3, 4, 2), 6, dtype=np.int64),
    (1, 2): np.full((2, 4, 2), 7, dtype=np.int64),
}


def test_from_screening_translates_global_indices_to_positions():
    cache = TableCache.from_screening(_screen_result(_F_FIDS, _F_TABLES), _F_FIDS, _F_TRIPLES)
    # n_codes y n_y salen de las formas de las tablas de screening, en el orden
    # de las candidatas.
    assert cache.n_codes == [3, 2, 4]
    assert cache.n_y == 2
    # uni(i) es la tabla del fid global candidates[i].
    for i, fid in enumerate(_F_FIDS):
        np.testing.assert_array_equal(cache.uni(i), _F_TABLES[fid])
    # El acceso por posición sigue siendo canónico.
    np.testing.assert_array_equal(cache.triple(1, 0), _F_TRIPLES[(0, 1)])
    np.testing.assert_array_equal(cache.triple(2, 0), _F_TRIPLES[(0, 2)])


def test_from_screening_rejects_non_canonical_key():
    bad = {(1, 0): _F_TRIPLES[(0, 1)], (0, 2): _F_TRIPLES[(0, 2)], (1, 2): _F_TRIPLES[(1, 2)]}
    with pytest.raises(ValueError, match="canónico"):
        TableCache.from_screening(_screen_result(_F_FIDS, _F_TABLES), _F_FIDS, bad)


def test_from_screening_rejects_key_out_of_position_range():
    # Clave en el espacio global (fid 4 y 3), no en posiciones 0..2.
    bad = {(3, 4): np.full((3, 4, 2), 1, dtype=np.int64)}
    with pytest.raises(ValueError, match="canónico"):
        TableCache.from_screening(_screen_result(_F_FIDS, _F_TABLES), _F_FIDS, bad)


def test_from_screening_rejects_missing_pair():
    bad = {(0, 1): _F_TRIPLES[(0, 1)], (0, 2): _F_TRIPLES[(0, 2)]}
    with pytest.raises(ValueError, match="falta el par"):
        TableCache.from_screening(_screen_result(_F_FIDS, _F_TABLES), _F_FIDS, bad)


def test_from_screening_rejects_shape_mismatch():
    # (0, 2) con los ejes de la tabla global (4, 3, 2) en vez de (3, 4, 2).
    bad = dict(_F_TRIPLES)
    bad[(0, 2)] = np.full((4, 3, 2), 6, dtype=np.int64)
    with pytest.raises(ValueError, match="se espera"):
        TableCache.from_screening(_screen_result(_F_FIDS, _F_TABLES), _F_FIDS, bad)


def test_from_screening_rejects_missing_screening_table():
    tables = {k: v for k, v in _F_TABLES.items() if k != 0}
    with pytest.raises(ValueError, match="tabla de screening"):
        TableCache.from_screening(_screen_result(_F_FIDS, tables), _F_FIDS, _F_TRIPLES)


def test_from_screening_accepts_single_candidate():
    # K = 1: no hay pares; la caché se arma solo con la univariante.
    cache = TableCache.from_screening(
        _screen_result([4], {4: _F_TABLES[4]}), [4], {}
    )
    assert cache.n_codes == [4]
    assert cache.n_y == 2
    np.testing.assert_array_equal(cache.uni(0), _F_TABLES[4])
