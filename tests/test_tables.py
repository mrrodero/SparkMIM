"""Tests de los builders de tablas en pase único y de TableCache.

Compara el pase ``mapInPandas`` contra un cómputo de referencia directo
(numpy) sobre un DataFrame pequeño.
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession

from sparkmim.tables import (
    TableCache,
    build_joint_tables,
    build_screening_tables,
    dense_from_joints,
    dense_from_screening,
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
