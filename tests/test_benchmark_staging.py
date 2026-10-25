"""Tests del staging de ``sparkmim.benchmarks.scale``.

Cubren el helper de slicing/batching (presupuesto de bytes) y un round-trip
pequeño: los lotes escritos en parquet (varios ``append``) leídos de vuelta
deben devolver **exactamente** los valores de entrada. Parquet no garantiza
el orden de las filas, así que la comparación es por multiconjunto de filas
completas (sin entregar ordinal de fila como feature: solo para el test).

En Windows, Spark requiere ``hadoop.dll`` + ``winutils.exe`` (``HADOOP_HOME``)
para operar el filesystem local; si no están, las pruebas con sesión se
saltan con un motivo claro (``python tools/fetch_winutils.py``).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from sparkmim.benchmarks import generate
from sparkmim.benchmarks import scale as scale_mod

_REPO = Path(__file__).resolve().parent.parent


def _spark_local_fs_available() -> bool:
    """Spark necesita hadoop.dll/winutils en Windows para el FS local."""
    if os.name != "nt":
        return True
    home = os.environ.get("HADOOP_HOME") or str(_REPO / "winutils")
    return (Path(home) / "bin" / "winutils.exe").is_file()


# ---------------------------------------------------------------------------
# Batching: presupuesto de bytes
# ---------------------------------------------------------------------------

def test_rows_per_batch_default_16mib():
    # N=25: 8 B × 26 valores = 208 B/fila; 16 MiB raw -> ~80k filas.
    assert scale_mod._rows_per_batch(25, scale_mod.DEFAULT_BATCH_BYTES) == (
        scale_mod.DEFAULT_BATCH_BYTES
    ) // 208
    # y está en la banda conservadora recomendada (80k-100k filas).
    assert 80_000 <= scale_mod._rows_per_batch(25, scale_mod.DEFAULT_BATCH_BYTES) <= 100_000


def test_rows_per_batch_floor_one_row():
    # Presupuesto menor que una fila completa -> 1 fila (nunca 0).
    assert scale_mod._rows_per_batch(1000, 16) == 1
    # Exacto divisor: sin residuo (N=3 -> 4 valores x 8 B = 32 B/fila).
    assert scale_mod._rows_per_batch(3, 128) == 4


def test_rows_per_batch_scales_with_n_features():
    # Más features -> menos filas por lote con el mismo presupuesto.
    assert scale_mod._rows_per_batch(1000, 16 * 1024 * 1024) < scale_mod._rows_per_batch(
        25, 16 * 1024 * 1024
    )


def test_default_staging_path_unique_under_dbfs_tmp():
    a, b = scale_mod.default_staging_path(), scale_mod.default_staging_path()
    assert a.startswith("dbfs:/tmp/sparkmim-benchmark/")
    assert a != b


def test_schema_doubles_and_long_target():
    from pyspark.sql.types import DoubleType, LongType, StructType

    schema = scale_mod._benchmark_schema(["x0", "x1"])
    assert isinstance(schema, StructType)
    assert [f.name for f in schema.fields] == ["x0", "x1", "y"]
    assert all(isinstance(f.dataType, DoubleType) for f in schema.fields[:-1])
    assert isinstance(schema.fields[-1].dataType, LongType)
    assert not schema.fields[-1].nullable


# ---------------------------------------------------------------------------
# Round-trip pequeño por lotes (Spark local)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("sparkmim-test-benchmark-staging")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


def _row_tuples(data, cols):
    """Multiconjunto de filas completas como tuplas ordenables."""
    return sorted(zip(*[data[c] for c in cols]))


@pytest.mark.skipif(
    not _spark_local_fs_available(),
    reason="Falta winutils/hadoop.dll en Windows: python tools/fetch_winutils.py",
)
def test_staged_roundtrip_exact_multiple_batches(spark, tmp_path):
    # Presupuesto de 256 KiB raw: con N=30 (248 B/fila) -> ~1k filas/lote,
    # de modo que 5000 filas fuerzan varios append parquet.
    data = generate(n=5000, n_features=30, n_informative=10, n_redundant=5, seed=7)
    staging = str(tmp_path / "staging")
    df = scale_mod._staged_df(spark, data, staging, batch_bytes=256 * 1024)

    cols = [c for c in data if c != "y"]
    assert df.count() == 5000
    # Tipos exactos tras el round-trip (LongType se imprime como
    # ``bigint`` en Spark; el schema de escritura es LongType, ver arriba).
    types = dict(df.dtypes)
    assert all(types[c] == "double" for c in cols)
    assert types["y"] == "bigint"

    got = df.toPandas()
    # Parquet no garantiza el orden de filas: comparar como multiconjunto
    # de filas completas (orden por todas las columnas; no se entrega
    # ordinal de fila como feature).
    expected = _row_tuples(data, cols + ["y"])
    actual = _row_tuples(got, cols + ["y"])
    assert len(actual) == len(expected)
    assert actual == expected


@pytest.mark.skipif(
    not _spark_local_fs_available(),
    reason="Falta winutils/hadoop.dll en Windows: python tools/fetch_winutils.py",
)
def test_staged_single_batch_spans_all_rows(spark, tmp_path):
    # Presupuesto enorme -> un solo lote; ninguna fila se pierde.
    data = generate(n=100, n_features=25, seed=3)
    staging = str(tmp_path / "staging1")
    df = scale_mod._staged_df(spark, data, staging, batch_bytes=16 * 1024 * 1024)

    cols = [c for c in data if c != "y"]
    assert df.count() == 100
    got = df.toPandas()
    assert set(np.unique(got["y"])) == {0, 1}
    expected = _row_tuples(data, cols + ["y"])
    actual = _row_tuples(got, cols + ["y"])
    assert actual == expected
