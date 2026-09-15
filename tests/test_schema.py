"""Tests de detección de esquema (numérico/categórico) y overrides."""

import pytest
from pyspark.sql import SparkSession

from sparkmim.config import SelectorConfig
from sparkmim.schema import build_schema, detect_kinds, resolve_bins


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("sparkmim-test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _cfg(**kw):
    base = dict(target="y")
    base.update(kw)
    return SelectorConfig(**base)


def test_detect_kinds_by_dtype(spark):
    # Incluye la target 'y' (excluida de kinds) para que detect_kinds no falle.
    df = spark.createDataFrame(
        [(1.0, 2, "a", True, 0), (3.0, 4, "b", False, 1)],
        "d: double, i: int, s: string, b: boolean, y: int",
    )
    kinds = detect_kinds(df, _cfg())
    assert kinds == {"d": "numeric", "i": "numeric", "s": "categorical", "b": "categorical"}


def test_target_not_in_features(spark):
    df = spark.createDataFrame([(1.0, 0)], "x: double, y: int")
    schema = build_schema(df, _cfg())
    assert schema.feature_names() == ["x"]
    assert schema.target.name == "y"
    assert schema.target.is_target


def test_override_numeric(spark):
    # Una columna string forzada a numérica (valores numéricos como texto).
    df = spark.createDataFrame([("0.5", 0), ("1.5", 1)], "x: string, y: int")
    schema = build_schema(df, _cfg(numeric_features=["x"]))
    assert schema.features[0].kind == "numeric"


def test_override_categorical(spark):
    df = spark.createDataFrame([(1.0, 0), (2.0, 1)], "x: double, y: int")
    schema = build_schema(df, _cfg(categorical_features=["x"]))
    assert schema.features[0].kind == "categorical"


def test_override_overlap_raises(spark):
    df = spark.createDataFrame([(1.0, 0)], "x: double, y: int")
    with pytest.raises(ValueError, match="ambas listas"):
        build_schema(df, _cfg(numeric_features=["x"], categorical_features=["x"]))


def test_missing_target_raises(spark):
    df = spark.createDataFrame([(1.0, 0)], "x: double, z: int")
    with pytest.raises(ValueError, match="no existe"):
        build_schema(df, _cfg())


def test_unsupported_dtype_raises(spark):
    from pyspark.sql.types import (
        ArrayType,
        DoubleType,
        IntegerType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("arr", ArrayType(DoubleType())),
            StructField("y", IntegerType()),
        ]
    )
    df = spark.createDataFrame([[ [1.0], 0 ]], schema)
    with pytest.raises(TypeError, match="no soportado"):
        build_schema(df, _cfg())


def test_resolve_bins_auto():
    assert resolve_bins(10**5, "auto") == 17  # round(log2 1e5) = 17
    assert resolve_bins(10**6, "auto") == 20  # clamp superior a 20
    assert resolve_bins(1000, "auto") == 10  # round(log2 1000) = 10
    assert resolve_bins(10, "auto") == 4  # clamp inferior a 4
    assert resolve_bins(10**6, 10) == 10  # int directo
