"""Detección de esquema (numérico/categórico) y specs de columnas.

La detección es por dtype de Spark; el usuario puede forzar el tipo con
``numeric_features`` / ``categorical_features`` en la config.

Los ``FeatureSpec`` se rellenan en la etapa 0 (``preprocess.py``) con el nº de
códigos, fronteras de bins y mapas de categorías.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DateType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)

__all__ = ["FeatureSpec", "Schema", "detect_kinds", "resolve_bins", "build_schema"]

_NUMERIC_TYPES = (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
)


def _is_numeric(dt: DataType) -> bool:
    return isinstance(dt, _NUMERIC_TYPES)


def _is_categorical(dt: DataType) -> bool:
    return isinstance(dt, (StringType, BooleanType, DateType, TimestampType))


@dataclass
class FeatureSpec:
    """Especificación de una columna (feature o target).

    - ``kind``: ``"numeric"`` | ``"categorical"``.
    - ``dtype``: nombre del tipo Spark (``"double"``, ``"string"``, ...).
    - ``n_codes``: nº de códigos tras preprocesado (se rellena en etapa 0).
      Incluye el código de missing si ``missing="category"``.
    - ``bin_edges``: fronteras de bins (numéricas, se rellena en etapa 0).
    - ``category_codes``: mapa valor -> código (categóricas, etapa 0).
    - ``missing_code``: código dedicado al missing (si aplica).
    """

    name: str
    kind: str
    dtype: str = ""
    n_codes: Optional[int] = None
    bin_edges: Optional[tuple] = None
    category_codes: Optional[Dict] = None
    missing_code: Optional[int] = None
    has_other: bool = False
    is_target: bool = False

    def ready(self) -> bool:
        return self.n_codes is not None and self.n_codes >= 1


@dataclass
class Schema:
    """Esquema completo: features + target."""

    features: List[FeatureSpec] = field(default_factory=list)
    target: Optional[FeatureSpec] = None

    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    def n_codes(self) -> List[int]:
        return [f.n_codes for f in self.features]

    def ready(self) -> bool:
        return self.target is not None and self.target.ready() and all(
            f.ready() for f in self.features
        )


def detect_kinds(df: DataFrame, config) -> Dict[str, str]:
    """Detecta ``{"col": "numeric"|"categorical"}`` por dtype, con overrides."""
    kinds: Dict[str, str] = {}
    for field_ in df.schema.fields:
        name = field_.name
        if name == config.target:
            continue
        if config.numeric_features and name in config.numeric_features:
            kinds[name] = "numeric"
        elif config.categorical_features and name in config.categorical_features:
            kinds[name] = "categorical"
        elif _is_numeric(field_.dataType):
            kinds[name] = "numeric"
        elif _is_categorical(field_.dataType):
            kinds[name] = "categorical"
        else:
            raise TypeError(
                f"Columna '{name}' con tipo no soportado: {field_.dataType}. "
                "Soportados: numéricos, string, boolean, date, timestamp."
            )
    # Target: numérico o categórico (binaria/multiclase como códigos o strings).
    target_field = next((f for f in df.schema.fields if f.name == config.target), None)
    if target_field is None:
        raise ValueError(
            f"La columna target '{config.target}' no existe en el DataFrame"
        )
    if not (_is_numeric(target_field.dataType) or _is_categorical(target_field.dataType)):
        raise TypeError(
            f"Target '{config.target}' con tipo no soportado: {target_field.dataType}"
        )
    return kinds


def resolve_bins(n_rows: int, bins) -> int:
    """Resuelve ``bins``: int directo o ``"auto"`` = clamp(round(log2 n), 4, 20)."""
    if bins == "auto":
        return int(min(20, max(4, round(math.log2(max(n_rows, 2))))))
    return int(bins)


def build_schema(df: DataFrame, config) -> Schema:
    """Construye el ``Schema`` (solo tipos) a partir del DataFrame y la config.

    El orden de las features es el del schema del DataFrame (sin el target).
    """
    kinds = detect_kinds(df, config)
    features: List[FeatureSpec] = []
    for field_ in df.schema.fields:
        name = field_.name
        if name == config.target:
            continue
        features.append(
            FeatureSpec(name=name, kind=kinds[name], dtype=field_.dataType.typeName())
        )
    # Target spec (tipo; el binning del target se resuelve en preprocess).
    target_field = next(f for f in df.schema.fields if f.name == config.target)
    target_kind = "numeric" if _is_numeric(target_field.dataType) else "categorical"
    target_spec = FeatureSpec(
        name=config.target, kind=target_kind, dtype=target_field.dataType.typeName(),
        is_target=True,
    )
    return Schema(features=features, target=target_spec)
