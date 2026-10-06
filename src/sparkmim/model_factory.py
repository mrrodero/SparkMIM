"""Fábrica de modelos: costura entre la lógica de evaluación y los backends.

La lógica de evaluación (``evaluate.py``) no entrena modelos directamente:
resuelve la task declarada, codifica las columnas categóricas, ensambla el
vector ``features`` y pasa el df vectorizado a una ``ModelFactory``, que
entrena el modelo y devuelve el df de predicciones. Cada backend es un
adaptador (docs/DESIGN.md §2, §7):

- ``GbtFactory``: ``GradientBoostedClassifier``/``GradientBoostedRegressor``
  de spark.ml (default, sin extras).
- ``XgboostFactory``: XGBoost (extra opcional, import perezoso).
- ``LightgbmFactory``: LightGBM (extra opcional, import perezoso).

Para tests, un adaptador trivial determinista (sin entrenamiento real)
permite ejercitar la lógica de evaluación sin Spark ML; se define en
``tests/test_evaluate.py``.

Coste: 1 entrenamiento por llamada a ``train``.
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

__all__ = [
    "ModelFactory",
    "GbtFactory",
    "XgboostFactory",
    "LightgbmFactory",
    "make_model_factory",
]


class ModelFactory(Protocol):
    """Fábrica de modelos para la evaluación (costura, docs/DESIGN.md §2).

    Dado el df vectorizado (columna ``features`` y la columna label) y la
    task resuelta (``"classifier_binary"`` | ``"classifier_multiclass"`` |
    ``"continuous"``), entrena un modelo y devuelve el df con las
    predicciones: ``probability`` (``ArrayType(DoubleType)``, una columna
    por clase) en clasificación y ``prediction`` (``DoubleType``) en
    continua.

    Coste: 1 entrenamiento por llamada.
    """

    name: str

    def train(self, df_vec: "DataFrame", task: str, label_col: str) -> "DataFrame":
        """Entrena y devuelve el df con las columnas de predicción."""
        ...


class GbtFactory:
    """Adaptador GBT (spark.ml, default)."""

    name = "gbt"

    def train(self, df_vec, task, label_col):
        if task in ("classifier_binary", "classifier_multiclass"):
            from pyspark.ml.classification import GBTClassifier

            est = GBTClassifier(featuresCol="features", labelCol=label_col)
        else:
            from pyspark.ml.regression import GBTRegressor

            est = GBTRegressor(featuresCol="features", labelCol=label_col)
        return est.fit(df_vec).transform(df_vec)


class XgboostFactory:
    """Adaptador XGBoost (extra opcional; import perezoso)."""

    name = "xgboost"

    def train(self, df_vec, task, label_col):
        if task in ("classifier_binary", "classifier_multiclass"):
            from pyspark.ml.classification import XGBoostClassifier

            est = XGBoostClassifier(featuresCol="features", labelCol=label_col)
        else:
            from pyspark.ml.regression import XGBoostRegressor

            est = XGBoostRegressor(featuresCol="features", labelCol=label_col)
        return est.fit(df_vec).transform(df_vec)


class LightgbmFactory:
    """Adaptador LightGBM (extra opcional; import perezoso)."""

    name = "lightgbm"

    def train(self, df_vec, task, label_col):
        if task in ("classifier_binary", "classifier_multiclass"):
            from pyspark.ml.classification import LightGBMClassifier

            est = LightGBMClassifier(featuresCol="features", labelCol=label_col)
        else:
            from pyspark.ml.regression import LightGBMRegressor

            est = LightGBMRegressor(featuresCol="features", labelCol=label_col)
        return est.fit(df_vec).transform(df_vec)


def make_model_factory(model_name: str) -> ModelFactory:
    """Devuelve el adaptador de backend para ``model_name``.

    ``"gbt"`` | ``"xgboost"`` | ``"lightgbm"``.
    """
    if model_name == "gbt":
        return GbtFactory()
    if model_name == "xgboost":
        return XgboostFactory()
    if model_name == "lightgbm":
        return LightgbmFactory()
    raise ValueError(
        f"model_name debe ser 'gbt', 'xgboost' o 'lightgbm' (recibido {model_name!r})"
    )
