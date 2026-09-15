"""Evaluación agnóstica al modelo (Hito 6).

``report`` entrena un clasificador/regresor sobre las features seleccionadas
(comparado con el baseline de todas las features) y devuelve métricas y la
curva de eficiencia. Sin sklearn:

- ``gbt`` (default): ``GradientBoostedClassifier``/``GradientBoostedRegressor``
  de spark.ml.
- ``xgboost`` / ``lightgbm``: extras opcionales (import perezoso).

AUC multiclase (macro, one-vs-rest) y R² se calculan en el driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, FloatType, StringType

from scipy.stats import rankdata

__all__ = [
    "EvaluationReport",
    "auc_binary",
    "auc_macro",
    "r2_score",
    "report",
    "efficiency_curve",
    "train_and_evaluate",
]


# ----------------------------------------------------------------------
# Métricas en driver.
# ----------------------------------------------------------------------
def auc_binary(y_true, y_prob) -> float:
    """AUC binaria (Mann-Whitney U) con empates promediados.

    ``y_true``: etiquetas 0/1. ``y_prob``: probabilidad de la clase 1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rango ascendente (1 = menor prob); con empates promediados. La fórmula
    # de Mann-Whitney usa el rango ascendente.
    ranks = rankdata(y_prob, method="average")
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_macro(y_true, y_prob_matrix) -> float:
    """AUC macro (one-vs-rest) para multiclase.

    ``y_true``: etiquetas (0..K-1). ``y_prob_matrix``: (K, n) prob de cada clase.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob_matrix = np.asarray(y_prob_matrix, dtype=float)
    classes = np.unique(y_true)
    aucs = []
    for c in classes:
        y_binary = (y_true == c).astype(int)
        aucs.append(auc_binary(y_binary, y_prob_matrix[c]))
    return float(np.mean(aucs))


def r2_score(y_true, y_pred) -> float:
    """R² (coeficiente de determinación)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ----------------------------------------------------------------------
# Preparación de features/target (codificación de categóricas).
# ----------------------------------------------------------------------
def _detect_task(df: DataFrame, target_col: str) -> str:
    """'classification' | 'regression' según el dtype del target."""
    dtype = df.schema[target_col].dataType
    if isinstance(dtype, StringType):
        return "classification"
    if isinstance(dtype, (FloatType, DoubleType)):
        return "regression"
    # Entero: clasificación si pocas clases distintas.
    n_unique = int(df.select(target_col).distinct().count())
    return "classification" if n_unique <= 20 else "regression"


def _encode_columns(df: DataFrame, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
    """Codifica columnas categóricas (string) a enteros; numéricas sin tocar."""
    from pyspark.ml.feature import StringIndexer

    df_enc = df
    out_cols: List[str] = []
    for col in cols:
        dtype = df.schema[col].dataType
        if isinstance(dtype, StringType):
            enc_col = f"{col}__enc"
            indexer = StringIndexer(inputCol=col, outputCol=enc_col, handleInvalid="keep")
            df_enc = indexer.fit(df_enc).transform(df_enc)
            out_cols.append(enc_col)
        else:
            out_cols.append(col)
    return df_enc, out_cols


# ----------------------------------------------------------------------
# Entrenamiento (agnóstico al modelo).
# ----------------------------------------------------------------------
def _train_model(
    df: DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    model_name: str,
    task: str,
):
    """Entrena el modelo y devuelve ``(model, assembler, df_enc)``."""
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.regression import GBTRegressor

    df_enc, enc_cols = _encode_columns(df, feature_cols)
    assembler = VectorAssembler(inputCols=enc_cols, outputCol="features")
    df_vec = assembler.transform(df_enc)

    if model_name == "gbt":
        if task == "classification":
            est = GBTClassifier(featuresCol="features", labelCol=label_col)
        else:
            est = GBTRegressor(featuresCol="features", labelCol=label_col)
    elif model_name == "xgboost":
        est = _xgboost_estimator(task, label_col)
    elif model_name == "lightgbm":
        est = _lightgbm_estimator(task, label_col)
    else:
        raise ValueError(f"model_name debe ser 'gbt', 'xgboost' o 'lightgbm' (recibido {model_name!r})")

    model = est.fit(df_vec)
    return model, assembler, df_enc


def _xgboost_estimator(task: str, label_col: str):
    if task == "classification":
        from pyspark.ml.classification import XGBoostClassifier  # type: ignore
        return XGBoostClassifier(featuresCol="features", labelCol=label_col)
    from pyspark.ml.regression import XGBoostRegressor  # type: ignore
    return XGBoostRegressor(featuresCol="features", labelCol=label_col)


def _lightgbm_estimator(task: str, label_col: str):
    if task == "classification":
        from pyspark.ml.classification import LightGBMClassifier  # type: ignore
        return LightGBMClassifier(featuresCol="features", labelCol=label_col)
    from pyspark.ml.regression import LightGBMRegressor  # type: ignore
    return LightGBMRegressor(featuresCol="features", labelCol=label_col)


# ----------------------------------------------------------------------
# Entrenamiento + evaluación.
# ----------------------------------------------------------------------
def train_and_evaluate(
    df: DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    model_name: str = "gbt",
    task: str | None = None,
) -> float:
    """Entrena sobre ``feature_cols`` y devuelve AUC (cl.) o R² (reg.)."""
    if task is None:
        task = _detect_task(df, target_col)
    # Target: codifica si es string.
    df_t, label_col = _encode_columns(df, [target_col]) if isinstance(
        df.schema[target_col].dataType, StringType
    ) else (df, target_col)
    model, assembler, df_enc = _train_model(df_t, feature_cols, label_col, model_name, task)
    # ``df_enc`` ya tiene las features codificadas; re-aplica el assembler.
    df_vec = assembler.transform(df_enc)
    df_pred = model.transform(df_vec)

    if task == "classification":
        pdf = df_pred.select(label_col, "probability").toPandas()
        y_true = pdf[label_col].to_numpy(dtype=int)
        # Cada valor de ``probability`` es un vector (DenseVector); a lista de floats.
        prob = np.array([list(v) for v in pdf["probability"]], dtype=float)  # (n, K).
        if prob.shape[1] == 2:
            # Binaria: prob de la clase 1.
            return auc_binary(y_true, prob[:, 1])
        # Multiclase: (K, n).
        return auc_macro(y_true, prob.T)
    else:
        pdf = df_pred.select(label_col, "prediction").toPandas()
        y_true = pdf[label_col].to_numpy(dtype=float)
        y_pred = pdf["prediction"].to_numpy(dtype=float)
        return r2_score(y_true, y_pred)


# ----------------------------------------------------------------------
# Curva de eficiencia.
# ----------------------------------------------------------------------
def efficiency_curve(
    df: DataFrame,
    feature_order: Sequence[str],
    target_col: str,
    model_name: str = "gbt",
    task: str | None = None,
) -> List[Tuple[int, float]]:
    """AUC/R² al añadir features en el orden ``feature_order`` (top-k)."""
    if task is None:
        task = _detect_task(df, target_col)
    curve: List[Tuple[int, float]] = []
    for k in range(1, len(feature_order) + 1):
        metric = train_and_evaluate(
            df, list(feature_order[:k]), target_col, model_name, task
        )
        curve.append((k, float(metric)))
    return curve


# ----------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------
@dataclass
class EvaluationReport:
    """Resultado de ``report``: métricas vs baseline y curva de eficiencia."""

    model_name: str
    task: str
    metric_selected: float
    metric_all: float
    selected_features: List[str]
    efficiency_curve: List[Tuple[int, float]] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {
            "model": self.model_name,
            "task": self.task,
            "metric_selected": self.metric_selected,
            "metric_all": self.metric_all,
            "n_selected": len(self.selected_features),
            "efficiency_curve": self.efficiency_curve,
        }


def report(
    df: DataFrame,
    selected_features: Sequence[str],
    ranking: Sequence[str],
    target_col: str,
    model_name: str = "gbt",
    task: str | None = None,
    all_features: Sequence[str] | None = None,
) -> EvaluationReport:
    """Evaluación agnóstica al modelo.

    - ``metric_selected``: AUC/R² con las features seleccionadas.
    - ``metric_all``: AUC/R² con todas las features (baseline).
    - ``efficiency_curve``: AUC/R² al añadir features en el orden ``ranking``.
    """
    if task is None:
        task = _detect_task(df, target_col)
    if all_features is None:
        all_features = [c for c in df.columns if c != target_col]

    metric_selected = train_and_evaluate(
        df, list(selected_features), target_col, model_name, task
    )
    metric_all = train_and_evaluate(
        df, list(all_features), target_col, model_name, task
    )
    curve = efficiency_curve(df, list(ranking), target_col, model_name, task)

    return EvaluationReport(
        model_name=model_name,
        task=task,
        metric_selected=metric_selected,
        metric_all=metric_all,
        selected_features=list(selected_features),
        efficiency_curve=curve,
    )
