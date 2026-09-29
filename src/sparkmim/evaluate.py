"""Evaluación agnóstica al modelo (Hito 6).

La lógica de evaluación no entrena modelos directamente: detecta la tarea
(``detect_task``), codifica las columnas categóricas, ensambla el vector
``features`` y pasa el df vectorizado a una ``ModelFactory``
(``model_factory.py``) — la costura entre la lógica de evaluación y los
backends (docs/DESIGN.md §2, §7). La fábrica entrena y devuelve el df de
predicciones; la métrica se computa en el driver.

- AUC (binaria: Mann-Whitney; multiclase: macro, one-vs-rest) y R².
- ``report``: AUC/R² con las features seleccionadas, con todas las features
  (baseline) y la curva de eficiencia. Coste: 2 + |ranking| entrenamientos.
- Costura ``ModelFactory``: adaptadores ``GbtFactory`` (default),
  ``XgboostFactory`` y ``LightgbmFactory`` (extras opcionales, import
  perezoso); para tests, un adaptador trivial determinista sin Spark ML.
- Detección de tarea: string → clasificación; float/double → regresión;
  entero → clasificación si ≤ 20 valores distintos (1 job de Spark), si no
  regresión.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, FloatType, StringType

from scipy.stats import rankdata

from .model_factory import ModelFactory, make_model_factory

__all__ = [
    "EvaluationReport",
    "auc_binary",
    "auc_macro",
    "r2_score",
    "detect_task",
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
# Detección de tarea y preparación (codificación de categóricas).
# ----------------------------------------------------------------------
def detect_task(df: DataFrame, target_col: str) -> str:
    """``'classification'`` | ``'regression'`` según el dtype del target.

    Regla: string → clasificación; float/double → regresión; entero →
    clasificación si ≤ 20 valores distintos (1 job de Spark), si no
    regresión.
    """
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


def _resolve_model(model: "str | ModelFactory") -> ModelFactory:
    """Resuelve el backend: nombre → adaptador (costura ``ModelFactory``)."""
    if isinstance(model, str):
        return make_model_factory(model)
    return model


# ----------------------------------------------------------------------
# Entrenamiento + evaluación (detrás de la costura ModelFactory).
# ----------------------------------------------------------------------
def train_and_evaluate(
    df: DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    model: "str | ModelFactory" = "gbt",
    task: str | None = None,
) -> float:
    """Entrena sobre ``feature_cols`` y devuelve AUC (cl.) o R² (reg.).

    ``model``: nombre de backend (``"gbt"``/``"xgboost"``/``"lightgbm"``) o
    una ``ModelFactory`` (costura, ``model_factory.py``).

    Coste: 1 entrenamiento. Si ``task`` es None se detecta con
    ``detect_task``.
    """
    if task is None:
        task = detect_task(df, target_col)
    factory = _resolve_model(model)
    # Target: codifica si es string.
    df_t, label_col = _encode_columns(df, [target_col]) if isinstance(
        df.schema[target_col].dataType, StringType
    ) else (df, target_col)
    # Features: codifica si son string y ensambla el vector.
    df_enc, enc_cols = _encode_columns(df_t, feature_cols)
    from pyspark.ml.feature import VectorAssembler

    df_vec = VectorAssembler(inputCols=enc_cols, outputCol="features").transform(df_enc)
    # Costura: la fábrica entrena y devuelve el df de predicciones.
    df_pred = factory.train(df_vec, task, label_col)

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
    model: "str | ModelFactory" = "gbt",
    task: str | None = None,
) -> List[Tuple[int, float]]:
    """AUC/R² al añadir features en el orden ``feature_order`` (top-k).

    Coste: ``len(feature_order)`` entrenamientos.
    """
    if task is None:
        task = detect_task(df, target_col)
    factory = _resolve_model(model)
    curve: List[Tuple[int, float]] = []
    for k in range(1, len(feature_order) + 1):
        metric = train_and_evaluate(
            df, list(feature_order[:k]), target_col, factory, task
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
    model: "str | ModelFactory" = "gbt",
    task: str | None = None,
    all_features: Sequence[str] | None = None,
) -> EvaluationReport:
    """Evaluación agnóstica al modelo.

    - ``metric_selected``: AUC/R² con las features seleccionadas.
    - ``metric_all``: AUC/R² con todas las features (baseline).
    - ``efficiency_curve``: AUC/R² al añadir features en el orden ``ranking``.

    ``model``: nombre de backend o una ``ModelFactory`` (costura).
    Coste: ``2 + len(ranking)`` entrenamientos.
    """
    if task is None:
        task = detect_task(df, target_col)
    if all_features is None:
        all_features = [c for c in df.columns if c != target_col]
    factory = _resolve_model(model)

    metric_selected = train_and_evaluate(
        df, list(selected_features), target_col, factory, task
    )
    metric_all = train_and_evaluate(
        df, list(all_features), target_col, factory, task
    )
    curve = efficiency_curve(df, list(ranking), target_col, factory, task)

    return EvaluationReport(
        model_name=factory.name,
        task=task,
        metric_selected=metric_selected,
        metric_all=metric_all,
        selected_features=list(selected_features),
        efficiency_curve=curve,
    )
