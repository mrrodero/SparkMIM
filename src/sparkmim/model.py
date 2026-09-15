"""SelectorModel: resultado de la selección (Hito 4, etapa 3).

Atributos:
- ``selected_features``: nombres originales de las features seleccionadas
  (en orden de selección).
- ``scores_``: puntaje del criterio por ronda (nats).
- ``ranking_``: ranking completo de candidatas ``(nombre, score)`` ordenado
  por score descendente (score = MI univariante, etapa 1).
- ``criterion``: criterio usado (``"mrmr" | "mim" | "jmi" | "jmim" | "cmim"``).
- ``n_rows``: nº de filas de ``df_prep``.
- ``target``: nombre de la columna target.

Métodos:
- ``transform(df)``: ``df`` con solo las features seleccionadas + el target.
- ``report(df, model="gbt")``: evaluación (Hito 6, ``evaluate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from pyspark.sql import DataFrame


@dataclass
class SelectorModel:
    selected_features: List[str]
    scores_: List[float]
    ranking_: List[Tuple[str, float]]
    criterion: str
    n_rows: int
    target: str

    def __post_init__(self):
        if len(self.selected_features) != len(self.scores_):
            raise ValueError(
                "selected_features y scores_ deben tener la misma longitud"
            )

    def transform(self, df: DataFrame) -> DataFrame:
        """Devuelve ``df`` con solo las features seleccionadas + el target.

        Opera sobre el df original (nombres originales de columnas).
        """
        cols = list(self.selected_features) + [self.target]
        return df.select(*cols)

    def report(self, df, model: str = "gbt"):
        """Evaluación agnóstica al modelo (AUC/R² vs baseline + curva de eficiencia).

        Entrena ``model`` (``"gbt"`` por defecto, ``"xgboost"``/``"lightgbm"``
        como extras) sobre las features seleccionadas y sobre todas las
        features (baseline), y devuelve un ``EvaluationReport``.
        """
        from .evaluate import report as _report

        ranking_names = [name for name, _ in self.ranking_]
        return _report(
            df,
            selected_features=self.selected_features,
            ranking=ranking_names,
            target_col=self.target,
            model_name=model,
        )
