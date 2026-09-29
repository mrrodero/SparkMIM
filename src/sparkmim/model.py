"""SelectorModel: resultado de la selección (Hito 4, etapa 3).

Atributos:
- ``selected_features``: nombres originales de las features seleccionadas
  (en orden de selección).
- ``scores_``: puntaje del criterio por ronda (nats).
- ``ranking_``: ranking completo de la población ``(nombre, score)`` ordenado
  por score descendente (score = MI univariante, etapa 1; con el estimador
  KSG, MI de kNN sobre el subsample).
- ``criterion``: criterio usado (``"mrmr" | "mim" | "jmi" | "jmim" | "cmim"``).
- ``n_rows``: nº de filas de ``df_prep``.
- ``target``: nombre de la columna target.

Métodos:
- ``transform(df)``: ``df`` con solo las features seleccionadas + el target.
- ``report(df, model="gbt")``: evaluación (Hito 6, ``evaluate.py``) detrás de
  la costura ``ModelFactory`` (``model_factory.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TYPE_CHECKING

from pyspark.sql import DataFrame

if TYPE_CHECKING:
    from .model_factory import ModelFactory


@dataclass
class SelectorModel:
    selected_features: List[str]
    scores_: List[float]
    ranking_: List[Tuple[str, float]]
    criterion: str
    n_rows: int
    target: str
    # Wall-time (s) por etapa, si se midió (benchmark). Vacío si no.
    timings_: Dict[str, float] = field(default_factory=dict)

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

    def report(self, df, model: "str | ModelFactory" = "gbt"):
        """Evaluación agnóstica al modelo (AUC/R² vs baseline + curva de eficiencia).

        ``model``: nombre de backend (``"gbt"`` por defecto, ``"xgboost"``/
        ``"lightgbm"`` como extras) o una ``ModelFactory`` (costura,
        ``model_factory.py``).

        Coste: ``2 + len(ranking_)`` entrenamientos.
        """
        from .evaluate import report as _report

        ranking_names = [name for name, _ in self.ranking_]
        return _report(
            df,
            selected_features=self.selected_features,
            ranking=ranking_names,
            target_col=self.target,
            model=model,
        )
