"""SparkMIM — framework PySpark distribuido de feature selection informacional.

Criterios greedy de información mutua condicional (JMIM/CMIM) con cálculo de
tablas en pases únicos, control de significancia (FDR) y evaluación agnóstica
al modelo. Sin sklearn.
"""

__version__ = "0.1.0"

# Núcleo de entropía y tablas (hitos 1-2).
from .info.entropy import (
    entropy_from_counts,
    joint_entropy_from_counts,
    mutual_information,
    conditional_mi,
    conditional_mi_multi,
)

# Configuración, esquema, preprocesado, screening (hitos 2-3).
from .config import SelectorConfig
from .model import SelectorModel
from .selector import (
    CMIMSelector,
    InfoSelector,
    JMIMSelector,
    MIMSelector,
    MRMRSelector,
)

__all__ = [
    "__version__",
    # entropía
    "entropy_from_counts",
    "joint_entropy_from_counts",
    "mutual_information",
    "conditional_mi",
    "conditional_mi_multi",
    # config / model / selectores
    "SelectorConfig",
    "SelectorModel",
    "InfoSelector",
    "JMIMSelector",
    "CMIMSelector",
    "MRMRSelector",
    "MIMSelector",
]
