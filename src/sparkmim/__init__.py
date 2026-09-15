"""SparkMIM — framework PySpark distribuido de feature selection informacional.

Criterios greedy de información mutua condicional (JMIM/CMIM) con cálculo de
tablas en pases únicos, control de significancia (FDR) y evaluación agnóstica
al modelo. Sin sklearn.
"""

__version__ = "0.1.0"

# Los exports públicos (selectores, config, model) se añaden a medida que se
# implementan los hitos. Por ahora, el núcleo de entropía y tablas.
from .info.entropy import (
    entropy_from_counts,
    joint_entropy_from_counts,
    mutual_information,
    conditional_mi,
)

__all__ = [
    "__version__",
    "entropy_from_counts",
    "joint_entropy_from_counts",
    "mutual_information",
    "conditional_mi",
]
