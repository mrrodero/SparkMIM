"""Benchmarks y datos sintéticos (incluido en el wheel).

Uso desde Databricks (wheel, sin clonar el repo):

    from sparkmim.benchmarks import run_point, generate, default_staging_path
"""

from .scale import (
    DEFAULT_BATCH_BYTES,
    DEFAULT_STAGING_ROOT,
    default_staging_path,
    run_point,
)
from .synthetic import generate

__all__ = [
    "DEFAULT_BATCH_BYTES",
    "DEFAULT_STAGING_ROOT",
    "default_staging_path",
    "generate",
    "run_point",
]
