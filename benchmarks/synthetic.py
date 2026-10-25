"""Generador de datos sintéticos — wrapper local del package (Hito 7 / DBR 15.4).

La implementación autónoma vive en ``sparkmim.benchmarks.synthetic``
(incluida en el wheel); este wrapper la expone para las herramientas
locales (``bench_scale.py``, ``bench_grid.py``) sin depender de
``tests/planted.py``. En Databricks no se usa este archivo: se importa
directamente ``sparkmim.benchmarks.generate`` del wheel.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sparkmim.benchmarks.synthetic import generate  # noqa: E402

__all__ = ["generate"]
