"""Benchmark de escala empaquetado con staging en parquet (Databricks DBR 15.4).

``run_point`` ejecuta un punto ``(n, N)`` de la rejilla en un entorno donde
SparkMIM se instala por **wheel** (Databricks 15.4 LTS, Spark 3.5.0, Spark
Connect o sesión clásica), sin clonar el repo ni usar ``sys.path``.

Problema que resuelve este módulo
---------------------------------
``generate()`` devuelve los arrays NumPy completos en el driver (~2,08 GB
decimales para ``n=10⁷, N=25``). La construcción anterior expandía todas
las filas en una lista Python y las enviaba en un solo payload de
``createDataFrame``; en Spark Connect ese payload superó
``spark.rpc.message.maxSize`` (256 MiB; 582.501.887 B medidos a 10M×25) y
la sesión murió.

Solución (confirmada): **pandas/Arrow por lotes + staging en parquet**.
Los slices de los arrays se convierten en DataFrames pandas pequeños con
un **schema explícito y consistente en todos los chunks** (features
``DoubleType``, target ``LongType``, mismas columnas y orden), cada uno se
escribe **inmediatamente** con ``write.mode("append").parquet`` en un
directorio de staging único — acciones separadas y materializadas, muy por
debajo del límite RPC — y la entrada se lee una sola vez con
``spark.read.parquet``. No se acumulan DataFrames locales ni se hace
``union``.

Memoria
-------
La entrada se genera con :func:`generate_chunks` (``synthetic``) y se
stagea **chunk a chunk**: el driver solo conserva un chunk de filas
(``O(chunk_rows × N)`` ≈ el presupuesto de bytes del lote, ~16 MiB) en
lugar de los arrays completos (~2,08 GB decimales a 10M×25). El driver de
la rejilla ya no necesita dimensionarse por ``n × N``: el caso N=1000,
que moría por memoria del kernel, queda desbloqueado. La secuencia del
RNG es por columna (``SeedSequence.spawn``), determinista e invariante a
``chunk_rows`` (ver ``synthetic.generate_chunks``); los valores difieren
de ``generate()`` (stream único), que se mantiene como referencia de
paridad con ``tests/planted.py``.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, LongType, StructField, StructType

from sparkmim import JMIMSelector

from .synthetic import _feature_names, generate_chunks

__all__ = [
    "DEFAULT_BATCH_BYTES",
    "DEFAULT_STAGING_ROOT",
    "default_staging_path",
    "run_point",
]

#: Raíz de staging por defecto (DBFS /tmp, aceptado). Configurable con
#: ``staging_path`` (p. ej. un UC Volume) si la raíz de DBFS no está
#: disponible en el compute.
DEFAULT_STAGING_ROOT = "dbfs:/tmp/sparkmim-benchmark"

#: Presupuesto de bytes *raw* por lote de staging: 16 MiB (a N=25 equivale
#: a ~80k filas). Conservador respecto al límite RPC de 256 MiB: deja mucho
#: margen para el overhead Arrow/Connect de cada envío.
DEFAULT_BATCH_BYTES = 16 * 1024 * 1024

try:  # psutil es opcional (no siempre disponible en el runtime de Databricks).
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def default_staging_path() -> str:
    """Directorio de staging único: ``<DEFAULT_STAGING_ROOT>/<uuid>``.

    Única por llamada: si una ejecución falla a mitad de staging, el
    reintento usa un directorio nuevo y no duplica partes parquet ya
    escritas en el anterior.
    """
    return f"{DEFAULT_STAGING_ROOT}/{uuid.uuid4()}"


def _rows_per_batch(n_features: int, batch_bytes: int) -> int:
    """Filas por lote según el presupuesto de bytes.

    Presupuesto aproximado de ``8 B`` por valor (``N`` doubles + 1 long):
    ``batch_bytes / (8 × (N + 1))``, con un mínimo de 1 fila.
    """
    return max(1, int(batch_bytes) // (8 * (int(n_features) + 1)))


def _benchmark_schema(cols: List[str]) -> StructType:
    """Schema explícito de todos los chunks: features double, target long."""
    return StructType(
        [StructField(c, DoubleType(), False) for c in cols]
        + [StructField("y", LongType(), False)]
    )


def _staged_df(
    spark: SparkSession,
    data: Dict[str, np.ndarray],
    staging_path: str,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
) -> DataFrame:
    """Escribe la entrada en ``staging_path`` por lotes y la lee una vez.

    Cada slice de los arrays produce un DataFrame pandas (Arrow) con el
    schema explícito y se escribe **inmediatamente** con ``append``:
    acciones separadas y materializadas, cada una con un payload acotado.
    No se regenera nada por chunk (no hay reset de seed) y no se acumula
    ni ``union`` de DataFrames locales.
    """
    import pandas as pd

    cols = [c for c in data if c != "y"]
    arrays = {c: np.ascontiguousarray(data[c], dtype=np.float64) for c in cols}
    y = np.ascontiguousarray(data["y"], dtype=np.int64)
    n = y.shape[0]
    schema = _benchmark_schema(cols)
    step = _rows_per_batch(len(cols), batch_bytes)
    for start in range(0, n, step):
        stop = min(start + step, n)
        block = {c: arrays[c][start:stop] for c in cols}
        block["y"] = y[start:stop]
        spark.createDataFrame(pd.DataFrame(block), schema).write.mode("append").parquet(
            staging_path
        )
    return spark.read.parquet(staging_path)


def _staged_generate_df(
    spark: SparkSession,
    n: int,
    n_features: int,
    n_informative: int,
    n_redundant: int,
    seed: int,
    staging_path: str,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
) -> DataFrame:
    """Genera la entrada **chunk a chunk** (``generate_chunks``) y stagea
    cada chunk en parquet inmediatamente (``append``), sin conservar los
    arrays completos en el driver.

    Mismo contrato que :func:`_staged_df` (schema explícito, lotes acotados
    por ``batch_bytes``) pero con memoria ``O(chunk × N)``: cada chunk
    generado se convierte a pandas/Arrow y se escribe antes de generar el
    siguiente.
    """
    import pandas as pd

    cols = _feature_names(n_features, n_informative, n_redundant)
    schema = _benchmark_schema(cols)
    step = _rows_per_batch(n_features, batch_bytes)
    first = True
    for chunk in generate_chunks(
        n=n,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        seed=seed,
        task="classification",
        chunk_rows=step,
    ):
        block = {c: np.ascontiguousarray(chunk[c], dtype=np.float64) for c in cols}
        block["y"] = np.ascontiguousarray(chunk["y"], dtype=np.int64)
        spark.createDataFrame(pd.DataFrame(block), schema).write.mode(
            "overwrite" if first else "append"
        ).parquet(staging_path)
        first = False
    return spark.read.parquet(staging_path)


def _driver_mem_mb() -> float:
    """RSS del proceso driver en MiB (``-1.0`` si psutil no está)."""
    if psutil is None:  # pragma: no cover
        return -1.0
    return psutil.Process().memory_info().rss / 1e6


def run_point(
    spark: SparkSession,
    n: int,
    N: int,
    n_informative: int,
    n_redundant: int,
    seed: int,
    max_features: int,
    staging_path: Optional[str] = None,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
) -> dict:
    """Ejecuta un punto ``(n, N)`` y devuelve las métricas.

    La entrada se genera **chunk a chunk** (:func:`generate_chunks`) y se
    stagea en parquet (lotes pandas/Arrow) en ``staging_path`` sin
    conservar los arrays completos en el driver; el selector solo ve el
    resultado de ``spark.read.parquet``.

    Args:
        spark: ``SparkSession`` (clásica) o cliente Spark Connect.
        n: nº de filas.
        N: nº total de features.
        n_informative: features informativas de la estructura plantada.
        n_redundant: features redundantes de la estructura plantada.
        seed: semilla del generador y del selector.
        max_features: nº máximo de features seleccionadas.
        staging_path: directorio exacto donde escribir las partes parquet
            (``dbfs:/...`` o local). Si es ``None``, se crea una única
            con :func:`default_staging_path`. **No se borra**: el entorno
            la limpia (``dbutils.fs.rm`` en Databricks; ``TemporaryDirectory``
            en el CLI local). Un reintento tras fallo debe usar un
            directorio nuevo.
        batch_bytes: presupuesto de bytes raw por lote
            (:data:`DEFAULT_BATCH_BYTES` = 16 MiB).

    Returns:
        Dict de métricas. ``total_s`` es solo el ``fit`` del selector
        (misma semántica que antes del staging: se mide tras tener la
        entrada lista); ``staging_s`` informa aparte del tiempo de crear y
        stagear la entrada. ``staging_path`` indica el directorio usado.
    """
    if staging_path is None:
        staging_path = default_staging_path()

    t_stage = time.perf_counter()
    df = _staged_generate_df(
        spark,
        n=n,
        n_features=N,
        n_informative=n_informative,
        n_redundant=n_redundant,
        seed=seed,
        staging_path=staging_path,
        batch_bytes=batch_bytes,
    )
    staging_s = time.perf_counter() - t_stage

    t0 = time.perf_counter()
    model = JMIMSelector(
        target="y",
        max_features=max_features,
        screen_top_k=200,
        significance="chi2",
        seed=seed,
    ).fit(df)
    wall = time.perf_counter() - t0
    timings = model.timings_
    selected = list(model.selected_features)
    return {
        "n": n,
        "N": N,
        "n_informative": n_informative,
        "n_redundant": n_redundant,
        "staging_s": round(staging_s, 3),
        "etapa0_s": round(timings.get("etapa0", 0.0), 3),
        "etapa1_s": round(timings.get("etapa1", 0.0), 3),
        "etapa2_s": round(timings.get("etapa2", 0.0), 3),
        "etapa3_s": round(timings.get("etapa3", 0.0), 3),
        "total_s": round(wall, 3),
        "driver_mem_mb": round(_driver_mem_mb(), 1),
        "n_selected": len(selected),
        "informative_recuperadas": sum(1 for f in selected if f.startswith("x")),
        "staging_path": staging_path,
    }
