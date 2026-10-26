"""Benchmark de escala — CLI local (Hito 7 / DBR 15.4).

La implementación vive en ``sparkmim.benchmarks.scale`` (incluida en el
wheel); este script solo añade la CLI local: rejilla, CSV, sesión Spark
local y staging en un ``TemporaryDirectory`` del sistema de archivos local
(con limpieza).

**No usar en notebook de Databricks** (crea una sesión Spark local):
importar ``sparkmim.benchmarks.run_point`` del wheel. Ver
``benchmarks/README.md``.

Uso:
    python benchmarks/bench_scale.py --csv out.csv
    python benchmarks/bench_scale.py --quick          # rejilla pequeña
    python benchmarks/bench_scale.py --n 100000 --N 100   # un punto

Nota: requiere JDK 17 (Arrow). Este script lo configura automáticamente
(si existe ``jdk17``).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

# --- JDK 17 (Arrow) ---
_REPO = Path(__file__).resolve().parent.parent
_JDK17 = _REPO / "jdk17"
if _JDK17.exists():
    os.environ["JAVA_HOME"] = str(_JDK17)
    os.environ["PATH"] = str(_JDK17 / "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

# --- winutils + hadoop.dll (Windows; ver tools/fetch_winutils.py) ---
_WINUTILS = _REPO / "winutils"
if (_WINUTILS / "bin").exists() and not os.environ.get("HADOOP_HOME"):
    os.environ["HADOOP_HOME"] = str(_WINUTILS)
    os.environ["PATH"] = str(_WINUTILS / "bin") + os.pathsep + os.environ.get("PATH", "")

# --- TEMP local (Windows): el launcher de PySpark escribe a %TEMP%; si no
# es escribible, el arranque de la sesión se queda a la espera. ---
if os.name == "nt":
    _SPARK_TMP = _REPO / ".tmp_spark"
    _SPARK_TMP.mkdir(exist_ok=True)
    os.environ["TEMP"] = str(_SPARK_TMP)
    os.environ["TMP"] = str(_SPARK_TMP)
else:
    _SPARK_TMP = None

# --- Import del package (implementación) ---
_SRC = str(_REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pyspark.sql import SparkSession  # noqa: E402

from sparkmim.benchmarks.scale import clean_stale_tmp, run_point  # noqa: E402

# Limpia los residuos de ejecuciones anteriores ANTES de crear la sesión
# (spark-*/blockmgr-*/nativas del JVM que no hicieron teardown).
clean_stale_tmp(_SPARK_TMP)


def _make_spark(master: str) -> SparkSession:
    return (
        SparkSession.builder
        .master(master)
        .appName("sparkmim-bench-scale")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark de escala SparkMIM")
    ap.add_argument("--csv", default="bench_scale.csv", help="ruta del CSV de salida")
    ap.add_argument("--quick", action="store_true", help="rejilla pequeña (n{1e4,1e5}, N{50,100})")
    ap.add_argument("--n", type=int, action="append", help="valores de n (repetible)")
    ap.add_argument("--N", type=int, action="append", help="valores de N (repetible)")
    ap.add_argument("--master", default="local[8]", help="master de Spark")
    ap.add_argument("--n-informative", type=int, default=10)
    ap.add_argument("--n-redundant", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.n and args.N:
        n_vals = args.n
        N_vals = args.N
    elif args.quick:
        n_vals = [10_000, 100_000]
        N_vals = [50, 100]
    else:
        n_vals = [100_000, 1_000_000, 10_000_000]
        N_vals = [100, 200, 500]

    spark = _make_spark(args.master)
    rows = []
    for n in n_vals:
        for N in N_vals:
            print(f"--- n={n:,} N={N} ---", flush=True)
            # Staging local: directorio temporal con limpieza automática.
            with tempfile.TemporaryDirectory(prefix="sparkmim-bench-") as tmp:
                row = run_point(
                    spark,
                    n=n,
                    N=N,
                    n_informative=args.n_informative,
                    n_redundant=args.n_redundant,
                    seed=args.seed,
                    max_features=args.max_features,
                    staging_path=tmp,
                )
            rows.append(row)
            print(
                f"    staging={row['staging_s']}s total={row['total_s']}s "
                f"(e0={row['etapa0_s']} e1={row['etapa1_s']} "
                f"e2={row['etapa2_s']} e3={row['etapa3_s']}) "
                f"mem={row['driver_mem_mb']}MB "
                f"sel={row['n_selected']} (info={row['informative_recuperadas']})",
                flush=True,
            )
    spark.stop()

    # Escribir CSV (sin la ruta temporal de staging).
    if rows:
        fieldnames = [k for k in rows[0].keys() if k != "staging_path"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV escrito en {args.csv}", flush=True)


if __name__ == "__main__":
    main()
