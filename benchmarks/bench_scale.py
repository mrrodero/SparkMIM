"""Benchmark de escala (Hito 7).

Ejecuta una rejilla ``n × N`` en ``local[8]``, mide wall-time por etapa y
memoria del driver, y escribe un CSV.

Meta (PLAN §9/§11): n=10⁶, N=200 < 15 min end-to-end.

Uso:
    python benchmarks/bench_scale.py --csv out.csv
    python benchmarks/bench_scale.py --quick          # rejilla pequeña
    python benchmarks/bench_scale.py --n 100000 --N 100   # un punto

Nota: requiere JDK 17 (Arrow). Este script lo configura automáticamente
(si existe ``sparkmim/jdk17``).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- JDK 17 (Arrow) ---
_REPO = Path(__file__).resolve().parent.parent
_JDK17 = _REPO / "jdk17"
if _JDK17.exists():
    os.environ["JAVA_HOME"] = str(_JDK17)
    os.environ["PATH"] = str(_JDK17 / "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

# --- Import del generador (mismo directorio) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import generate  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402
from sparkmim import JMIMSelector  # noqa: E402

# --- Memoria del driver ---
try:
    import psutil  # type: ignore

    def _driver_mem_mb() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
except Exception:  # pragma: no cover
    def _driver_mem_mb() -> float:
        return -1.0  # no disponible.


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


def _to_df(spark: SparkSession, data: dict) -> "SparkSession.DataFrame":
    cols = [c for c in data if c != "y"]
    y = data["y"]
    rows = [
        tuple(float(v) for v in row) + (int(y[i]),)
        for i, row in enumerate(zip(*[data[c] for c in cols]))
    ]
    return spark.createDataFrame(rows, cols + ["y"])


def run_point(
    spark: SparkSession,
    n: int,
    N: int,
    n_informative: int,
    n_redundant: int,
    seed: int,
    max_features: int,
) -> dict:
    """Ejecuta un punto (n, N) y devuelve las métricas."""
    data = generate(
        n=n,
        n_features=N,
        n_informative=n_informative,
        n_redundant=n_redundant,
        seed=seed,
        task="classification",
    )
    df = _to_df(spark, data)
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
    return {
        "n": n,
        "N": N,
        "n_informative": n_informative,
        "n_redundant": n_redundant,
        "etapa0_s": round(timings.get("etapa0", 0.0), 3),
        "etapa1_s": round(timings.get("etapa1", 0.0), 3),
        "etapa2_s": round(timings.get("etapa2", 0.0), 3),
        "etapa3_s": round(timings.get("etapa3", 0.0), 3),
        "total_s": round(wall, 3),
        "driver_mem_mb": round(_driver_mem_mb(), 1),
        "n_selected": len(model.selected_features),
        "informative_recuperadas": sum(
            1 for f in model.selected_features if f.startswith("x")
        ),
    }


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
            row = run_point(
                spark,
                n=n,
                N=N,
                n_informative=args.n_informative,
                n_redundant=args.n_redundant,
                seed=args.seed,
                max_features=args.max_features,
            )
            rows.append(row)
            print(
                f"    total={row['total_s']}s "
                f"(e0={row['etapa0_s']} e1={row['etapa1_s']} "
                f"e2={row['etapa2_s']} e3={row['etapa3_s']}) "
                f"mem={row['driver_mem_mb']}MB "
                f"sel={row['n_selected']} (info={row['informative_recuperadas']})",
                flush=True,
            )
    spark.stop()

    # Escribir CSV.
    if rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV escrito en {args.csv}", flush=True)


if __name__ == "__main__":
    main()
