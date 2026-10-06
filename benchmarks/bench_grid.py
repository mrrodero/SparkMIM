"""SparkMIM — Benchmark de escala: rejilla n × N (hasta 10M filas × 1000 features).

Script autónomo y sin intervención: se lanza, corre solo y deja un informe.
Cada punto de la rejilla corre en un **JVM nuevo** (subproceso), de modo que
la RAM del driver se libera entre puntos y no hay acumulación de memoria.

Salidas (incremental, a prueba de cortes, reanudable):
    benchmarks/bench_grid.json          puntos + metadatos (v2)
    benchmarks/bench_grid_report.html   informe interactivo (Plotly)
    benchmarks/bench_grid.log           consola con marca de tiempo

Uso (desde la raíz del repo, con el venv):
    python benchmarks/bench_grid.py                 # rejilla completa (54 puntos)
    python benchmarks/bench_grid.py --quick         # rejilla pequeña (4 puntos)
    python benchmarks/bench_grid.py --n 100000 --N 200
    python benchmarks/bench_grid.py --plot-only     # regenerar solo el HTML
    python benchmarks/bench_grid.py --point 1000000 200   # modo trabajador (1 punto)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT / "src"), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

JDK = REPO_ROOT / "jdk17"

DEFAULT_JSON = BENCH_DIR / "bench_grid.json"
DEFAULT_HTML = BENCH_DIR / "bench_grid_report.html"
DEFAULT_LOG = BENCH_DIR / "bench_grid.log"

# Rejillas por defecto.
FULL_N = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000]
FULL_NF = [25, 50, 100, 200, 500, 1000]
QUICK_N = [10_000, 100_000]
QUICK_NF = [25, 200]

STAGES = ("etapa0", "etapa1", "etapa2", "etapa3")

# Configuración base (comparable con los benchmarks anteriores).
CONFIG = {
    "max_features": 20,
    "screen_top_k": 200,
    "significance": "chi2",
    "seed": 42,
    "n_informative": 10,
    "n_redundant": 5,
    "task": "classification",
}


# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

def _setup_jdk() -> None:
    """Fija JDK 17 (requisito de PySpark para Arrow) si existe en el repo."""
    if JDK.exists():
        os.environ["JAVA_HOME"] = str(JDK)
        os.environ["PATH"] = str(JDK / "bin") + os.pathsep + os.environ.get("PATH", "")


def _cpu_name() -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=15,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor() or "n/d"


def machine_info() -> dict:
    info = {
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "ram_gb": None,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import pyspark
        info["pyspark"] = pyspark.__version__
    except Exception:
        pass
    return info


def _versions() -> dict:
    v = {"python": platform.python_version()}
    for name in ("pyspark", "plotly", "numpy", "pandas"):
        try:
            mod = __import__(name)
            v[name] = getattr(mod, "__version__", "n/d")
        except Exception:
            v[name] = None
    return v


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _make_spark(n: int = 0, nf: int = 0) -> "SparkSession":  # noqa: F821
    from pyspark.sql import SparkSession
    # Heap adaptativo del driver: base 8g + margen proporcional al tamaño de
    # los datos (8 bytes/valor, float64). Los datos se cargan vía parquet
    # temporal (evita el pico de createDataFrame(pandas)); el driver debe
    # sostener el RDD (≈data_gb) + el working set de los jobs. Escalamos con
    # n*nf (×3 + 8g) y lo topamos a 24g para que driver + Python quepa en RAM.
    data_gb = (n * nf * 8) / 1e9
    driver_gb = min(24, max(8, int(data_gb * 3) + 8))
    spark = SparkSession.builder \
        .master("local[8]") \
        .appName("sparkmim-bench-grid") \
        .config("spark.driver.memory", f"{driver_gb}g") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------------
# JSON v2 (compatible con v1: lista plana de puntos)
# ---------------------------------------------------------------------------

def load_doc(path: Path) -> dict:
    if not path.exists():
        return {"meta": {}, "points": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):  # v1: lista plana
        return {"meta": {}, "points": data}
    return {"meta": data.get("meta", {}), "points": data.get("points", [])}


def save_doc(doc: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _cfg_from_args(args: argparse.Namespace) -> dict:
    return {
        "max_features": args.max_features,
        "screen_top_k": args.screen_top_k,
        "significance": args.significance,
        "seed": args.seed,
        "n_informative": CONFIG["n_informative"],
        "n_redundant": CONFIG["n_redundant"],
        "task": CONFIG["task"],
    }


# ---------------------------------------------------------------------------
# Trabajador: un punto en un proceso (JVM) nuevo
# ---------------------------------------------------------------------------

class _MemSampler:
    """Pico de RSS del proceso Python y de la JVM del driver (hijo de py4j).

    El campo driver_mem_mb medía en realidad el RSS de Python; ahora se
    separan ambos: python_rss_mb y jvm_rss_mb, muestreados cada 2 s durante
    el fit.
    """

    def __init__(self):
        import psutil
        self._psutil = psutil
        self._python = psutil.Process()
        # En Windows pyspark lanza la JVM con "cmd /c java …", así que java
        # puede ser nieta del proceso Python: buscar en todos los descendientes.
        self._java = next(
            (c for c in self._python.children(recursive=True)
             if "java" in c.name().lower()),
            None,
        )
        self._peak_python = self._python.memory_info().rss
        self._peak_java = self._java.memory_info().rss if self._java else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(2.0):
            try:
                self._peak_python = max(self._peak_python,
                                        self._python.memory_info().rss)
                if self._java is not None:
                    self._peak_java = max(self._peak_java,
                                          self._java.memory_info().rss)
            except self._psutil.NoSuchProcess:
                break

    def peaks(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=3.0)
        return {
            "python_rss_mb": round(self._peak_python / 1e6, 1),
            "jvm_rss_mb": (round(self._peak_java / 1e6, 1)
                           if self._peak_java is not None else None),
        }


def run_point(n: int, nf: int, cfg: dict) -> dict:
    import pandas as pd
    import synthetic
    from sparkmim import JMIMSelector

    data = synthetic.generate(
        n, nf,
        n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"],
        seed=cfg["seed"],
        task=cfg["task"],
    )
    pdf = pd.DataFrame(data)
    spark = _make_spark(n, nf)
    mem = _MemSampler()
    tmp_parquet = BENCH_DIR / ".tmp" / f"bench_grid_{n}_{nf}_{os.getpid()}.parquet"
    try:
        # Escribir a parquet temporal y leerlo con Spark: evita el pico de
        # memoria de createDataFrame(pandas) (varias copias numpy→pandas→Arrow
        # en Python, ~16 GB para 1M×500). El parquet en disco deja que el JVM
        # lea los datos en streaming sin duplicarlos en el proceso Python.
        pdf.to_parquet(tmp_parquet, engine="pyarrow", compression="snappy")
        del data, pdf
        df = spark.read.parquet(str(tmp_parquet))
        t0 = time.perf_counter()
        model = JMIMSelector(
            target="y",
            max_features=cfg["max_features"],
            screen_top_k=cfg["screen_top_k"],
            significance=cfg["significance"],
            seed=cfg["seed"],
        ).fit(df)
        total = time.perf_counter() - t0
    finally:
        spark.stop()
        try:
            tmp_parquet.unlink(missing_ok=True)
        except OSError:
            pass

    timings = model.timings_
    selected = list(model.selected_features)
    informative = sum(1 for f in selected if f.startswith("x"))
    peaks = mem.peaks()
    return {
        "n": n,
        "N": nf,
        **{f"{s}_s": round(timings.get(s, 0.0), 3) for s in STAGES},
        "total_s": round(total, 3),
        "python_rss_mb": peaks["python_rss_mb"],
        "jvm_rss_mb": peaks["jvm_rss_mb"],
        "n_selected": len(selected),
        "informative_recuperadas": informative,
        "finished_at": _now_iso(),
    }


def main_worker(args: argparse.Namespace) -> int:
    _setup_jdk()
    cfg = _cfg_from_args(args)
    n, nf = args.point
    point = run_point(n, nf, cfg)
    json_path = Path(args.json)
    doc = load_doc(json_path)
    doc["points"] = [p for p in doc["points"]
                     if not (p.get("n") == n and p.get("N") == nf)]
    doc["points"].append(point)
    doc["points"].sort(key=lambda p: (p.get("N", 0), p.get("n", 0)))
    save_doc(doc, json_path)
    print(json.dumps(point, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# Orquestador: rejilla completa
# ---------------------------------------------------------------------------

class _Log:
    def __init__(self, path: Path):
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, line: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._fh.write(f"[{stamp}] {line}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def est_peak_bytes(n: int, nf: int) -> float:
    """Estimación conservadora del pico de RAM de un punto.

    Los datos se cargan vía parquet temporal (ver run_point), de modo que el
    pico es el máximo de dos picos secuenciales (no simultáneos):
      * escritura del parquet en Python: numpy + Arrow ≈ 2 GB por GB de datos
        + 1 GB de overhead;
      * ejecución de tareas en el driver JVM: RDD (≈data_gb) + working set de
        los jobs + residuo de Python (~2 GB).
    Calibrado con el pico medido de 1M×500 (~9 GB Python en la escritura).
    """
    data_gb = (n * nf * 8) / 1e9
    python_peak_gb = 2.0 * data_gb + 1.0
    driver_usage_gb = 2.0 * data_gb + 4.0
    peak_gb = max(python_peak_gb, driver_usage_gb) + 2.0
    return peak_gb * 1e9


def _run_subprocess(n: int, nf: int, cfg: dict, json_path: Path, log: _Log) -> bool:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--point", str(n), str(nf),
        "--json", str(json_path),
        "--max-features", str(cfg["max_features"]),
        "--screen-top-k", str(cfg["screen_top_k"]),
        "--significance", cfg["significance"],
        "--seed", str(cfg["seed"]),
    ]
    print(f"→ punto n={n:,} N={nf} (JVM nuevo)…")
    log.write(f"→ punto n={n:,} N={nf} iniciado")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(line)
            log.write(line)
    rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        print(f"✗ punto n={n:,} N={nf} falló (exit {rc}, {elapsed:.0f} s)")
        log.write(f"✗ punto n={n:,} N={nf} falló rc={rc} {elapsed:.0f}s")
        return False
    print(f"✓ punto n={n:,} N={nf} completado ({elapsed:.0f} s)")
    log.write(f"✓ punto n={n:,} N={nf} completado {elapsed:.0f}s")
    return True


def main_orchestrator(args: argparse.Namespace) -> int:
    _setup_jdk()
    cfg = _cfg_from_args(args)
    json_path = Path(args.json)
    html_path = Path(args.html)
    log_path = Path(args.log)

    n_list = args.n or (QUICK_N if args.quick else FULL_N)
    nf_list = args.N or (QUICK_NF if args.quick else FULL_NF)

    doc = load_doc(json_path)
    meta = doc.setdefault("meta", {})
    meta["version"] = 2
    meta["created"] = meta.get("created") or _now_iso()
    meta["machine"] = machine_info()
    meta["versions"] = _versions()
    meta["config"] = cfg
    meta["grid"] = {"n": n_list, "N": nf_list}
    save_doc(doc, json_path)

    budget = args.ram_budget_gb
    if budget is None:
        ram = meta["machine"].get("ram_gb")
        # 85 % de la RAM: deja ~5 GB para el SO y procesos, y permite correr
        # puntos cuyo pico real (~28 GB en 1M×500) cabe en la máquina.
        budget = round(ram * 0.85, 1) if ram else 24.0

    log = _Log(log_path)
    print(f"SparkMIM benchmark — rejilla {len(n_list)}×{len(nf_list)} = {len(n_list) * len(nf_list)} puntos")
    print(f"Presupuesto RAM por punto: {budget} GB | JSON: {json_path}")
    log.write(f"=== inicio rejilla {len(n_list)}x{len(nf_list)} (budget {budget} GB) ===")

    total = len(n_list) * len(nf_list)
    i = 0
    for nf in nf_list:
        for n in n_list:
            i += 1
            tag = f"[{i}/{total}] n={n:,} N={nf}"
            doc = load_doc(json_path)
            done = {(p.get("n"), p.get("N")) for p in doc["points"] if p.get("n") is not None}
            if (n, nf) in done:
                print(f"{tag} → ya completado, se omite")
                log.write(f"{tag} ya completado")
                continue
            est_gb = est_peak_bytes(n, nf) / 1e9
            if est_gb > budget:
                reason = f"RAM estimada {est_gb:.1f} GB > presupuesto {budget} GB"
                print(f"{tag} → omitido ({reason})")
                log.write(f"{tag} omitido: {reason}")
                meta = doc.setdefault("meta", {})
                skipped = meta.setdefault("skipped", [])
                if not any(s.get("n") == n and s.get("N") == nf for s in skipped):
                    skipped.append({"n": n, "N": nf, "reason": reason})
                save_doc(doc, json_path)
                continue
            ok = _run_subprocess(n, nf, cfg, json_path, log)
            doc = load_doc(json_path)
            meta = doc.setdefault("meta", {})
            done = {(p.get("n"), p.get("N")) for p in doc["points"] if p.get("n") is not None}
            failed = meta.get("failed", [])
            if ok and (n, nf) not in done:
                failed = [f for f in failed if not (f.get("n") == n and f.get("N") == nf)]
            if not ok:
                if not any(f.get("n") == n and f.get("N") == nf for f in failed):
                    failed.append({"n": n, "N": nf})
            meta["failed"] = failed
            meta["updated"] = _now_iso()
            save_doc(doc, json_path)

    doc = load_doc(json_path)
    doc["meta"]["updated"] = _now_iso()
    save_doc(doc, json_path)
    n_done = len(doc["points"])
    log.write(f"=== fin: {n_done} puntos ===")
    log.close()
    print(f"\n{len(doc['points'])} puntos en {json_path}")

    if not args.no_report:
        from plot_grid import build_report
        build_report(json_path, html_path)
        print(f"Informe: {html_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark de escala n × N de SparkMIM (JVM nuevo por punto).",
    )
    ap.add_argument("--point", nargs=2, type=int, metavar=("NROWS", "NFEAT"),
                    help="modo trabajador: corre un único punto")
    ap.add_argument("--n", nargs="+", type=int, help="filas por punto (rejilla)")
    ap.add_argument("--N", nargs="+", type=int, help="features por punto (rejilla)")
    ap.add_argument("--quick", action="store_true", help="rejilla pequeña de verificación (4 puntos)")
    ap.add_argument("--plot-only", action="store_true", help="solo regenera el HTML desde el JSON")
    ap.add_argument("--no-report", action="store_true", help="no genera el HTML al final")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="JSON de salida")
    ap.add_argument("--html", default=str(DEFAULT_HTML), help="HTML de salida")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="log de consola")
    ap.add_argument("--ram-budget-gb", type=float, default=None,
                    help="presupuesto de RAM por punto (por defecto 85%% de la RAM física)")
    ap.add_argument("--max-features", type=int, default=CONFIG["max_features"])
    ap.add_argument("--screen-top-k", type=int, default=CONFIG["screen_top_k"])
    ap.add_argument("--significance", default=CONFIG["significance"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    return ap.parse_args(argv)


def _fix_stdio() -> None:
    """Consolas legacy (cp1252) no soporta '→' y similares; forzamos UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _fix_temp() -> None:
    """Apunta TEMP/TMP a un directorio del workspace.

    En algunos entornos (p. ej. sesiones sandbox) los procesos descendientes
    no pueden escribir en %TEMP% del usuario; el launcher de Spark
    (spark-class2.cmd) redirige su salida a %TEMP% y fallaría con
    'Acceso denegado'. Un directorio dentro del workspace siempre es
    escribible y es inofensivo en una sesión normal.
    """
    tmp = BENCH_DIR / ".tmp"
    tmp.mkdir(exist_ok=True)
    os.environ["TEMP"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    # El executor de Spark lanza los workers Python resolviendo 'python' por
    # PATH; en Windows hay un alias de Microsoft Store que falla. Fijamos el
    # Python del venv (el que ejecuta este script) de forma explícita.
    os.environ["PYSPARK_PYTHON"] = sys.executable


def main(argv=None) -> int:
    _fix_stdio()
    _fix_temp()
    args = _parse_args(argv)
    if args.point:
        return main_worker(args)
    if args.plot_only:
        from plot_grid import build_report
        build_report(Path(args.json), Path(args.html))
        print(f"Informe: {args.html}")
        return 0
    return main_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
