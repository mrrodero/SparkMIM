# Benchmark de escala — SparkMIM

## En Databricks 15.4 LTS (wheel + staging)

La implementación del benchmark vive en el paquete (`sparkmim.benchmarks`,
empaquetada en el wheel): el notebook **no necesita clonar el repo ni
`sys.path`**.

### Instalación del wheel

```python
%pip install --no-deps /Volumes/<catalog>/<schema>/<volume>/sparkmim-0.2.0-py3-none-any.whl
# Si ya importó SparkMIM en la sesión: reiniciar Python antes de continuar.
```

`--no-deps`: no reemplazar los paquetes administrados por Databricks.
Matriz validada (DBR 15.4 LTS, Spark 3.5.0): Python 3.11, `pyspark==3.5.0`,
`numpy==1.23.5`, `pandas==1.5.3`, `scipy==1.11.1`, `pyarrow==14.0.1`.

### Ejecución de un punto (Spark Connect)

`run_point` stagea la entrada en **parquet por lotes** (pandas/Arrow) para
no superar el payload RPC de Spark Connect:

```python
from sparkmim.benchmarks import run_point, default_staging_path

staging = default_staging_path()   # dbfs:/tmp/sparkmim-benchmark/<uuid>
try:
    row = run_point(
        spark,                       # cliente Spark Connect del notebook
        n=10_000_000, N=25,
        n_informative=10, n_redundant=5,
        seed=42, max_features=20,
        staging_path=staging,
    )
finally:
    dbutils.fs.rm(staging, recurse=True)   # el paquete no limpia solo
```

- **Staging**: la entrada se genera chunk a chunk (`generate_chunks`) y
  cada chunk se envía como `LocalRelation` (payload ≤128 MiB raw por
  defecto, `batch_bytes` configurable; a N=25 ≈ 640k filas/chunk, ~1.9× de
  margen frente al límite RPC de 256 MiB). Por defecto (`single_write=True`)
  todos los chunks se escriben en **un único job** (union balanceada + un
  solo commit); el coste de staging medido en DBR escalaba con el *número*
  de jobs de escritura (~25–80 s por job de commit, frente a ~70 ms de
  trabajo real), así que un solo job lo reduce a una sola vez. El precio:
  el driver JVM retiene la tabla completa como `LocalRelation` (`n × (N+1) × 8`
  bytes; ≤16,1 GB en el punto más grande de la rejilla). Con
  `single_write=False` se escribe un job por chunk (memoria del driver
  acotada a un chunk; útil en drivers con poca RAM). La entrada se lee una
  sola vez con `spark.read.parquet`. **No elevar**
  `spark.rpc.message.maxSize`: el staging ya resuelve el payload.
- **Limpieza**: el paquete no asume el global `dbutils` del notebook y no
  borra paths compartidos arbitrarios; el entorno limpia (receta de arriba).
  Si una ejecución falla a mitad de staging, el reintento debe usar un
  directorio nuevo (`default_staging_path()` genera uno único por llamada)
  para no duplicar partes parquet.
- **Permiso DBFS**: comprobar que `dbfs:/tmp` está habilitado y escribible
  en el compute (`dbutils.fs.ls("dbfs:/tmp")`). Si la raíz de DBFS no está
  disponible, usar un UC Volume: `staging_path="dbfs:/Volumes/<c>/<s>/<v>/sparkmim-benchmark/<uuid>"`.
- **Memoria del driver**: `generate()` conserva los arrays NumPy completos
  en el driver (~2,08 GB decimales a 10M×25) más el buffer del lote en
  curso. Baseline de 10M×25 en C2 (8×E4as_v5, driver E16as_v5): pico
  `driver_mem_mb` 2465 MB. Dimensionar el driver con margen suficiente.
- **Semántica de tiempos**: `total_s` mide solo el `fit` del selector
  (empieza después de crear la entrada, igual que en los benchmarks
  anteriores, comparable con las tablas existentes); `staging_s` informa
  aparte del tiempo de crear + stagear la entrada. `staging_path` en el
  resultado indica el directorio usado.
- **Paridad**: el generador del wheel (`generate`) produce exactamente los
  mismos valores que el generador local (misma secuencia RNG; ver
  `tests/test_benchmark_synthetic.py`).

## Benchmark de escala local (CLI)

El mismo benchmark (el del paquete) con sesión Spark local y staging en un
directorio temporal con limpieza automática:

```bash
python benchmarks/bench_scale.py --quick                 # n{1e4,1e5} × N{50,100}
python benchmarks/bench_scale.py --n 100000 --N 100      # un punto
python benchmarks/bench_scale.py --csv out.csv           # rejilla completa
```

Escribe un CSV con las mismas columnas que `run_point` (etapas, memoria
driver, informativas recuperadas). **No usar en notebook de Databricks**
(crea una sesión local): allí, importar `sparkmim.benchmarks.run_point` del
wheel, como arriba.

## Rejilla local (CLI)

Rejilla completa: **n ∈ {10k, 50k, 100k, 250k, 500k, 1M, 2.5M, 5M, 10M} ×
N ∈ {25, 50, 100, 200, 500, 1000}** (54 puntos), sobre datos sintéticos con
estructura plantada (10 informativas + 5 redundantes + ruido).

Cada punto corre en un **JVM nuevo** (subproceso, heap 8 g), de modo que la
RAM del driver se libera entre puntos. El JSON se escribe de forma
**incremental** (a prueba de cortes) y la rejilla es **reanudable**: los
puntos ya completados se saltan. Al final se genera un **informe HTML
interactivo** con tema oscuro (Plotly, autocontenido).

## Uso

Doble clic en `run_benchmark.ps1`, o desde PowerShell:

```powershell
cd SparkMIM\benchmarks
..\..\.venv\Scripts\python.exe bench_grid.py           # rejilla completa
python bench_grid.py --quick                           # 4 puntos de verificación
python bench_grid.py --n 100000 --N 200                # puntos sueltos
python bench_grid.py --plot-only                       # regenerar solo el HTML
```

Salidas:

| Archivo | Contenido |
| --- | --- |
| `bench_grid.json` | Puntos + metadatos (máquina, versiones, config, omitidos) |
| `bench_grid_report.html` | Informe interactivo (KPIs, heatmaps, etapas, tabla) |
| `bench_grid.log` | Consola con marca de tiempo |

## Notas

- `JAVA_HOME` se fija automáticamente a `jdk17` (JDK 17, requisito de PySpark).
- **Windows**: `bench_scale.py` y `conftest.py` (tests) configuran solos
  `PYSPARK_PYTHON`, `HADOOP_HOME`/`PATH` y `TEMP`/`TMP`. Si falta winutils
  (`winutils/bin/winutils.exe`), Spark no puede leer/escribir el filesystem
  local: ejecutar `python tools/fetch_winutils.py` una vez (descarga
  ~0,5 MB; gitignoreado; ver el docstring del script para la fuente y la
  verificación de compatibilidad con Hadoop 3.3.4). En Databricks no hace
  falta nada de esto (workers Linux).
- Presupuesto de RAM por defecto: **70 % de la RAM física**; los puntos cuya
  RAM estimada lo supera se omiten y se marcan en el informe
  (`--ram-budget-gb <GB>` para cambiarlo).
- Config base (comparable con los benchmarks anteriores):
  `max_features=20, screen_top_k=200, significance="chi2", seed=42`.
- Modo trabajador (interno, usado por el orquestador):
  `python bench_grid.py --point 1000000 200`.
