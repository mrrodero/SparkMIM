# Benchmark de escala — SparkMIM

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
- Presupuesto de RAM por defecto: **70 % de la RAM física**; los puntos cuya
  RAM estimada lo supera se omiten y se marcan en el informe
  (`--ram-budget-gb <GB>` para cambiarlo).
- Config base (comparable con los benchmarks anteriores):
  `max_features=20, screen_top_k=200, significance="chi2", seed=42`.
- Modo trabajador (interno, usado por el orquestador):
  `python bench_grid.py --point 1000000 200`.
