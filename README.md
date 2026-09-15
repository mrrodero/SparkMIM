# SparkMIM

Selección de features distribuida por información mutua en PySpark 3.5 — sin
Scala, sin sklearn. Calcula tablas de contingencia en **pases únicos** (evita
los O(N) jobs que estrangulan las implementaciones previas), controla la
significancia con FDR, soporta regresión/multiclase y ofrece un modo KSG
opcional para variables continuas. Evaluación integrada agnóstica al modelo
(GBT/XGBoost/LightGBM en Spark).

> **English:** see [README.en.md](README.en.md).
> **Diseño completo:** [DESIGN.md](DESIGN.md) (ES) / [docs/DESIGN.en.md](docs/DESIGN.en.md) (EN).

---

## Instalación

Requiere Python ≥ 3.11 y un JDK 17 (Spark 3.5 + Arrow).

```bash
python -m venv venv
# Windows: venv\Scripts\activate ; Linux/macOS: source venv/bin/activate
pip install wheel
pip install --no-build-isolation -e .
# extras opcionales de evaluación:
pip install --no-build-isolation -e ".[xgb]"    # xgboost4j-spark
pip install --no-build-isolation -e ".[lgbm]"   # lightgbm4j-spark
```

> **Nota (JDK):** Spark 3.5 con Arrow requiere JDK 17. Si tu JDK de sistema es
> 21, apunta `JAVA_HOME` a un JDK 17 antes de lanzar Spark. El `conftest.py`
> y `benchmarks/bench_scale.py` lo hacen automáticamente si existe `jdk17/`.

---

## Quickstart

```python
from pyspark.sql import SparkSession
from sparkmim import JMIMSelector

spark = SparkSession.builder.master("local[8]").getOrCreate()
df = spark.read.parquet("datos.parquet")   # o cualquier fuente

sel = JMIMSelector(
    target="y",          # columna objetivo
    max_features=50,     # nº máximo de features a seleccionar
    screen_top_k=200,    # candidatas tras el screening
    significance="fdr",  # "chi2" | "permutation" | "fdr" | "none"
    seed=42,
)
model = sel.fit(df)

model.selected_features   # list[str]: features seleccionadas (orden de selección)
model.scores_             # puntaje del criterio por ronda (nats)
model.ranking_            # ranking completo (nombre, MI univariante)
model.transform(df)       # df solo con features seleccionadas + target

# Evaluación (Hito 6): AUC/R² vs baseline + curva de eficiencia.
report = model.report(df, model="gbt")
report.metric_selected    # AUC (cl.) o R² (reg.) con las seleccionadas
report.metric_all         # AUC/R² con todas las features (baseline)
report.efficiency_curve   # [(k, métrica), ...] al añadir features por ranking
```

### Variantes de criterio

| Clase | Criterio |
|---|---|
| `JMIMSelector` (default) | `min_{Xi∈S} CMI(X;Y\|Xi)` |
| `CMIMSelector` | `CMI(X;Y\|S_m)`, `S_m` = top-m por MI (m=2) |
| `MRMRSelector` | `MI(X;Y) − (1/\|S\|)·Σ MI(X;Xi)` |
| `MIMSelector` | `MI(X;Y) − Σ MI(X;Xi)` |

Todas aceptan los mismos `config_kwargs`. El criterio se fija por la clase;
para JMIM/CMIM/mRMR/mIM se usa `InfoSelector(criterion=...)` directamente.

### Modo KSG (opcional, continuas)

Para variables continuas donde el binning no es aceptable, usa el estimador
kNN de Kraskov-Stögbauer-Grassberger:

```python
sel = JMIMSelector(target="y", estimator="ksg", ksg_k=10, max_features=20)
model = sel.fit(df)
```

> **Trade-off:** el modo KSG submuestrea ≤ `ksg_subsample` (250k) filas al
> driver y calcula MI/CMI por kNN allí. Es la única ruta con kNN global en
> driver; adecuada para n moderado, no para 10⁷.

---

## Referencia de API

### `sparkmim.InfoSelector(criterion="jmim", **config_kwargs)`

Orquesta las etapas 0–3 (ver [DESIGN.md](DESIGN.md)). `fit(df) -> SelectorModel`.

### `sparkmim.SelectorModel`

| Atributo | Tipo | Descripción |
|---|---|---|
| `selected_features` | `list[str]` | Features seleccionadas (orden de selección). |
| `scores_` | `list[float]` | Puntaje del criterio por ronda (nats). |
| `ranking_` | `list[(str, float)]` | Ranking completo por MI univariante. |
| `criterion` | `str` | Criterio usado. |
| `n_rows` | `int` | Nº de filas de `df_prep`. |
| `target` | `str` | Nombre de la columna target. |
| `timings_` | `dict[str, float]` | Wall-time (s) por etapa (si se midió). |

| Método | Descripción |
|---|---|
| `transform(df)` | `df` con solo las features seleccionadas + target. |
| `report(df, model="gbt")` | Evaluación (Hito 6): AUC/R² vs baseline + curva de eficiencia. |

### `sparkmim.SelectorConfig`

Ver la [tabla de configuración](#configuración).

---

## Configuración

`SelectorConfig` (campos por defecto):

| Campo | Tipo | Defecto | Descripción |
|---|---|---|---|
| `target` | `str` | — | Columna objetivo (requerido). |
| `criterion` | `str` | `"jmim"` | *(fijado por la clase; no aquí)* |
| `estimator` | `str` | `"histogram"` | `"histogram"` (binning) o `"ksg"` (kNN). |
| `max_features` | `int` | `50` | Nº máximo de features a seleccionar. |
| `screen_top_k` | `int` | `200` | Candidatas tras el screening (top-K por MI). |
| `significance` | `str` | `"fdr"` | `"chi2"` \| `"permutation"` \| `"fdr"` \| `"none"`. |
| `alpha` | `float` | `0.05` | Nivel de significancia (FDR/χ²/permutación). |
| `n_permutations` | `int` | `200` | Permutaciones para el test de permutación. |
| `bins` | `int` | `10` | Bins cuantiles por feature numérica. |
| `bins_auto` | `bool` | `False` | Si `True`, `bins = clamp(round(log2 n), 4, 20)`. |
| `max_categories` | `int` | `50` | Top-C categorías por frecuencia (+ "other"). |
| `missing` | `str` | `"category"` | `"category"` (código dedicado) o `"drop"`. |
| `subsample` | `int` | `1_000_000` | Filas máx. para las tablas conjuntas (etapa 2). |
| `ksg_k` | `int` | `10` | k del estimador KSG. |
| `ksg_subsample` | `int` | `250_000` | Filas máx. para el modo KSG. |
| `cmim_m` | `int` | `2` | Tamaño de `S_m` en CMIM exacto. |
| `cmim_approx` | `str` | `"exact"` | `"exact"` o `"max_min"` (ruta CMIM→JMIM). |
| `min_score` | `float` | `1e-4` | Parada temprana: puntaje mínimo del criterio. |
| `max_cache_cells` | `int` | `5_000_000` | Presupuesto de celdas del caché (reduce K). |
| `seed` | `int` | `0` | Semilla (determinismo). |

---

## Rendimiento

Diseño de **pases únicos** (2 pasadas completas + 1 mapeo sin shuffle + 1 pase
sobre subsample + bucle greedy solo en driver). Meta: **n=10⁶, N=200 < 15 min**
end-to-end en `local[8]` (validada por el benchmark, no por asunción).

Escala: lineal en `n` y en `N` (etapa 1) / `K²` (etapa 2).

### Benchmark de escala

```bash
# Rejilla completa (n ∈ {10⁵,10⁶,10⁷}, N ∈ {100,200,500}) en local[8]:
python benchmarks/bench_scale.py --csv bench_scale.csv

# Rejilla pequeña (rápido):
python benchmarks/bench_scale.py --quick --csv bench_quick.csv

# Un punto concreto:
python benchmarks/bench_scale.py --n 1000000 --N 200 --csv bench_point.csv
```

El CSV incluye: `n, N, n_informative, n_redundant, etapa0_s, etapa1_s,
etapa2_s, etapa3_s, total_s, driver_mem_mb, n_selected,
informative_recuperadas`.

El generador sintético (`benchmarks/synthetic.py`) planta estructura conocida
(informativas + redundantes + ruido) para validar la recuperación.

---

## Estructura del proyecto

```
sparkmim/
├── src/sparkmim/
│   ├── config.py        # SelectorConfig
│   ├── schema.py        # detección de esquema
│   ├── preprocess.py    # pase 0a/0b (bins, missing, cardinalidad)
│   ├── info/
│   │   ├── entropy.py   # entropía / MI / CMI (tablas)
│   │   └── ksg.py       # estimador KSG (kNN)
│   ├── tables.py        # tablas conjuntas en pase único + caché
│   ├── screen.py        # etapa 1: screening + significancia
│   ├── significance.py  # χ², permutación distribuida, BH-FDR
│   ├── criteria.py      # JMIM/JMI/mRMR/mIM/CMIM
│   ├── selector.py      # orquestación de etapas 0-3
│   ├── model.py         # SelectorModel
│   └── evaluate.py      # evaluación agnóstica al modelo
├── benchmarks/
│   ├── synthetic.py     # generador sintético
│   └── bench_scale.py   # benchmark de escala
├── tests/               # pytest (unitarios + e2e)
├── DESIGN.md            # diseño (ES)
├── docs/DESIGN.en.md    # diseño (EN)
├── README.md            # (ES)
└── README.en.md         # (EN)
```

---

## Referencias

- Kraskov, Stögbauer & Grassberger (2004). *Estimating mutual information
  for continuous variables.* [infomeasure](https://infomeasure.readthedocs.io/en/0.5.0/guide/mutual_information/kraskov_stoegbauer_grassberger/).
- Bañón-Garrote et al. (2020). *JMIM: Joint Mutual Information Maximization.*
- Calomiris et al. (2023). *CMIM: Conditional Mutual Information Maximization.*
- Peng et al. (2005). *Feature selection using mutual information.* (mRMR/mIM).
- Benjamini & Hochberg (1995). *Controlling the false discovery rate* (BH-FDR).
