# SparkMIM

Distributed mutual-information feature selection in PySpark 3.5 — no Scala, no
sklearn. Computes contingency tables in **single passes** (avoiding the O(N)
jobs that strangle previous implementations), controls significance with FDR,
supports regression/multiclass, and offers an optional KSG mode for continuous
variables. Integrated model-agnostic evaluation (GBT/XGBoost/LightGBM in
Spark).

> **Español:** see [README.md](README.md).
> **Full design:** [DESIGN.md](DESIGN.md) (ES) / [docs/DESIGN.en.md](docs/DESIGN.en.md) (EN).

---

## Installation

Requires Python ≥ 3.11 and a JDK 17 (Spark 3.5 + Arrow).

```bash
python -m venv venv
# Windows: venv\Scripts\activate ; Linux/macOS: source venv/bin/activate
pip install wheel
pip install --no-build-isolation -e .
# optional evaluation extras:
pip install --no-build-isolation -e ".[xgb]"    # xgboost4j-spark
pip install --no-build-isolation -e ".[lgbm]"   # lightgbm4j-spark
```

> **Note (JDK):** Spark 3.5 with Arrow requires JDK 17. If your system JDK is
> 21, point `JAVA_HOME` at a JDK 17 before launching Spark. `conftest.py` and
> `benchmarks/bench_scale.py` do this automatically if `jdk17/` exists.

---

## Quickstart

```python
from pyspark.sql import SparkSession
from sparkmim import JMIMSelector

spark = SparkSession.builder.master("local[8]").getOrCreate()
df = spark.read.parquet("data.parquet")   # or any source

sel = JMIMSelector(
    target="y",          # target column
    max_features=50,     # max number of features to select
    screen_top_k=200,    # candidates after screening
    significance="fdr",  # "chi2" | "permutation" | "fdr" | "none"
    seed=42,
)
model = sel.fit(df)

model.selected_features   # list[str]: selected features (selection order)
model.scores_             # criterion score per round (nats)
model.ranking_            # full ranking (name, univariate MI)
model.transform(df)       # df with only selected features + target

# Evaluation (Hito 6): AUC/R² vs baseline + efficiency curve.
report = model.report(df, model="gbt")
report.metric_selected    # AUC (cl.) or R² (reg.) with the selected features
report.metric_all         # AUC/R² with all features (baseline)
report.efficiency_curve   # [(k, metric), ...] as features are added by ranking
```

### Criterion variants

| Class | Criterion |
|---|---|
| `JMIMSelector` (default) | `min_{Xi∈S} CMI(X;Y\|Xi)` |
| `CMIMSelector` | `CMI(X;Y\|S_m)`, `S_m` = top-m by MI (m=2) |
| `MRMRSelector` | `MI(X;Y) − (1/\|S\|)·Σ MI(X;Xi)` |
| `MIMSelector` | `MI(X;Y) − Σ MI(X;Xi)` |

All accept the same `config_kwargs`. The criterion is fixed by the class; for
JMIM/CMIM/mRMR/mIM use `InfoSelector(criterion=...)` directly.

### KSG mode (optional, continuous)

For continuous variables where binning is not acceptable, use the
Kraskov-Stögbauer-Grassberger kNN estimator:

```python
sel = JMIMSelector(target="y", estimator="ksg", ksg_k=10, max_features=20)
model = sel.fit(df)
```

> **Trade-off:** KSG mode subsamples ≤ `ksg_subsample` (250k) rows to the
> driver and computes MI/CMI by kNN there. It is the only route with global
> kNN in the driver; suitable for moderate n, not for 10⁷.

---

## API reference

### `sparkmim.InfoSelector(criterion="jmim", **config_kwargs)`

Orchestrates stages 0–3 (see [docs/DESIGN.en.md](docs/DESIGN.en.md)).
`fit(df) -> SelectorModel`.

### `sparkmim.SelectorModel`

| Attribute | Type | Description |
|---|---|---|
| `selected_features` | `list[str]` | Selected features (selection order). |
| `scores_` | `list[float]` | Criterion score per round (nats). |
| `ranking_` | `list[(str, float)]` | Full ranking by univariate MI. |
| `criterion` | `str` | Criterion used. |
| `n_rows` | `int` | Number of rows in `df_prep`. |
| `target` | `str` | Target column name. |
| `timings_` | `dict[str, float]` | Wall-time (s) per stage (if measured). |

| Method | Description |
|---|---|
| `transform(df)` | `df` with only the selected features + target. |
| `report(df, model="gbt")` | Evaluation (Hito 6): AUC/R² vs baseline + efficiency curve. |

### `sparkmim.SelectorConfig`

See the [configuration table](#configuration).

---

## Configuration

`SelectorConfig` (default fields):

| Field | Type | Default | Description |
|---|---|---|---|
| `target` | `str` | — | Target column (required). |
| `criterion` | `str` | `"jmim"` | *(set by the class; not here)* |
| `estimator` | `str` | `"histogram"` | `"histogram"` (binning) or `"ksg"` (kNN). |
| `max_features` | `int` | `50` | Max number of features to select. |
| `screen_top_k` | `int` | `200` | Candidates after screening (top-K by MI). |
| `significance` | `str` | `"fdr"` | `"chi2"` \| `"permutation"` \| `"fdr"` \| `"none"`. |
| `alpha` | `float` | `0.05` | Significance level (FDR/χ²/permutation). |
| `n_permutations` | `int` | `200` | Permutations for the permutation test. |
| `bins` | `int` | `10` | Quantile bins per numeric feature. |
| `bins_auto` | `bool` | `False` | If `True`, `bins = clamp(round(log2 n), 4, 20)`. |
| `max_categories` | `int` | `50` | Top-C categories by frequency (+ "other"). |
| `missing` | `str` | `"category"` | `"category"` (dedicated code) or `"drop"`. |
| `subsample` | `int` | `1_000_000` | Max rows for the joint tables (stage 2). |
| `ksg_k` | `int` | `10` | k of the KSG estimator. |
| `ksg_subsample` | `int` | `250_000` | Max rows for KSG mode. |
| `cmim_m` | `int` | `2` | Size of `S_m` in exact CMIM. |
| `cmim_approx` | `str` | `"exact"` | `"exact"` or `"max_min"` (CMIM→JMIM route). |
| `min_score` | `float` | `1e-4` | Early stopping: minimum criterion score. |
| `max_cache_cells` | `int` | `5_000_000` | Cache cell budget (reduces K). |
| `seed` | `int` | `0` | Seed (determinism). |

---

## Performance

**Single-pass** design (2 full passes + 1 shuffle-free mapping + 1 pass over a
subsample + a driver-only greedy loop). Target: **n=10⁶, N=200 < 15 min**
end-to-end on `local[8]` (validated by the benchmark, not assumed).

Scaling: linear in `n` and in `N` (stage 1) / `K²` (stage 2).

### Scale benchmark

```bash
# Full grid (n ∈ {10⁵,10⁶,10⁷}, N ∈ {100,200,500}) on local[8]:
python benchmarks/bench_scale.py --csv bench_scale.csv

# Small grid (fast):
python benchmarks/bench_scale.py --quick --csv bench_quick.csv

# A single point:
python benchmarks/bench_scale.py --n 1000000 --N 200 --csv bench_point.csv
```

The CSV includes: `n, N, n_informative, n_redundant, etapa0_s, etapa1_s,
etapa2_s, etapa3_s, total_s, driver_mem_mb, n_selected,
informative_recuperadas`.

The synthetic generator (`benchmarks/synthetic.py`) plants known structure
(informative + redundant + noise) to validate recovery.

---

## Project structure

```
sparkmim/
├── src/sparkmim/
│   ├── config.py        # SelectorConfig
│   ├── schema.py        # schema detection
│   ├── preprocess.py    # pass 0a/0b (bins, missing, cardinality)
│   ├── info/
│   │   ├── entropy.py   # entropy / MI / CMI (tables)
│   │   └── ksg.py       # KSG estimator (kNN)
│   ├── tables.py        # joint tables in a single pass + cache
│   ├── screen.py        # stage 1: screening + significance
│   ├── significance.py  # χ², distributed permutation, BH-FDR
│   ├── criteria.py      # JMIM/JMI/mRMR/mIM/CMIM
│   ├── selector.py      # stage 0-3 orchestration
│   ├── model.py         # SelectorModel
│   └── evaluate.py      # model-agnostic evaluation
├── benchmarks/
│   ├── synthetic.py     # synthetic generator
│   └── bench_scale.py   # scale benchmark
├── tests/               # pytest (unit + e2e)
├── DESIGN.md            # design (ES)
├── docs/DESIGN.en.md    # design (EN)
├── README.md            # (ES)
└── README.en.md         # (EN)
```

---

## References

- Kraskov, Stögbauer & Grassberger (2004). *Estimating mutual information
  for continuous variables.* [infomeasure](https://infomeasure.readthedocs.io/en/0.5.0/guide/mutual_information/kraskov_stoegbauer_grassberger/).
- Bañón-Garrote et al. (2020). *JMIM: Joint Mutual Information Maximization.*
- Calomiris et al. (2023). *CMIM: Conditional Mutual Information Maximization.*
- Peng et al. (2005). *Feature selection using mutual information.* (mRMR/mIM).
- Benjamini & Hochberg (1995). *Controlling the false discovery rate* (BH-FDR).
