# SparkMIM Design

Full design: math, justification vs SOTA, and complexity analysis.

> **Español:** see [DESIGN.md](DESIGN.md).
> **Quickstart/API:** [README.en.md](../README.en.md).

---

## 1. Problem

Select an informative, non-redundant subset of features at distributed scale
(n up to 10⁷, N up to 500) in PySpark, without Scala and without sklearn.

Previous approaches (e.g. MI by `groupBy` per pair) launch **O(N) jobs** (one
job per feature or per pair), which becomes the bottleneck in the driver
(scheduling, serialization). SparkMIM removes that bottleneck by computing
contingency tables in **single passes**.

---

## 2. Algorithm (passes over the data)

### Stage 0 — Schema and preprocessing (2 passes + 1 shuffle-free mapping)

1. **Numeric vs categorical detection** by dtype (optional manual override).
2. **Pass 0a (ONE `mapInPandas` over the original df):** per partition and per
   feature, quantiles `part[fid].quantile([i/b…])` (numeric) and
   `value_counts()` (categorical) → emits `(fid, i, q, n_part)` and
   `(fid, val, count)` → a single `groupBy` → driver. **Avoids N calls to
   `approxQuantile` and N `groupBy` (O(N) jobs).**
3. **Driver:** global quantiles weighted by `n_part` → bin boundaries
   (`b=10`, or `auto`: `clamp(round(log2 n), 4, 20)`); top-C=50 categories by
   frequency + "other" code (bounded tables; documented bias).
4. **Pass 0b (mapping, no shuffle):** `when`/`map` expressions per column
   applying bins and codes; missing → dedicated code (or drop).
5. **Target Task:** declared by the user in `SelectorConfig.task`
   (`"auto"` | `"classifier_binary"` | `"classifier_multiclass"` |
   `"continuous"`), resolved once with the shared `schema.resolve_task` rule
   validating against the data, and carried in `SelectorModel.task`
   (ADR-0002). Representation: classification → one code per distinct value
   (no truncation); continuous → quantile binning `b_y=10`.

### Stage 1 — Univariate screening (1 pass)

A single `mapInPandas` emits the crosstabs `(fid, code_x, code_y, count)` for
each feature → a `groupBy` → driver. Per feature:

- Contingency table `P(x, y)` (dense, `n_codes × n_y`).
- **Univariate MI** `I(X;Y)` (nats).
- **Significance:** `SignificanceTest` interface with three adapters (χ²,
  distributed permutation, no test) + FDR control (see §4).
- **Candidates C:** top-`screen_top_k` by MI among those passing the
  significance filter. The cut policy is the shared pure
  `screen.select_candidates` (the KSG mode reuses it without the significance
  filter, pending).

### Stage 2 — Joint tables (1 pass over a subsample)

Over a subsample ≤ `subsample` (10⁶) rows — the shared `selector.subsample`
policy, the same one the KSG mode applies with `ksg_subsample` — a
`mapInPandas` emits the joint tables of **pairs** and **triples**
`(x_i, x_j, y)` for the candidates → driver → `TableCache`. This allows
computing CMI exactly (or approximately) in stage 3 without touching the data
again.

### Stage 3 — Greedy selection (driver only)

Single greedy loop (`selection.greedy_select`) behind the
`InformationOracle` seam (`oracles.py`), operating in **candidate positions**
(0..K−1). Two adapters (ADR-0001):

- **`HistogramOracle`:** queries over the `TableCache` (univariate MI, pair
  MI, triple CMI). `cmi_set_all` runs the distributed CMIM pass
  (`mapInPandas` over the subsample) when the criterion is exact CMIM.
- **`KsgOracle`:** MI/CMI by kNN (KSG estimator) over the driver subsample,
  no binning.

- **Round 1:** argmax univariate MI.
- **Subsequent rounds:** score each unselected candidate with the criterion
  (JMIM/CMIM/mRMR/mIM) and pick the argmax.
- **Stopping:** `best_score < min_score` or `max_features` reached.
- **Ranking:** full population (N features) by univariate MI (descending),
  derived by the shared pure `screen.rank`: screening MI in the histogram
  estimator (no extra cost) and kNN MI in the KSG estimator.

### Module map

```
preprocess.py   schema + preprocessing (stage 0)
screen.py       univariate screening (stage 1)
significance.py SignificanceTest seam + adapters (stage 1)
tables.py       joint_counts (counting core) + tables + TableCache (stage 2, oriented accessors)
oracles.py      InformationOracle + adapters (seam, stage 3)
selection.py    single greedy loop (stage 3, driver)
criteria.py     criterion formulas (pure over the oracle)
info/ksg.py     KSG estimator (MI/CMI by kNN)
info/entropy.py MI/CMI over discrete tables
selector.py     orchestration of stages 0-3
model.py        SelectorModel (result + transform/report)
model_factory.py ModelFactory seam + adapters (evaluation)
evaluate.py     evaluation (AUC/R² + efficiency curve, composes the seam)
```

```
greedy_select (selection.py)
        │  queries
   InformationOracle (oracles.py)   ← seam
      ┌────────────────┴────────────────┐
  HistogramOracle                   KsgOracle
  (TableCache + CMIM pass)         (kNN in driver)
```

---

## 3. Math

### Entropy and MI (discrete tables)

Given the contingency table `P(x, y)`:

```
H(X)   = −Σ_x P(x) log P(x)
H(Y|X) = −Σ_{x,y} P(x,y) log (P(y|x))
I(X;Y) = H(Y) − H(Y|X) = Σ_{x,y} P(x,y) log (P(x,y) / (P(x) P(y)))
```

CMI (conditioned on Z):

```
I(X;Y|Z) = Σ_{x,y,z} P(x,y,z) log (P(x,y,z) P(z) / (P(x,z) P(y,z)))
```

Implementation: `info/entropy.py` (with `np.where` to avoid `0·log 0 = NaN`).

### Greedy criteria

| Criterion | Formula | Oracle query |
|---|---|---|
| **JMIM** | `min_{Xi∈S} I(X;Y\|Xi)` | `cmi_single` |
| **CMIM** | `I(X;Y\|S_m)`, `S_m` = top-m by MI (m=2) | `cmi_set_all` |
| **mRMR** | `I(X;Y) − (1/\|S\|)·Σ_{Xi∈S} I(X;Xi)` | `mi_pair` |
| **mIM** | `I(X;Y) − Σ_{Xi∈S} I(X;Xi)` | `mi_pair` |
| **JMI** | `Σ_{Xi∈S} I(X;Y\|Xi)` | `cmi_single` |

The fast criteria (JMIM/mRMR/mIM/JMI) are pure oracle arithmetic (no Spark
jobs); only exact CMIM touches Spark through `cmi_set_all` (distributed pass
per round).

**Approximate CMIM (`max_min`):** instead of the exact per-round pass, CMIM is
routed to JMIM (`min_{Xi∈S} I(X;Y\|Xi)`), which is a lower bound on the CMI
conditioned on the set. This avoids the distributed CMIM pass in stage 3.

**Effective conditioning set (CMIM):** `S_m \ {x}` (the candidate `x` is
excluded to avoid the degeneracy `I(X;Y|X)=0`).

### KSG estimator (continuous)

For continuous variables, the Kraskov-Stögbauer-Grassberger estimator:

```
I(X;Y) = (1/n) · Σ_i [ ψ(k) + ln(n) − ψ(n_x) − ψ(n_y) ]
```

where `ψ` is the digamma function, `n_x`/`n_y` are the neighbor counts
**including the point itself** within the sphere of radius `ε_i` (distance to
the k-th neighbor in the joint space), and `k` is the number of neighbors
(10 by default). Implementation: `info/ksg.py` (with `scipy.spatial.cKDTree`).

**CMI by identity:** `I(X;Y|Z) = I(XZ;Y) − I(Z;Y)` (with `|Z| ≤ 2` per plan).
The CMI is **not** clamped to 0 (negative = no information); the MI is clamped
to 0.

> **Trade-off:** the KSG estimator subsamples ≤ `ksg_subsample` (250k) rows
> to the driver and computes MI/CMI by kNN there. It is the only route with
> global kNN in the driver; suitable for moderate n, not for 10⁷. For
> classification, the target is coded losslessly per distinct value and
> treated as a numeric variable in the kNN (no binning required).
>
> **Real stage 1:** the KSG mode shares the stage-1 policy with the histogram
> mode — top-K cut by `screen.select_candidates` (without the significance
> filter, pending) and ranking by `screen.rank` — and exposes per-stage
> timings (`etapa0`, `etapa1`, `etapa3` and `total`; there is no stage 2).

---

## 4. Significance

Stage-1 seam: the `SignificanceTest` interface (`significance.py`) — "given
the screening tables and the MI per feature (and, for permutation, the
prepared df), return the p-values and the significance mask per feature".
`screen()` only composes: tables → MI → significance → top-K.

Three adapters (`significance` field):

- **`chi2`** (`Chi2Test`): χ² test on the contingency table (fast,
  asymptotic): `G² = 2·n·MI ~ χ²`.
- **`permutation`** (`PermutationTest`): distributed permutation test (ONE
  `mapInPandas` over a subsample ≤ `permutation_rows`; `B = n_permutations`
  permutations of y per partition, seed `seed + b`; empirical p-value). It is
  only applied to the top-`screen_top_k` by MI (pre-filter); the rest stays
  at p = 1.
- **`none`** (`NoTest`): no filter (p = 1, all significant).

Benjamini-Hochberg FDR control (`fdr_q`, 0.05 by default) is applied inside
each adapter over its p-values.

The candidates are those that pass the significance filter **and** are in the
top-`screen_top_k` by MI.

---

## 5. Justification vs SOTA

| Aspect | Previous approaches | SparkMIM |
|---|---|---|
| **Table computation** | O(N) jobs (one `groupBy` per feature/pair) | **Single passes** (1 `mapInPandas` + 1 `groupBy`) |
| **CMI** | Resampling or pairwise approximation | Joint tables in 1 pass over a subsample |
| **Significance** | Only χ² (driver) | χ² / distributed permutation / no test (+ FDR control) |
| **Continuous** | Fixed binning | Quantile binning + optional KSG estimator |
| **Evaluation** | External (sklearn) | Integrated, model-agnostic (GBT/XGBoost/LightGBM behind the `ModelFactory` seam) |
| **Dependencies** | sklearn, scipy (driver) | Only pyspark, numpy, pandas, scipy (driver) |

**Bottleneck removed:** the scheduling of O(N) jobs in the driver. With single
passes, the number of jobs is constant (independent of N), and the dominant
cost is the computation in workers (linear in n and N).

---

## 6. Complexity analysis

Let `n` = rows, `N` = features, `K` = candidates, `b` = bins, `C` = categories.

| Stage | Time | Jobs | Memory (driver) |
|---|---|---|---|
| **0a** (quantiles/card.) | `O(n·N)` in workers | 1 `mapInPandas` + 1 `groupBy` | `O(N·(b+C))` |
| **0b** (mapping) | `O(n·N)` in workers | 1 mapping (no shuffle) | — |
| **1** (screening) | `O(n·N)` in workers | 1 `mapInPandas` + 1 `groupBy` | `O(N·b·n_y)` |
| **2** (joint tables) | `O(n_sub·K²)` in workers | 1 `mapInPandas` + 1 `groupBy` | `O(K²·b²·n_y)` |
| **3** (greedy) | `O(max_features·K²)` in driver | 0 (exact CMIM: 1 `mapInPandas` per round) | `O(K²)` |

**Scaling:** linear in `n` and in `N` (stage 1) / `K²` (stage 2). The greedy
loop is driver-only and launches no jobs (except the exact-CMIM pass: 1
`mapInPandas` per round).

**Evaluation:** `efficiency_curve` trains one model per ranking prefix (cost
O(|ranking|) trainings); with the full-population ranking it scales with N,
not with K. Training goes through the `ModelFactory` seam
(`model_factory.py`): 1 training per `train` call.

**Target:** n=10⁶, N=200, K=100, `local[8]` → stage 1 ~1–3 min, stage 2 ~1–3
min, greedy ~seconds. **End-to-end < 15 min** (validated by the benchmark).

---

## 7. Design decisions

- **Quantile binning (not fixed):** bins of frequency ≈ 1/b (balanced tables,
  more stable MI). `bins_auto` adapts b to n.
- **Top-C categories + "other":** bounds the tables of high-cardinality
  categoricals (documented bias: rare categories are grouped).
- **Missing as a category:** by default, missing is treated as a dedicated
  code (preserves information); `missing="drop"` removes it.
- **Subsample in stage 2:** the joint tables grow with `K²`; the subsample
  (10⁶) limits the driver memory without losing precision (the tables are
  probability estimates).
- **Cell budget:** if `K²·b²·n_y` exceeds `max_cache_cells`, K is reduced
  (dropping candidates with lower MI).
- **Round 1 = argmax MI:** avoids the CMI cost in the first round (where
  redundancy does not matter).
- **CMIM `max_min`:** routes CMIM to JMIM to avoid the distributed CMIM pass
  in stage 3 (lower bound of the CMI).
- **Optional KSG estimator:** the only route with global kNN in the driver;
  for continuous variables where binning is not acceptable (moderate n).
- **Greedy selection behind an oracle (ADR-0001):** a single loop
  (`greedy_select`) over the `InformationOracle` interface, with two
  adapters (histogram and KSG); the fast criteria are pure oracle
  arithmetic.
- **Full-population ranking:** the ranking covers all N features by
  univariate MI (screening in the histogram estimator, kNN in the KSG
  estimator); the efficiency curve scales with N.
- **Significance behind a seam:** the `SignificanceTest` interface
  (`significance.py`) with three adapters (χ², permutation, no test);
  `screen()` only composes tables → MI → significance → top-K and FDR
  control lives inside each adapter.
- **Evaluation behind a seam:** the `ModelFactory` interface
  (`model_factory.py`) with three adapters (GBT, XGBoost, LightGBM); the
  evaluation logic (`evaluate.py`) composes encoding → assembly → prediction
  → metric and accepts a backend name or a factory; the task arrives already
  resolved (3 values; the legacy vocabulary `"classification"` /
  `"regression"` is accepted as an alias) and the cost (2 + |ranking|
  trainings) is declared in the interface.
- **Cache with oriented accessors:** `TableCache.cmi_table(i, j)`
  (`tables.py`) returns the triple in the orientation the entropy functions
  expect (x, Y, z), regardless of the order of (i, j); canonicalization
  (min, max) and axis knowledge live in one place (the cache) and
  `HistogramOracle.cmi_single` stays a thin formula.
- **Mode-aware config:** `SelectorConfig` declares per-mode relevance in one
  place (histogram / KSG / shared fields); a field of the other mode that
  differs from its default raises `ValueError` at construction, and KSG mode
  requires numeric columns with a clear error in `fit` (not a deep crash in
  `to_numpy`).
- **User-declared task (ADR-0002):** `SelectorConfig.task` (`"auto"` |
  `"classifier_binary"` | `"classifier_multiclass"` | `"continuous"`, a
  shared field of both modes) is resolved once in stage 0 with the shared
  `schema.resolve_task` rule (validating against the data) and carried in
  `SelectorModel.task` until `report`, which measures the raw Target; KSG
  mode supports classification with a lossless coding of the target per
  distinct value.
- **Shared planted-structure generator:** `tests/planted.py` defines the
  planted structure once (feature roles: informative/independent/redundant/
  correlated; target modes: `and`, `linear` with thresholds, `flip`); the
  tests (selector, KSG, evaluation, screening, significance) and
  `benchmarks/synthetic.generate` consume it, so changing the planted
  structure is a one-place change.

---

## 8. Limitations

- **Binning:** the histogram estimator loses information in high-dimensional
  continuous variables; the KSG estimator mitigates this but is more
  expensive (driver).
- **Rare categories:** the top-C + "other" groups rare categories (bias).
- **Exact CMIM:** the set `S_m` is fixed (top-m by MI); it is not updated per
  round (documented approximation).
- **KSG:** only for moderate n (subsample ≤ 250k to the driver).
- **Multiclass:** the MI is computed over the target (discrete labels); there
  is no special treatment of imbalance (the FDR filter mitigates it).
