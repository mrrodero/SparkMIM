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
5. **Target:** classification (discrete labels) or regression (quantile
   binning `b_y=10`).

### Stage 1 — Univariate screening (1 pass)

A single `mapInPandas` emits the crosstabs `(fid, code_x, code_y, count)` for
each feature → a `groupBy` → driver. Per feature:

- Contingency table `P(x, y)` (dense, `n_codes × n_y`).
- **Univariate MI** `I(X;Y)` (nats).
- **Significance:** χ², distributed permutation test, or BH-FDR (see §4).
- **Candidates C:** top-`screen_top_k` by MI among those passing the
  significance filter.

### Stage 2 — Joint tables (1 pass over a subsample)

Over a subsample ≤ `subsample` (10⁶) rows, a `mapInPandas` emits the joint
tables of **pairs** and **triples** `(x_i, x_j, y)` for the candidates →
driver → `TableCache`. This allows computing CMI exactly (or approximately) in
stage 3 without touching the data again.

### Stage 3 — Greedy selection (driver only)

Greedy loop over the `TableCache` (no Spark jobs):

- **Round 1:** argmax univariate MI.
- **Subsequent rounds:** score each unselected candidate with the criterion
  (JMIM/CMIM/mRMR/mIM) and pick the argmax.
- **Stopping:** `best_score < min_score` or `max_features` reached.
- **Ranking:** all candidates by univariate MI (descending).

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

| Criterion | Formula |
|---|---|
| **JMIM** | `min_{Xi∈S} I(X;Y\|Xi)` |
| **CMIM** | `I(X;Y\|S_m)`, `S_m` = top-m by MI (m=2) |
| **mRMR** | `I(X;Y) − (1/\|S\|)·Σ_{Xi∈S} I(X;Xi)` |
| **mIM** | `I(X;Y) − Σ_{Xi∈S} I(X;Xi)` |
| **JMI** | `Σ_{Xi∈S} I(X;Y\|Xi)` |

**Approximate CMIM (`max_min`):** instead of the exact per-round pass, CMIM is
routed to JMIM (`min_{Xi∈S} I(X;Y\|Xi)`), which is a lower bound on the CMI
conditioned on the set. This avoids the extra pass in stage 3.

**Effective conditioning set (CMIM):** `S_m \ {x}` (the candidate `x` is
excluded to avoid the degeneracy `I(X;Y|X)=0`).

### KSG mode (continuous)

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

> **Trade-off:** KSG mode subsamples ≤ `ksg_subsample` (250k) rows to the
> driver and computes MI/CMI by kNN there. It is the only route with global
> kNN in the driver; suitable for moderate n, not for 10⁷.

---

## 4. Significance

Three methods (`significance` field):

- **`chi2`:** χ² test on the contingency table (fast, asymptotic).
- **`permutation`:** distributed permutation test (resampling of `n`
  permutations in workers, empirical p-value).
- **`fdr`:** false discovery rate control (BH) over the p-values of χ² or
  permutation. `alpha` (0.05 by default).
- **`none`:** no filter.

The candidates are those that pass the significance filter **and** are in the
top-`screen_top_k` by MI.

---

## 5. Justification vs SOTA

| Aspect | Previous approaches | SparkMIM |
|---|---|---|
| **Table computation** | O(N) jobs (one `groupBy` per feature/pair) | **Single passes** (1 `mapInPandas` + 1 `groupBy`) |
| **CMI** | Resampling or pairwise approximation | Joint tables in 1 pass over a subsample |
| **Significance** | Only χ² (driver) | χ² / distributed permutation / BH-FDR |
| **Continuous** | Fixed binning | Quantile binning + optional KSG mode |
| **Evaluation** | External (sklearn) | Integrated, model-agnostic (GBT/XGBoost/LightGBM) |
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
| **3** (greedy) | `O(max_features·K²)` in driver | 0 | `O(K²)` |

**Scaling:** linear in `n` and in `N` (stage 1) / `K²` (stage 2). The greedy
loop is driver-only and launches no jobs.

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
- **CMIM `max_min`:** routes CMIM to JMIM to avoid the extra pass in stage 3
  (lower bound of the CMI).
- **Optional KSG:** the only route with global kNN in the driver; for
  continuous variables where binning is not acceptable (moderate n).

---

## 8. Limitations

- **Binning:** the histogram mode loses information in high-dimensional
  continuous variables; the KSG mode mitigates this but is more expensive
  (driver).
- **Rare categories:** the top-C + "other" groups rare categories (bias).
- **Exact CMIM:** the set `S_m` is fixed (top-m by MI); it is not updated per
  round (documented approximation).
- **KSG:** only for moderate n (subsample ≤ 250k to the driver).
- **Multiclass:** the MI is computed over the target (discrete labels); there
  is no special treatment of imbalance (the quantile binning and the FDR
  mitigate it).
