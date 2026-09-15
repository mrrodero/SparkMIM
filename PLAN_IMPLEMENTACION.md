# SparkMIM — Plan de implementación

**Framework PySpark distribuido de feature selection informacional (JMIM/CMIM)**

> Dado un DataFrame con N variables (categóricas o numéricas) y una variable objetivo (binaria, multiclase o regresión), seleccionar un subconjunto inteligente de features de forma distribuida y ultrarrápida, escalable a 10⁵–10⁷ filas y 10²–10⁴ características.

---

## 1. Objetivo y criterios de éxito

**Criterios de éxito:**

- **Calidad:** en datos sintéticos con estructura plantada, selecciona primero las variables verdaderamente relevantes y penaliza las redundantes (propiedad max-relevance/min-redundancy); el ruido no pasa el filtro de significancia.
- **Velocidad:** diseño de *pases únicos* (2 pasadas completas: cuantiles/cardinalidades + screening; 1 pase de mapeo sin shuffle; 1 pase sobre subsample para tablas conjuntas; bucle greedy solo en driver). Meta: n=10⁶, N=200 → < 15 min end-to-end en `local[8]` (validado por benchmark, no asunción).
- **Correctitud:** tests unitarios con valores analíticos de entropía/MI/CMI; determinismo con semilla fija.
- **API:** usable en ~5 líneas (ver §5).

**Ubicación del proyecto:** `C:\Users\mrrodero\Desktop\Projects\sparkmim\`

---

## 2. Estado del arte (base del diseño)

- **Criterios informacionales greedy:**
  - mRMR — Peng, Long & Ding (2005), IEEE TPAMI.
  - mIM — Huang et al. (2010), *A novel feature selection criterion for multi-class problems*, ESWA.
  - **JMIM** — Bennasar, Hicks & Setchi (2015), ESWA 42(22): estrategia *max-min*, `argmax_X min_{Xi∈S} CMI(X;Y|Xi)` (criterio verificado contra la [implementación de referencia `mifs`](https://github.com/danielhomola/mifs); también [IJCA 2016](https://research.ijcaonline.org/etc2016/number4/etc2016274.pdf)).
  - **CMIM** — Brown, Manolopoulos, Marshall & Buhman (2005), IEEE TPAMI: multiclase, condiciona sobre el conjunto S completo.
  - [CMIM-2](https://www.researchgate.net/publication/301552944_CMIM-2_AN_ENHANCED_CONDITIONAL_MUTUAL_INFORMATION_MAXIMIZATION_CRITERION_FOR_FEATURE_SELECTION) (2016) y CMI de alto orden ([HOCMIM, 2022](https://www.sciencedirect.com/science/article/pii/S0031320322003764)).
- **IT-FS distribuido previo:**
  - Framework Spark/MLlib de la Univ. de Granada: mRMR, CMI, JMI — [Ramírez-Gallego et al., IEEE TSMC: Systems (2017)](https://sci2s.ugr.es/node/317). Precedente más cercano (Scala, Spark 2.x, jobs por característica).
  - **fast-mRMR** en CPU/GPU/Spark — [Mouriño-Talín et al. (2019)](https://onlinelibrary.wiley.com/doi/abs/10.1002/int.21833).
  - [Feature selection paralelo en memoria distribuida](http://gac.udc.es/~juan/papers/informationsciences2019.pdf) (2019).
- **Estimación de MI con continuas:** estimador kNN **KSG** — [Kraskov, Stögbauer & Grassberger (2004)](https://infomeasure.readthedocs.io/en/0.5.0/guide/mutual_information/kraskov_stoegbauer_grassberger/).
- **Significancia:** [permutation test para MI](https://www.researchgate.net/publication/221166361_The_permutation_test_for_feature_selection_by_mutual_information) (McKean et al.) + aproximación χ² de la distribución nula de MI (estadístico G² = 2n·MI ~ χ²).

**Hueco que rellena SparkMIM:** PySpark nativo (sin Scala), Spark 3.5, JMIM/CMIM con condicionamiento acotado, **cálculo de tablas en pases únicos** (evita O(N) jobs — el cuello de botella del trabajo previo), control de significancia FDR, soporte de regresión/multiclase, modo KSG opcional, y evaluación integrada agnóstica al modelo (GBT/XGBoost/LightGBM en Spark, **sin sklearn**).

---

## 3. Fundamento matemático

Variables discretizadas X, Y (ver §6):

- `H(X) = −Σ p(x) log p(x)`
- `MI(X;Y) = H(X) + H(Y) − H(X,Y)`
- `CMI(X;Y|Z) = Σ p(x,y,z) · log [ p(x,y,z)·p(z) / (p(x,z)·p(y,z)) ]`

Criterios greedy (S = conjunto seleccionado, X = candidata), como interfaz `Criterion` intercambiable:

| Método | Puntaje | Tablas necesarias |
|---|---|---|
| mRMR | `MI(X;Y) − (1/\|S\|)·Σ_{Xi∈S} MI(X;Xi)` | pares (X,Xi) |
| mIM | `MI(X;Y) − Σ_{Xi∈S} MI(X;Xi)` | pares (X,Xi) |
| JMI | `Σ_{Xi∈S} CMI(X;Y\|Xi)` | triples (X,Y,Xi) |
| **JMIM** (default) | `min_{Xi∈S} CMI(X;Y\|Xi)` | triples (X,Y,Xi) |
| **CMIM** | `CMI(X;Y\|S_m)`, S_m = top-m seleccionadas por MI univariante (m=2) | conjunta (X,Y,S_m) por ronda |

Identidad usada para CMI con KSG: `CMI(X;Y|Z) = MI(XZ;Y) − MI(Z;Y)`.

---

## 4. Arquitectura del paquete

```
sparkmim/                    # repo git local (git init; sin remote)
├── .gitignore               # venv, __pycache__, .pytest_cache, derby/, spark-logs/
├── pyproject.toml           # deps: pyspark>=3.5, numpy, pandas, scipy; extras: xgb, lgbm; requires-python >=3.11
├── README.md                # ES — quickstart, API, config, rendimiento, referencias
├── README.en.md             # EN — misma estructura
├── README.zh.md             # ZH — opcional
├── DESIGN.md                # ES — diseño completo: matemáticas, SOTA, complejidad
├── docs/DESIGN.en.md        # EN — diseño completo
├── docs/DESIGN.zh.md        # ZH — diseño completo (opcional)
├── PLAN_IMPLEMENTACION.md   # este documento
├── src/sparkmim/
│   ├── __init__.py           # exports públicos
│   ├── config.py             # SelectorConfig (dataclass con todos los hiperparámetros)
│   ├── schema.py             # detección numérico/categórico, specs de columnas
│   ├── preprocess.py         # binning cuantil, codificación, missing, target
│   ├── tables.py             # builders de tablas en pase único (mapInPandas) + TableCache
│   ├── info/entropy.py       # H, MI, CMI desde tablas (numpy vectorizado)
│   ├── info/ksg.py           # estimador KSG opcional (subsample, cKDTree)
│   ├── significance.py       # χ² nula, permutation test, BH-FDR
│   ├── criteria.py           # mRMR, mIM, JMI, JMIM, CMIM
│   ├── selector.py           # InfoSelector + JMIMSelector, CMIMSelector, ...
│   ├── model.py              # SelectorModel (selected_features, scores_, ranking_, transform)
│   └── evaluate.py           # report() agnóstico al modelo: GBT spark.ml (default), xgboost4j/lightgbm4j (extras); sin sklearn
├── tests/                    # unitarios + e2e (ver §9)
└── benchmarks/               # generador sintético + bench de escala
```

---

## 5. API pública

```python
from sparkmim import JMIMSelector, CMIMSelector, MRMRSelector, MIMSelector

sel = JMIMSelector(
    target="y",
    max_features=50,          # nº de features a seleccionar
    screen_top_k=100,         # candidatas tras screening (etapa 1)
    bins=10,                  # bins para continuas (o "auto")
    max_categories=50,        # tope de cardinalidad categóricas (top-C + "other")
    missing="category",       # "category" | "drop"
    significance="chi2",      # "chi2" | "permutation" | None
    fdr_q=0.05,
    min_score=1e-4,           # umbral de parada (nats)
    subsample=1_000_000,      # filas para tablas conjuntas (etapa 2)
    estimator="histogram",    # "histogram" | "ksg"
    seed=42,
)
model = sel.fit(df)
model.selected_features       # list[str]
model.scores_                 # puntaje por ronda
model.ranking_                # ranking completo
model.transform(df)           # df solo con features seleccionadas (+ target)
model.report(df, model="gbt")     # evaluación AUC/R² vs baseline (gbt | xgboost | lightgbm)
```

---

## 6. Algoritmo (pases sobre los datos)

### Etapa 0 — Esquema y preprocesado (2 pasadas + 1 mapeo sin shuffle)

1. Detección numérico vs categórico por dtype (override manual opcional).
2. **Pase 0a (UN `mapInPandas` sobre el df original):** por partición y por feature, cuantiles `part[fid].quantile([i/b…])` (numéricas) y `value_counts()` (categóricas) → emite `(fid, i, q, n_part)` y `(fid, val, count)` → un único `groupBy` → driver. **Evita N llamadas a `approxQuantile` y N `groupBy` (O(N) jobs).**
3. **Driver:** cuantiles globales ponderados por `n_part` → fronteras de bins (`b=10`, o `auto`: `clamp(round(log2 n), 4, 20)`); top-C=50 categorías por frecuencia + código "other" (mantiene tablas acotadas; sesgo documentado).
4. **Pase 0b (mapeo, sin shuffle):** expresiones `when`/`map` por columna aplicando bins y códigos; missing → código dedicado (o drop).
5. Target:
   - Binaria → tal cual.
   - Multiclase → códigos.
   - **Regresión → bins cuantiles del target** (`b_y=10`) en modo histograma (modo KSG conserva Y continua).
6. Se materializa `df_prep` (códigos enteros) con `persist(MEMORY_AND_DISK)`.

### Etapa 1 — Screening univariante (UN solo pase sobre `df_prep`)

- `mapInPandas`: por partición y por feature, `pd.crosstab(X, Y)` → emite `(fid, x, y, count)`.
- Un único `groupBy(fid, x, y).sum()` → N tablas pequeñas en el driver (sparse → numpy denso).
- En driver: `MI(X;Y)` por feature + p-valor (χ² por defecto, O(1); o **permutation test distribuido**: UN pase `mapInPandas` sobre subsample ≤5·10⁴ filas que recalcula MI de las top-K candidatas bajo B=100 permutaciones de y (permutaciones numpy por partición, semilla fija) → p-valores + BH-FDR q=0.05).
- Se conservan las top-`screen_top_k` por MI entre las que pasan significancia → conjunto C de candidatas (|C| ≤ K).

### Etapa 2 — Caché de tablas conjuntas (UN pase sobre subsample ≤10⁶ filas)

- `mapInPandas`: por partición, para cada par (i,j) de candidatas:
  - `crosstab(Xi, Xj)` → tabla par.
  - `crosstab([Xi, Xj], Y)` → tabla triple.
- Emite tablas ya agregadas → un único `groupBy` → `TableCache` en driver como **arrays numpy densos** (b×b y b×b×b float64).
- Coste: C(K,2) pares + C(K,2) triples.
- Memoria driver: `C(K,2)·b³·8 B` (triples) + `C(K,2)·b²·8 B` (pares) → K=100, b=10 ≈ 44 MB.
- Guard `max_cache_cells` (default 2.5·10⁸ celdas ≈ 2 GB) que reduce K o b automáticamente si se excede el presupuesto.

### Etapa 3 — Selección greedy (SOLO en driver)

- Ronda 1: `X₁ = argmax MI univariante`.
- Ronda r: `X* = argmax_{X∉S} criterion(X, S, cache)`; parada si `score < min_score` o `|S| = max_features`.
- JMIM/JMI/mRMR/mIM: solo aritmética sobre el caché (milisegundos).
- **CMIM:** por ronda, 1 pase `mapInPandas` sobre el subsample construyendo la conjunta `(X, Y, S_m)` para todas las candidatas (S_m = top-m=2 seleccionadas por MI); coste K rondas × K crosstabs sobre el subsample (K=50 → ~4–8 min). Modo `cmim_approx="max_min"` (= JMIM) como alternativa ultrarrápida.
- Salida: features seleccionadas (nombres originales), scores por ronda, ranking completo, tablas (opcional).

### Modo KSG (opcional, `estimator="ksg"`)

- Subsample ≤250k filas al driver.
- MI por KSG (k=10, `cKDTree` de scipy).
- CMI por la identidad `MI(XZ;Y) − MI(Z;Y)` con |Z|≤2.
- Para continuas donde el binning no es aceptable; más lento, para n moderado. Es el único modo con cómputo pesado en driver (kNN global); documentado como trade-off.

### Reparto driver/workers (justificación)

- **En workers (todo el cómputo pesado):** cuantiles y cardinalidades (pase 0a), crosstabs de screening (pase 1), crosstabs de pares/triples (pase 2), tablas conjuntas de CMIM por ronda, permutation test, y los modelos de evaluación (spark.ml GBT / xgboost4j / lightgbm4j).
- **En driver (solo lo imprescindible):** tablas agregadas (≤ N·b·c + C(K,2)·b³ filas), MI/CMI vectorizados con numpy, y el bucle greedy O(K²) — microsegundos a milisegundos para K≤500; distribuirlo solo añadiría overhead de shuffle sin beneficio.
- **Regla de diseño:** nada de tamaño O(n) ni O(N·n) cruza al driver; todo lo que llega es agregado y acotado por `max_cache_cells`.

---

## 7. Complejidad y rendimiento (análisis)

- **Pasadas de datos:** 2 completas (cuantiles/cardinalidades + screening) + 1 mapeo sin shuffle + 1 (tablas, subsample) + K (solo CMIM, sobre subsample). **No hay O(N) jobs.**
- **Shuffles:** pequeños (≤ N·b·c filas en etapa 1; ≤ K²·b³ filas en etapa 2).
- **Driver:** greedy O(K²) aritmética; caché acotada por `max_cache_cells`.
- **Estimación n=10⁶, N=200, K=100, local[8]:** etapa 1 ~1–3 min, etapa 2 ~1–3 min, greedy ~segundos.
- Escala lineal en n y en N (etapa 1) / K² (etapa 2). El benchmark (§9) valida la meta.

---

## 8. Casos límite y manejo de fallos

- **Missing:** código dedicado por defecto (no sesga hacia drop); opción `drop`.
- **Cardinalidad alta:** tope top-C + "other" (documentado); `max_categories` configurable.
- **CMIM exacto exponencial en |S|:** acotado por diseño (m≤2–3); documentado como aproximación de CMIM completo.
- **Binning subestima MI** (conservador): `bins` configurable y modo KSG.
- **Memoria driver:** guard `max_cache_cells` con auto-reducción de K/b; error claro si el presupuesto es inalcanzable.
- **Clases desbalanceadas:** binning cuantil del target y FDR mitigan; evaluación con AUC (no solo accuracy).
- **Spark session:** se acepta `spark` por parámetro; si no se pasa, `getOrCreate()`.
- **Determinismo:** semilla fija para subsample y permutation test → misma selección.

---

## 9. Plan de tests

### Unitarios

- `test_entropy`: MI(X;X)=H(X) (uniforme y no uniforme); MI=0 para independientes; tabla 2×2 a mano; CMI=0 cuando Z determina X e Y; CMI=MI cuando Z es independiente.
- `test_tables`: el builder en pase único coincide con `groupBy` directo (df pequeño); conversión sparse→denso correcta.
- `test_preprocess`: frecuencias de bins ≈ 1/b; missing → código dedicado; tope de cardinalidad; target de regresión binneado.
- `test_significance`: χ² en tabla conocida; permutation: independientes → no significativas, dependientes → p<0.05.
- `test_evaluate`: `report()` con GBT spark.ml en df sintético — AUC(seleccionadas) ≥ AUC(todas) − ε; curva de eficiencia no decreciente.

### Integración e2e (estructura plantada)

- Datos: `Y = f(X1, X2) + ruido`, `X3 = X1` (redundante), `X4..X8` ruido independiente.
- Assertions:
  - X1, X2 se seleccionan primero.
  - X3 se rankea por debajo de una feature independiente con MI igual (penalización de redundancia).
  - El ruido no pasa el filtro de significancia.
- Tipos: binaria, multiclase (3–5 clases), regresión.
- `transform` devuelve solo columnas seleccionadas; `max_features` respetado; parada temprana por `min_score`; determinismo con semilla.

### Benchmarks

- `benchmarks/synthetic.py`: generador con estructura plantada.
- `benchmarks/bench_scale.py`: rejilla n ∈ {10⁵, 10⁶, 10⁷}, N ∈ {100, 200, 500} en `local[8]`; mide wall-time por etapa y memoria del driver; salida CSV.
- Meta: n=10⁶, N=200 < 15 min end-to-end.

---

## 10. Empaquetado y documentación

- **Repo git local** (sin remote): `git init` al inicio, `.gitignore` (venv, `__pycache__`, `.pytest_cache`, `derby/`, `spark-logs/`), un commit por hito.
- `pyproject.toml`: nombre `sparkmim` v0.1.0; `requires-python = ">=3.11"`; deps `pyspark>=3.5, numpy>=1.24, pandas>=2.0, scipy>=1.10`; extras opcionales `xgb` (xgboost4j-spark) y `lgbm` (lightgbm4j-spark) para evaluación; `dev` (pytest). **Sin sklearn** (ni en runtime ni como extra).
- **Documentación bilingüe (ES/EN, opcionalmente ZH):**
  - `README.md` (ES) + `README.en.md` (+ `README.zh.md` opcional): quickstart, referencia de API, tabla de configuración, notas de rendimiento, referencias.
  - `DESIGN.md` (ES) + `docs/DESIGN.en.md` (+ `docs/DESIGN.zh.md` opcional): matemáticas completas, justificación de diseño vs SOTA, análisis de complejidad.

---

## 11. Hitos de implementación (orden de trabajo)

0. **Repo git local:** `git init`, `.gitignore`, `pyproject.toml` inicial, commit base.
1. **Núcleo:** `info/entropy.py` + `tables.py` + tests unitarios.
2. **Preprocesado:** `schema.py` + `preprocess.py` (pase 0a cuantiles/cardinalidades, mapeo 0b, missing, target) + tests.
3. **Etapa 1:** screening en pase único + `significance.py` (χ² + permutation distribuido) + tests.
4. **Etapa 2 + selección:** `criteria.py` + `selector.py`/`model.py` (JMIM, JMI, mRMR, mIM, CMIM) + tests e2e.
5. **Modo KSG** opcional.
6. **Evaluación:** `evaluate.py` — report agnóstico al modelo: GBT spark.ml (default) + extras xgboost/lightgbm; AUC multiclase en driver; curva de eficiencia; sin sklearn.
7. **Benchmarks + docs bilingües:** benchmarks, `README.md`/`README.en.md` (+ ZH opcional), `DESIGN.md`/`docs/DESIGN.en.md` (+ ZH opcional).

---

## 12. Supuestos y riesgos

- **Supuestos:**
  - PySpark con pandas disponible en executors (estándar).
  - Spark 3.5; Python 3.11+.
  - El driver puede alojar la caché de tablas (presupuesto configurable).
- **Riesgos principales:**
  - (a) Coste de etapa 2 con K grande → mitigado con `screen_top_k` y guard de celdas.
  - (b) Sesgo por binning/cardinalidad → configurable y documentado.
  - (c) CMIM exacto inabordable → acotado por diseño y documentado como aproximación.
- **Fuera de alcance (futuro):** estimación de MI con sketching sublineal, GPU, streaming, feature selection profunda, HOCMIM de orden >2.
