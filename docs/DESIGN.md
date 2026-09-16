# Diseño de SparkMIM

Diseño completo: matemáticas, justificación frente a SOTA y análisis de
complejidad.

> **English:** see [DESIGN.en.md](DESIGN.en.md).
> **Quickstart/API:** [README.md](../README.md).

---

## 1. Problema

Seleccionar un subconjunto de features informativo y no redundante a escala
distribuida (n hasta 10⁷, N hasta 500) en PySpark, sin Scala y sin sklearn.

Los enfoques previos (p. ej. MI por `groupBy` por par) lanzan **O(N) jobs**
(un job por feature o por par), que se convierte en el cuello de botella en
driver (scheduling, serialización). SparkMIM elimina ese cuello calculando
tablas de contingencia en **pases únicos**.

---

## 2. Algoritmo (pases sobre los datos)

### Etapa 0 — Esquema y preprocesado (2 pasadas + 1 mapeo sin shuffle)

1. **Detección numérico vs categórico** por dtype (override manual opcional).
2. **Pase 0a (UN `mapInPandas` sobre el df original):** por partición y por
   feature, cuantiles `part[fid].quantile([i/b…])` (numéricas) y
   `value_counts()` (categóricas) → emite `(fid, i, q, n_part)` y
   `(fid, val, count)` → un único `groupBy` → driver. **Evita N llamadas a
   `approxQuantile` y N `groupBy` (O(N) jobs).**
3. **Driver:** cuantiles globales ponderados por `n_part` → fronteras de bins
   (`b=10`, o `auto`: `clamp(round(log2 n), 4, 20)`); top-C=50 categorías por
   frecuencia + código "other" (tablas acotadas; sesgo documentado).
4. **Pase 0b (mapeo, sin shuffle):** expresiones `when`/`map` por columna
   aplicando bins y códigos; missing → código dedicado (o drop).
5. **Target:** clasificación (etiquetas discretas) o regresión (binning
   cuantil `b_y=10`).

### Etapa 1 — Screening univariante (1 pase)

Un único `mapInPandas` emite los crosstabs `(fid, code_x, code_y, count)` para
cada feature → un `groupBy` → driver. Por feature:

- Tabla de contingencia `P(x, y)` (densa, `n_codes × n_y`).
- **MI univariante** `I(X;Y)` (nats).
- **Significancia:** χ², test de permutación distribuido o BH-FDR (ver §4).
- **Candidatas C:** top-`screen_top_k` por MI entre las que pasan el filtro de
  significancia.

### Etapa 2 — Tablas conjuntas (1 pase sobre subsample)

Sobre un subsample ≤ `subsample` (10⁶) filas, un `mapInPandas` emite las
tablas conjuntas de **pares** y **triples** `(x_i, x_j, y)` para las candidatas
→ driver → `TableCache`. Esto permite calcular CMI de forma exacta (o
aproximada) en la etapa 3 sin volver a tocar los datos.

### Etapa 3 — Selección greedy (solo driver)

Bucle greedy sobre el `TableCache` (sin jobs de Spark):

- **Ronda 1:** argmax MI univariante.
- **Rondas siguientes:** puntuar cada candidata no seleccionada con el
  criterio (JMIM/CMIM/mRMR/mIM) y elegir el argmax.
- **Parada:** `best_score < min_score` o `max_features` alcanzados.
- **Ranking:** todas las candidatas por MI univariante (descendente).

---

## 3. Matemáticas

### Entropía y MI (tablas discretas)

Dada la tabla de contingencia `P(x, y)`:

```
H(X)   = −Σ_x P(x) log P(x)
H(Y|X) = −Σ_{x,y} P(x,y) log (P(y|x))
I(X;Y) = H(Y) − H(Y|X) = Σ_{x,y} P(x,y) log (P(x,y) / (P(x) P(y)))
```

CMI (condicionada en Z):

```
I(X;Y|Z) = Σ_{x,y,z} P(x,y,z) log (P(x,y,z) P(z) / (P(x,z) P(y,z)))
```

Implementación: `info/entropy.py` (con `np.where` para evitar `0·log 0 = NaN`).

### Criterios greedy

| Criterio | Fórmula |
|---|---|
| **JMIM** | `min_{Xi∈S} I(X;Y\|Xi)` |
| **CMIM** | `I(X;Y\|S_m)`, `S_m` = top-m por MI (m=2) |
| **mRMR** | `I(X;Y) − (1/\|S\|)·Σ_{Xi∈S} I(X;Xi)` |
| **mIM** | `I(X;Y) − Σ_{Xi∈S} I(X;Xi)` |
| **JMI** | `Σ_{Xi∈S} I(X;Y\|Xi)` |

**CMIM aproximado (`max_min`):** en lugar del pase exacto por ronda, CMIM se
rutea a JMIM (`min_{Xi∈S} I(X;Y\|Xi)`), que es una cota inferior de la CMI
condicionada en el conjunto. Esto evita el pase extra de la etapa 3.

**Conjunto efectivo de condicionamiento (CMIM):** `S_m \ {x}` (se excluye la
candidata `x` para evitar la degeneración `I(X;Y|X)=0`).

### Modo KSG (continuas)

Para variables continuas, el estimador de Kraskov-Stögbauer-Grassberger:

```
I(X;Y) = (1/n) · Σ_i [ ψ(k) + ln(n) − ψ(n_x) − ψ(n_y) ]
```

donde `ψ` es la función digamma, `n_x`/`n_y` son los recuentos de vecinos
**incluyendo el propio punto** dentro de la esfera de radio `ε_i` (distancia al
k-ésimo vecino en el espacio conjunto), y `k` es el número de vecinos
(10 por defecto). Implementación: `info/ksg.py` (con `scipy.spatial.cKDTree`).

**CMI por identidad:** `I(X;Y|Z) = I(XZ;Y) − I(Z;Y)` (con `|Z| ≤ 2` por plan).
La CMI **no** se recorta a 0 (negativa = sin información); la MI sí se recorta
a 0.

> **Trade-off:** el modo KSG submuestrea ≤ `ksg_subsample` (250k) filas al
> driver y calcula MI/CMI por kNN allí. Es la única ruta con kNN global en
> driver; adecuada para n moderado, no para 10⁷.

---

## 4. Significancia

Tres métodos (campo `significance`):

- **`chi2`:** test de χ² sobre la tabla de contingencia (rápido, asintótico).
- **`permutation`:** test de permutación distribuido (re-muestreo de `n`
  permutaciones en workers, p-valor empírico).
- **`fdr`:** control de la tasa de error falso (BH) sobre los p-valores de
  χ² o permutación. `alpha` (0.05 por defecto).
- **`none`:** sin filtro.

Las candidatas son las que superan el filtro de significancia **y** están en el
top-`screen_top_k` por MI.

---

## 5. Justificación frente a SOTA

| Aspecto | Enfoques previos | SparkMIM |
|---|---|---|
| **Cómputo de tablas** | O(N) jobs (un `groupBy` por feature/par) | **Pases únicos** (1 `mapInPandas` + 1 `groupBy`) |
| **CMI** | Re-muestreo o aproximación por par | Tablas conjuntas en 1 pase sobre subsample |
| **Significancia** | Solo χ² (driver) | χ² / permutación distribuida / BH-FDR |
| **Continuas** | Binning fijo | Binning cuantil + modo KSG opcional |
| **Evaluación** | Externa (sklearn) | Integrada, agnóstica al modelo (GBT/XGBoost/LightGBM) |
| **Dependencias** | sklearn, scipy (driver) | Solo pyspark, numpy, pandas, scipy (driver) |

**Cuello de botella eliminado:** el scheduling de O(N) jobs en driver. Con
pases únicos, el número de jobs es constante (independiente de N), y el coste
dominante es el cómputo en workers (lineal en n y N).

---

## 6. Análisis de complejidad

Sea `n` = filas, `N` = features, `K` = candidatas, `b` = bins, `C` = categorías.

| Etapa | Tiempo | Jobs | Memoria (driver) |
|---|---|---|---|
| **0a** (cuantiles/card.) | `O(n·N)` en workers | 1 `mapInPandas` + 1 `groupBy` | `O(N·(b+C))` |
| **0b** (mapeo) | `O(n·N)` en workers | 1 mapeo (sin shuffle) | — |
| **1** (screening) | `O(n·N)` en workers | 1 `mapInPandas` + 1 `groupBy` | `O(N·b·n_y)` |
| **2** (tablas conjuntas) | `O(n_sub·K²)` en workers | 1 `mapInPandas` + 1 `groupBy` | `O(K²·b²·n_y)` |
| **3** (greedy) | `O(max_features·K²)` en driver | 0 | `O(K²)` |

**Escala:** lineal en `n` y en `N` (etapa 1) / `K²` (etapa 2). El bucle greedy
es solo driver y no lanza jobs.

**Meta:** n=10⁶, N=200, K=100, `local[8]` → etapa 1 ~1–3 min, etapa 2 ~1–3 min,
greedy ~segundos. **End-to-end < 15 min** (validada por el benchmark).

---

## 7. Decisiones de diseño

- **Binning cuantil (no fijo):** bins de frecuencia ≈ 1/b (tablas equilibradas,
  MI más estable). `bins_auto` adapta b a n.
- **Top-C categorías + "other":** acota las tablas de categóricas de alta
  cardinalidad (sesgo documentado: las categorías raras se agrupan).
- **Missing como categoría:** por defecto, el missing se trata como un código
  dedicado (preserva información); `missing="drop"` lo elimina.
- **Subsample en etapa 2:** las tablas conjuntas crecen con `K²`; el subsample
  (10⁶) limita la memoria del driver sin perder precisión (las tablas son
  estimaciones de probabilidad).
- **Presupuesto de celdas:** si `K²·b²·n_y` excede `max_cache_cells`, se reducen
  K (descartando candidatas de menor MI).
- **Ronda 1 = argmax MI:** evita el coste de CMI en la primera ronda (donde la
  redundancia no importa).
- **CMIM `max_min`:** rutea CMIM a JMIM para evitar el pase extra de la etapa
  3 (cota inferior de la CMI).
- **KSG opcional:** única ruta con kNN global en driver; para continuas donde el
  binning no es aceptable (n moderado).

---

## 8. Limitaciones

- **Binning:** el modo histogram pierde información en continuas de alta
  dimensionalidad; el modo KSG lo mitiga pero es más caro (driver).
- **Categorías raras:** el top-C + "other" agrupa categorías raras (sesgo).
- **CMIM exacto:** el conjunto `S_m` es fijo (top-m por MI); no se actualiza por
  ronda (aproximación documentada).
- **KSG:** solo para n moderado (subsample ≤ 250k al driver).
- **Multiclase:** la MI se calcula sobre el target (etiquetas discretas); no
  hay tratamiento especial de desbalance (el binning cuantil y el FDR
  mitigan).
