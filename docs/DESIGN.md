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
- **Significancia:** interfaz `SignificanceTest` con tres adaptadores (χ²,
  permutación distribuida, sin test) + control FDR (ver §4).
- **Candidatas C:** top-`screen_top_k` por MI entre las que pasan el filtro de
  significancia.

### Etapa 2 — Tablas conjuntas (1 pase sobre subsample)

Sobre un subsample ≤ `subsample` (10⁶) filas, un `mapInPandas` emite las
tablas conjuntas de **pares** y **triples** `(x_i, x_j, y)` para las candidatas
→ driver → `TableCache`. Esto permite calcular CMI de forma exacta (o
aproximada) en la etapa 3 sin volver a tocar los datos.

### Etapa 3 — Selección greedy (solo driver)

Bucle greedy único (`selection.greedy_select`) detrás de la costura
`InformationOracle` (`oracles.py`), operando en **posiciones de candidata**
(0..K−1). Dos adaptadores (ADR-0001):

- **`HistogramOracle`:** consultas sobre el `TableCache` (MI univariante, MI
  de par, CMI de triple). `cmi_set_all` ejecuta el pase CMIM distribuido
  (`mapInPandas` sobre el subsample) cuando el criterio es CMIM exacto.
- **`KsgOracle`:** MI/CMI por kNN (estimador KSG) sobre el subsample en el
  driver, sin binning.

- **Ronda 1:** argmax MI univariante.
- **Rondas siguientes:** puntuar cada candidata no seleccionada con el
  criterio (JMIM/CMIM/mRMR/mIM) y elegir el argmax.
- **Parada:** `best_score < min_score` o `max_features` alcanzados.
- **Ranking:** población completa (N features) por MI univariante
  (descendente): la MI de screening en el estimador de histograma (sin coste
  adicional) y la MI de kNN en el estimador KSG.

### Mapa de módulos

```
preprocess.py   esquema + preprocesado (etapa 0)
screen.py       screening univariante (etapa 1)
significance.py costura SignificanceTest + adaptadores (etapa 1)
tables.py       tablas conjuntas + TableCache (etapa 2, accesores orientados)
oracles.py      InformationOracle + adaptadores (costura, etapa 3)
selection.py    bucle greedy único (etapa 3, driver)
criteria.py     fórmulas de criterios + pase CMIM (mapInPandas)
info/ksg.py     estimador KSG (MI/CMI por kNN)
info/entropy.py MI/CMI sobre tablas discretas
selector.py     orquestación de las etapas 0-3
model.py        SelectorModel (resultado + transform/report)
model_factory.py costura ModelFactory + adaptadores (evaluación)
evaluate.py     evaluación (AUC/R² + curva de eficiencia, compone la costura)
```

```
greedy_select (selection.py)
        │  consulta
   InformationOracle (oracles.py)   ← costura
      ┌────────────────┴────────────────┐
  HistogramOracle                   KsgOracle
  (TableCache + pase CMIM)         (kNN en driver)
```

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

| Criterio | Fórmula | Consulta al oráculo |
|---|---|---|
| **JMIM** | `min_{Xi∈S} I(X;Y\|Xi)` | `cmi_single` |
| **CMIM** | `I(X;Y\|S_m)`, `S_m` = top-m por MI (m=2) | `cmi_set_all` |
| **mRMR** | `I(X;Y) − (1/\|S\|)·Σ_{Xi∈S} I(X;Xi)` | `mi_pair` |
| **mIM** | `I(X;Y) − Σ_{Xi∈S} I(X;Xi)` | `mi_pair` |
| **JMI** | `Σ_{Xi∈S} I(X;Y\|Xi)` | `cmi_single` |

Los criterios rápidos (JMIM/mRMR/mIM/JMI) son aritmética pura sobre el
oráculo (sin jobs de Spark); solo el CMIM exacto toca Spark a través de
`cmi_set_all` (pase distribuido por ronda).

**CMIM aproximado (`max_min`):** en lugar del pase exacto por ronda, CMIM se
rutea a JMIM (`min_{Xi∈S} I(X;Y\|Xi)`), que es una cota inferior de la CMI
condicionada en el conjunto. Esto evita el pase CMIM distribuido de la etapa 3.

**Conjunto efectivo de condicionamiento (CMIM):** `S_m \ {x}` (se excluye la
candidata `x` para evitar la degeneración `I(X;Y|X)=0`).

### Estimador KSG (continuas)

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

> **Trade-off:** el estimador KSG submuestrea ≤ `ksg_subsample` (250k) filas
> al driver y calcula MI/CMI por kNN allí. Es la única ruta con kNN global en
> driver; adecuada para n moderado, no para 10⁷.

---

## 4. Significancia

Costura de la etapa 1: la interfaz `SignificanceTest` (`significance.py`) —
"dadas las tablas de screening y la MI por feature (y, para la permutación,
el df preparado), devuelve los p-valores y la máscara de significancia por
feature". `screen()` solo compone: tablas → MI → significancia → top-K.

Tres adaptadores (campo `significance`):

- **`chi2`** (`Chi2Test`): test de χ² sobre la tabla de contingencia (rápido,
  asintótico): `G² = 2·n·MI ~ χ²`.
- **`permutation`** (`PermutationTest`): test de permutación distribuido (UN
  `mapInPandas` sobre un subsample ≤ `permutation_rows`; `B = n_permutations`
  permutaciones de y por partición, semilla `seed + b`; p-valor empírico).
  Solo se aplica a las top-`screen_top_k` por MI (pre-filtro); el resto queda
  con p = 1.
- **`none`** (`NoTest`): sin filtro (p = 1, todas significativas).

El control FDR de Benjamini-Hochberg (`fdr_q`, 0.05 por defecto) se aplica
dentro de cada adaptador sobre sus p-valores.

Las candidatas son las que superan el filtro de significancia **y** están en el
top-`screen_top_k` por MI.

---

## 5. Justificación frente a SOTA

| Aspecto | Enfoques previos | SparkMIM |
|---|---|---|
| **Cómputo de tablas** | O(N) jobs (un `groupBy` por feature/par) | **Pases únicos** (1 `mapInPandas` + 1 `groupBy`) |
| **CMI** | Re-muestreo o aproximación por par | Tablas conjuntas en 1 pase sobre subsample |
| **Significancia** | Solo χ² (driver) | χ² / permutación distribuida / sin test (+ control FDR) |
| **Continuas** | Binning fijo | Binning cuantil + estimador KSG opcional |
| **Evaluación** | Externa (sklearn) | Integrada, agnóstica al modelo (GBT/XGBoost/LightGBM detrás de la costura `ModelFactory`) |
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
| **3** (greedy) | `O(max_features·K²)` en driver | 0 (CMIM exacto: 1 `mapInPandas` por ronda) | `O(K²)` |

**Escala:** lineal en `n` y en `N` (etapa 1) / `K²` (etapa 2). El bucle greedy
es solo driver y no lanza jobs (salvo el pase CMIM exacto: 1 `mapInPandas`
por ronda).

**Evaluación:** `efficiency_curve` entrena un modelo por prefijo del ranking
(coste O(|ranking|) entrenamientos); con el ranking de población completa
escala con N, no con K. El entrenamiento pasa por la costura `ModelFactory`
(`model_factory.py`): 1 entrenamiento por llamada a `train`.

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
- **CMIM `max_min`:** rutea CMIM a JMIM para evitar el pase CMIM distribuido de
  la etapa 3 (cota inferior de la CMI).
- **Estimador KSG opcional:** única ruta con kNN global en driver; para
  continuas donde el binning no es aceptable (n moderado).
- **Selección greedy detrás de un oráculo (ADR-0001):** un solo bucle
  (`greedy_select`) sobre la interfaz `InformationOracle`, con dos
  adaptadores (histograma y KSG); los criterios rápidos son aritmética pura
  sobre el oráculo.
- **Ranking de población completa:** el ranking cubre las N features por MI
  univariante (screening en el estimador de histograma, kNN en el estimador
  KSG); la curva de eficiencia escala con N.
- **Significancia detrás de una costura:** la interfaz `SignificanceTest`
  (`significance.py`) con tres adaptadores (χ², permutación, sin test);
  `screen()` solo compone tablas → MI → significancia → top-K y el control
  FDR vive dentro de cada adaptador.
- **Evaluación detrás de una costura:** la interfaz `ModelFactory`
  (`model_factory.py`) con tres adaptadores (GBT, XGBoost, LightGBM); la
  lógica de evaluación (`evaluate.py`) compone detección de tarea →
  codificación → ensamblaje → predicción → métrica y acepta un nombre de
  backend o una fábrica; el coste (2 + |ranking| entrenamientos) y la regla
  de detección de tarea se declaran en la interfaz.
- **Caché con accesores orientados:** `TableCache.cmi_table(i, j)`
  (`tables.py`) devuelve la triple en la orientación que esperan las
  funciones de entropía (x, Y, z), sin importar el orden de (i, j); la
  canonicidad (min, max) y el conocimiento de ejes viven en un solo sitio
  (el caché) y `HistogramOracle.cmi_single` queda como fórmula delgada.
- **Config por modo:** `SelectorConfig` declara la relevancia por modo en un
  solo sitio (campos de histograma / de KSG / compartidos); un campo del otro
  modo distinto de su valor por defecto lanza `ValueError` en la
  construcción, y el modo KSG exige columnas numéricas con error claro en
  `fit` (no un crash profundo en `to_numpy`).
- **Generador de estructura plantada compartido:** `tests/planted.py`
  define la estructura plantada una sola vez (roles de feature:
  informativa/independiente/redundante/correlada; modos de target: `and`,
  `linear` con umbrales, `flip`); los tests (selector, KSG, evaluación,
  screening, significancia) y `benchmarks/synthetic.generate` la consumen,
  de modo que cambiar la estructura plantada es un cambio en un solo sitio.

---

## 8. Limitaciones

- **Binning:** el estimador de histograma pierde información en continuas de
  alta dimensionalidad; el estimador KSG lo mitiga pero es más caro (driver).
- **Categorías raras:** el top-C + "other" agrupa categorías raras (sesgo).
- **CMIM exacto:** el conjunto `S_m` es fijo (top-m por MI); no se actualiza por
  ronda (aproximación documentada).
- **KSG:** solo para n moderado (subsample ≤ 250k al driver).
- **Multiclase:** la MI se calcula sobre el target (etiquetas discretas); no
  hay tratamiento especial de desbalance (el binning cuantil y el FDR
  mitigan).
