# Análisis de rendimiento — Databricks 15.4 LTS

Diagnóstico de los resultados del benchmark de escala en DBR 15.4 (Spark
3.5.0; rejilla n × N de la sección "Resultados de la rejilla" del
[handoff](handoff-databricks-15.4.md)), y de las optimizaciones aplicadas en
la rama `feature/staging-parquet-dbr154`.

## 1. Qué dicen los resultados

`etapa0` (preprocesado: estadísticas + mapeo a códigos) es **>99 % del
tiempo** en todos los puntos (etapa1/2/3 suman segundos; el driver memory
crece con n, no con N). Dos patrones:

- **En n, a N=25**: lineal (210 s a 1M → 2 252 s a 10M).
- **En N, a n fijo**: superlineal, con exponente creciente. A n=1M:
  e0(25)=210 s → e0(200)=53 566 s (14,9 h) → factor 255 para N×8, es decir
  ≈ N^2,7. A n=10k: e0(25)=37,8 s → e0(500)=9 059 s (2,5 h) → ≈ N^1,8 y
  subiendo.

Dato clave: **C1 (4 workers) ≈ C3 (16 workers) a igual n** — las filas de
la rejilla son casi idénticas. Con la implementación anterior, añadir
workers no cambiaba nada: el trabajo no se distribuía.

## 2. Causa raíz

La implementación anterior construía el DataFrame con
`createDataFrame(lista de n tuplas)` → en Spark eso es un **LocalRelation**:
los datos viven en un nodo (el del driver) y el trabajo se ejecuta con
**1 partición / 1 core**, sin importar el tamaño del cluster. Sobre esa
base, tres costes se encadenan en `etapa0` (`prepare`):

1. **Pase 0a** (UN `mapInPandas` sobre todo el df): por *batch* (20k filas)
   y por *columna*, `dropna` + `min` + `max` + `pandas.quantile` (una
   ordenación de 20k), emitiendo **N×(b+1) filas Python** (b = nº de bins,
   ≤20) que pasan por un DataFrame object + `astype` + codificación Arrow
   + retorno al driver. Todo en 1 core (LocalRelation).
2. **Pase 0b** (mapeo a códigos): cadena de `when` de b-1 ramas por
   columna, en un `select` de N columnas anchas; coste crece con N
   (medido localmente: 1,8 s a N=100 / 5,6 s a N=200, n=100k, 8 cores).
3. **`generate()`** conserva los arrays completos en el driver (~2,08 GB a
   10M×25) + buffer de staging: presión de memoria del driver (el caso
   N=1000 murió por "Python Kernel unresponsive").

El superlineal en N es coherente con 1 core ejecutando trabajo Python + JVM
sobre un df cada vez más ancho, con degradación por memoria a partir de
N=200 (2,86 s/feature a N=200 vs 0,63 a N=25; 18 s/feature a N=500).

## 3. Optimizaciones aplicadas (rama actual)

### 3.1 Staging en parquet por lotes (la palanca mayor)

`run_point` ya no expande las filas a una lista Python: la entrada se
genera **chunk a chunk** (`generate_chunks`; RNG por columna invariante al
tamaño de chunk, test incluido) y cada chunk se convierte en un DataFrame
pandas pequeño (schema explícito, features `DoubleType`, target `LongType`)
que viaja como `LocalRelation`. Dos palancas:

- **Un solo job de escritura** (`single_write=True`, por defecto): los
  chunks se unen en árbol balanceado y se escriben con un único
  `write.overwrite.parquet`. El coste medido en DBR 15.4 escalaba con el
  *número* de jobs de escritura (~25–80 s por job de commit en el volumen,
  frente a ~70 ms de trabajo real: 20 104 s de staging en 10M×200 con el
  modo por lotes, 12 037 s en 1M×1000), así que un solo job lo reduce a
  una sola vez. `single_write=False` conserva el modo por lotes (un job
  por chunk) para drivers con poca RAM.
- **Presupuesto de bytes maximizado** (`batch_bytes` 16 → 128 MiB por
  defecto): cada chunk pesa ≤~140 MiB con overhead Arrow (~1,9× de margen
  frente al límite RPC de 256 MiB), por lo que a N=25 son ~640k filas por
  chunk en vez de ~80k (menos RPCs y menos partes parquet).

Consecuencias:

- El input es un **DataSource distribuido** (las partes del único job de
  escritura): los pases 0a/0b se ejecutan en **todos los workers**, no en
  1 core.
- El payload de cada `createDataFrame` pesa ≤~140 MiB → **resuelve el
  fallo de `spark.rpc.message.maxSize`** (256 MiB; 582,5 MB medidos a
  10M×25) sin tocar ninguna config de Spark.
- El directorio de staging es único por ejecución (un fallo no contamina
  el reintento).

Es la mejora estructural: los números de e0 anteriores no son extrapolables
a la nueva implementación porque la forma en que se distribuye el trabajo
ha cambiado.

### 3.2 Pase 0a compacto y vectorizado (nuevo)

Antes: N×(b+1) filas Python por batch (una por cuantil) + cálculo pandas
por columna. Ahora:

- **Una fila por (columna, batch)**: `(fid, min_v, max_v, quantiles[])` —
  el output del UDF baja un factor **~b+1 ≈ 14×** (menos codificación
  Arrow, menos transferencia, menos `collect`). Las categóricas emiten la
  misma fila por valor (inalterado).
- **Cálculo vectorizado**: min/max con `np.min/max(axis=0)` y cuantiles con
  `np.partition` 2D agrupado por b (features vs target). Misma semántica
  que `pandas.quantile` (interpolación lineal en `q·(n-1)`): **verificado
  bit a bit** contra la implementación anterior (min/max y cuantiles,
  columnas constantes, batches con NaN → fallback pandas con `dropna`).
- **Min/max global correcto**: la versión anterior dejaba el min/max del
  *último* batch (sobrescrita en el driver); ahora se toma mín/máx sobre
  todos los batches.

Medición local (n=50k, `local[8]`, 16 cores; profile de `etapa0`,
mismo script antes y después):

| N   | stats+collect antes (s) | stats+collect nuevo (s) | filas emitidas (total del pase) |
|-----|-------------------------|-------------------------|----------------------------------|
| 25  | 4,4                     | 4,0                     | 216                              |
| 50  | 3,9                     | 3,6                     | 572                              |
| 100 | 4,3                     | 4,0                     | 816                              |
| 200 | 4,4                     | 4,1                     | 1 616                            |
| 500 | —                       | 4,5                     | 4 016                            |

Antes, el mismo pase emitía **(N+1)×(b+1) filas por batch** (b+1 ≈ 14
con b=13 bins), i.e. un factor ~14× más que ahora (una fila por columna
por batch). Localmente el paso está limitado por el *startup* del job
(~4 s fijos), por lo que la ganancia local es modesta; en DBR el output
del UDF cruza la red y se agrega en el driver, donde el factor 14× sí se
nota. El mapeo (cadenas `when`) se reescribió en §3.3.

### 3.3 `apply_mapping` como un único `Project` (plan O(N) en vez de O(N²))

El pase 0b se emitía como **N `withColumn` encadenados**: cada uno apila un
`Project` completo de N+1 campos sobre el anterior. Medición local
(n=1000, `local[8]`; `scratch/diag_e0_decomp.py`, OLD = cadena de
`withColumn`, NEW = un único `select`):

| N    | construcción del plan OLD (s) | NEW (s) | e0 OLD (s) |
|------|-------------------------------|---------|------------|
| 25   | 0,6                           | 0,5     | 16,8       |
| 200  | 85,2                          | 2,8     | 125,7      |
| 1000 | 642,2                         | 6,0     | 905,7      |

- **Construcción del plan O(N²)**: 642 s *solo para construir el plan* a
  N=1000 (análisis/optimización/serialización del árbol de `Project`s;
  en Spark Connect el plan viaja por RPC en cada acción). Nuevo: 6 s, con
  crecimiento lineal.
- **Codegen**: con los `Project` apilados la proyección entera **no**
  entraba en whole-stage (fallback a intérprete por método, visible en el
  log como `CodegenFallback: ... too large` en la cadena de `doForward`);
  un único `Project` se codegen en un solo stage.
- **Tráfico O(N²) por fila**: cada nivel apilado re-materializaba la fila
  entera (N+1 campos) para escribir su columna.

La semántica es idéntica (mismo filtro `not-null`, mismos mapeos, mismos
nombres de salida; los tests semánticos de `tests/test_preprocess.py` —
frecuencias de binning, códigos de missing, cardinalidad, target binned —
pasan sin cambios sobre el nuevo `prepare()`).

## 4. Otras optimizaciones

### 4.1 `generate_chunks()`: generación y staging chunk a chunk (hecho)

Nuevo `sparkmim.benchmarks.generate_chunks`: misma estructura plantada,
producida en chunks de filas. Cada columna tiene su
propio `Generator` derivado de un único `SeedSequence(seed)`
(`spawn`); la secuencia dentro de cada columna es la de generarla
completa, por lo que **los valores no dependen de `chunk_rows`** (prueba
`test_generate_chunks_invariant_to_chunk_size`). Los valores difieren de
`generate()` (stream único compartido), que se mantiene intacto como
referencia de paridad bit a bit con `tests/planted.py`; el target se
calcula por chunk (misma fórmula).

`run_point` ahora stagea **cada chunk generado** a parquet
(`_staged_generate_df`), sin conservar los arrays completos: la memoria
del driver pasa de `O(n × N)` a `O(chunk × N)` (≈ el presupuesto de bytes
del lote, 128 MiB por defecto, más las copias transitorias del buffer
Arrow y del payload RPC — pico de unos 500 MB, holgado en un driver de
64 GB). Medición local (n=500k, N=100, driver del mismo proceso):

| Ruta | pico RSS (driver) |
|------|-------------------|
| `generate()` + `_staged_df` (arrays completos) | 672 MB (incluye los 400 MB de arrays) |
| `generate_chunks` + `_staged_generate_df` | 289 MB |

A 10M×25 equivale a ~2,08 GB → ~16 MB de arrays en el driver: el caso
N=1000 (kernel unresponsive en el notebook DBR) queda desbloqueado y la
rejilla grande ya no exige dimensionar el driver por `n × N`.

### 4.2 Restantes

1. **0b (cadenas `when`): no toca**. Con el input distribuido el job
   escala con los workers; sustituirlo por un pandas UDF de binning
   (`searchsorted`) añadiría un pase Python sobre *todo* el df, que no
   compensa para b≤20.
2. **Cuantiles aproximados en SQL (`percentile_approx`): opción**. Un solo
   job, sin Python; aproximado (error relativo). Solo tiene sentido si el
cálculo de estadísticas vuelve a dominar; hoy no.
3. **Tuning del cluster**: con staging, el paralelismo del benchmark ya no
   está limitado por el input; los tiempos residuales a 10M×25 estarán
   marcados por el *startup* de jobs (latencia del scheduler de DBR) y el
   throughput de red/DBFS — ahí, si hace falta, ayudan más workers (ya se
   usa todo el cluster) y `dbfs` vs UC Volume para el staging.

## 5. ¿Tiene sentido Rust?

**No como siguiente paso.**

- Tras (3.1)+(3.2), el cálculo por elemento es C (NumPy) y el cuello de
  botella es **distribución + latencia de arranque de jobs + transferencia
  Arrow/RPC** — nada de lo cual Rust se beneficia (vive en el scheduler de
  Spark, no en el UDF).
- Rust ganaría 2-4× sobre NumPy en el *cálculo* de estadísticas, que
  localmente es ~0,3 s por grupo de batches a N=500: dejaría de ser el
  cuello de botella antes de escribir una línea.
- El coste no es pequeño: wheels por plataforma (cp311 linux/windows/macOS),
  cadena de build con `maturin` en CI, wheel más grande y una pieza nativa
  más que instalar en DBR — a cambio de ganar en un tramo que ya no es
  crítico.
- **Cuando sí tendría sentido**: (a) N muy grande (1000+) con muchos bins,
  si el cálculo de estadísticas vuelve a dominar; (b) mover la etapa 3
  (greedy + oráculo) fuera del driver con Arrow zero-copy (hoy es NumPy y
  va en milisegundos); (c) si se abandona Spark para el benchmark (un
  generador/selector estilo Polars).

## 6. Cómo verificar

```bash
# Suite completa (~6-8 min local; Spark local + winutils; ver
# benchmarks/README.md, sección Windows)
python -m pytest tests/ -q

# Punto de escala local (misma vía que DBR, menos el cluster)
python benchmarks/bench_scale.py --n 50000 --N 25 --master local[4] --csv out.csv
```

Los números de DBR tras staging deben re-medirse en el cluster (no son
extrapolables desde local): se espera que e0 a 1M×25 baje de ~210 s al
orden de decenas de segundos (floor de latencia de jobs) y que la curva en
N se aplane.
