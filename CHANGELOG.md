# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/) y el
versionado sigue [SemVer](https://semver.org/lang/es/).

> **Convención (git flow):** la entrada de cada release/hotfix se escribe
> **en su rama de soporte** (`release-*`/`hotfix-*`), junto con el bump de
> versión en `pyproject.toml`, no antes. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Changed

- **Costura de oráculo de información** (`oracles.py`, `selection.py`,
  `criteria.py`): la selección greedy (etapa 3) ahora opera detrás de la
  interfaz `InformationOracle` con dos adaptadores (`HistogramOracle`,
  `KsgOracle`); un único bucle greedy en posiciones de candidata (ADR-0001).
- **Ranking de población completa:** `SelectorModel.ranking_` cubre las N
  features (no solo las candidatas) por MI univariante descendente.
- **Significancia detrás de una costura** (`significance.py`): interfaz
  `SignificanceTest` con tres adaptadores (`Chi2Test`, `PermutationTest`,
  `NoTest`) y control BH-FDR dentro de cada adaptador; `screen.py` solo
  compone tablas → MI → significancia → top-K.
- Reorganización de documentación: `DESIGN.md` movido a `docs/DESIGN.md`
  (versiones ES y EN juntas en `docs/`); eliminado `PLAN_IMPLEMENTACION.md`
  (hitos 0–7 completados) y redirigidas sus referencias a `docs/DESIGN.md §2`.
- **Evaluación detrás de una costura** (`model_factory.py`, `evaluate.py`):
  nueva interfaz `ModelFactory` con tres adaptadores (`GbtFactory`,
  `XgboostFactory`, `LightgbmFactory`); `train_and_evaluate`,
  `efficiency_curve` y `report` aceptan un nombre de backend o una
  `ModelFactory`; el coste de entrenamiento (2 + |ranking|) se declara en la
  interfaz y la task llega ya resuelta (3 valores) sobre la que los
  adaptadores ramifican.
- **Caché con accesores orientados** (`tables.py`, `oracles.py`):
  `TableCache.cmi_table(i, j)` devuelve la triple en la orientación que
  espera `conditional_mi` (x, Y, z), sin importar el orden de (i, j); la
  canonicidad (min, max) y el conocimiento de ejes quedan en un solo sitio
  (el caché) y `HistogramOracle.cmi_single` queda como fórmula delgada.
- **Config por modo** (`config.py`, `selector.py`): `SelectorConfig` declara
  la relevancia por modo en un solo sitio (campos de histograma / de KSG /
  compartidos); un campo del otro modo distinto de su valor por defecto lanza
  `ValueError` en la construcción; el modo KSG exige columnas numéricas con
  error claro en `fit` (no un crash profundo en `to_numpy`).
- **Generador de estructura plantada compartido** (`tests/planted.py`,
  `benchmarks/synthetic.py`): los tests (selector, KSG, evaluación, screening,
  significancia) y el benchmark comparten un único generador parametrizado por
  roles de feature (informativa/independiente/redundante/correlada) y modo de
  target (`and`/`linear`/`flip`); `benchmarks/synthetic.generate` queda como
  adaptador fino sobre `planted.make_planted` (misma firma, mismos errores).
- **Task declarada por el usuario** (`schema.py`, `config.py`,
  `preprocess.py`, `model.py`, `selector.py`, `evaluate.py`,
  `model_factory.py`): nueva `SelectorConfig.task` (`"auto"` |
  `"classifier_binary"` | `"classifier_multiclass"` | `"continuous"`, campo
  compartido de ambos modos); la task se resuelve una vez en la etapa 0 con la
  regla compartida `schema.resolve_task` (validando contra los datos) y viaja
  en `SelectorModel.task` hasta el `report`, que mide el Target crudo. La
  regla `"auto"` es la antigua detección de `evaluate.detect_task`, ahora
  eliminada de la API pública; el vocabulario legacy `"classification"` /
  `"regression"` sigue aceptándose en `train_and_evaluate`, `efficiency_curve`
  y `report` (mapa a `"classifier_multiclass"` / `"continuous"`). El modo KSG
  ahora soporta clasificación: target no numérico con codificación lossless
  por valor distinto (ADR-0002).
- **Etapa 1 real del modo KSG** (`screen.py`, `selector.py`): el modo KSG
  comparte la política de la etapa 1 con el histograma — corte top-K por la
  pura `screen.select_candidates` (sin filtro de significancia, pendiente) y
  ranking de población por la pura `screen.rank` — y expone timings por etapa
  (`etapa0`, `etapa1`, `etapa3` y `total`; no hay etapa 2). La reducción de
  filas se unifica en la función compartida a nivel de módulo
  `selector.subsample` (etapa 2 con `subsample`, KSG con `ksg_subsample`).
- **Un solo hogar para las pasadas de conteo conjunto** (`tables.py`): el núcleo
  `joint_counts(cols, sizes)` (pack→bincount→reshape, forma `tuple(sizes)` en el
  orden de las columnas) resuelve las tres pasadas — screening (2 columnas),
  tablas conjuntas de etapa 2 (3) y CMIM por ronda (4) —; los builders de
  screening y de etapa 2 quedan como adaptadores delgados sobre él, y
  `cmim_scores` se traslada a `tables.py` junto a su `mapInPandas`.
  `criteria.py` conserva solo las fórmulas de criterios (`CRITERIA` +
  `criterion_score`). En `info/entropy.py` `conditional_mi` acepta cualquier
  grado (ndim ≥ 2): `mutual_information` pasa a ser el alias 2D y
  `conditional_mi_multi` desaparece. `ksg_cmi` con condicionante vacío delega en
  `ksg_mi`.
- **Traducción de índices dentro de la caché** (`tables.py`, `selector.py`):
  `TableCache.from_screening(screen_result, candidates, triples)` es el camino
  del pipeline — la traducción de índices globales de Feature a posiciones de
  Candidate (0..K−1) y la derivación de `n_codes`/`n_y` desde las formas de las
  tablas de screening viven en la caché, que además valida el cableado por
  construcción (claves canónicas dentro de 0..K−1, todos los pares presentes,
  formas `(n_i, n_j, n_y)`) y lanza `ValueError` si algo no corresponde.
  `TableCache.__init__` sigue disponible para construcción directa.

### Fixed

- **Índices del bucle greedy (KSG):** el bucle greedy indexaba la lista de
  candidatas (longitud K) con índices globales de feature; ahora opera en
  posiciones de candidata 0..K−1.
- **`Decimal` como numérico en el modo histograma:** ahora se acepta (unificado
  con el modo KSG, que ya la aceptaba); antes lanzaba `TypeError`.

## [0.1.0] - 2026-09-16

Primera release: framework completo (hitos 0–7).

### Added

- **Núcleo de información** (`src/sparkmim/info/entropy.py`): entropía, MI y
  CMI desde tablas de contingencia (numpy vectorizado).
- **Tablas en pase único** (`src/sparkmim/tables.py`): crosstabs de pares y
  triples con UN `mapInPandas` + UN `groupBy` (elimina los O(N) jobs de las
  implementaciones previas) + `TableCache` acotado por `max_cache_cells`.
- **Detección de esquema y preprocesado** (`schema.py`, `preprocess.py`):
  pase 0a (cuantiles/cardinalidades en un solo `mapInPandas`), pase 0b (mapeo
  sin shuffle), bins cuantiles, missing como categoría dedicada, top-C +
  "other" para cardinalidad alta.
- **Screening univariante** (`screen.py`): etapa 1 en un solo pase sobre
  `df_prep` + MI por feature.
- **Significancia** (`significance.py`): χ² (G² = 2n·MI), test de permutación
  distribuido y control BH-FDR.
- **Criterios greedy** (`criteria.py`): JMIM (default), JMI, mRMR, mIM y CMIM
  (exacto y aproximado `max_min`).
- **Selector y modelo** (`selector.py`, `model.py`): `InfoSelector` + variantes
  (`JMIMSelector`, `CMIMSelector`, `MRMRSelector`, `MIMSelector`);
  `SelectorModel` con `transform`, ranking completo y timings por etapa.
- **Modo KSG opcional** (`info/ksg.py`): estimador kNN de
  Kraskov-Stögbauer-Grassberger para variables continuas (subsample ≤ 250k
  filas al driver).
- **Evaluación agnóstica al modelo** (`evaluate.py`): `report()` con GBT
  spark.ml (default) o XGBoost/LightGBM (extras opcionales); AUC/R² vs
  baseline + curva de eficiencia.
- **Benchmarks** (`benchmarks/`): generador sintético con estructura plantada
  (informativas + redundantes + ruido) y benchmark de escala (rejilla
  n × N, wall-time por etapa, memoria driver, salida CSV).
- **Documentación bilingüe**: `README.md`/`README.en.md` y
  `DESIGN.md`/`docs/DESIGN.en.md`.
- **Flujo de trabajo git flow**: `CONTRIBUTING.md`, script de ayuda
  `git-flow.ps1` (feature/release/hotfix), ramas `main`/`develop` y tag
  anotado `v0.1.0`.
