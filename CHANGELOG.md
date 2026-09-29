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
  `ModelFactory`; el coste de entrenamiento (2 + |ranking|) y la regla de
  detección de tarea se declaran en la interfaz; `detect_task` ahora es
  pública.
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

### Fixed

- **Índices del bucle greedy (KSG):** el bucle greedy indexaba la lista de
  candidatas (longitud K) con índices globales de feature; ahora opera en
  posiciones de candidata 0..K−1.

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
