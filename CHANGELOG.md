# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/) y el
versionado sigue [SemVer](https://semver.org/lang/es/).

> **Convención (git flow):** la entrada de cada release/hotfix se escribe
> **en su rama de soporte** (`release-*`/`hotfix-*`), junto con el bump de
> versión en `pyproject.toml`, no antes. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Changed

- Reorganización de documentación: `DESIGN.md` movido a `docs/DESIGN.md`
  (versiones ES y EN juntas en `docs/`); eliminado `PLAN_IMPLEMENTACION.md`
  (hitos 0–7 completados) y redirigidas sus referencias a `docs/DESIGN.md §2`.

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
