# Flujo de trabajo (git flow)

Este repo sigue el [modelo de ramificación git flow](https://nvie.com/posts/a-successful-git-branching-model/).
Guía rápida para el mantenimiento habitual.

## Ramas

| Rama | Vida | Propósito |
|---|---|---|
| `main` | infinita | **Producción.** El `HEAD` siempre refleja un estado listo para producción. Cada commit en `main` es una release formal (etiquetada `vX.Y.Z`). |
| `develop` | infinita | **Integración.** Recibe las features terminadas. Refleja el estado del próximo release (no garantizado 100% estable). |
| `feature-*` | limitada | Nueva feature. Sale de `develop`, vuelve a `develop`. |
| `release-*` | limitada | Preparación de una release. Sale de `develop`, vuelve a `main` (+ tag) y a `develop`. |
| `hotfix-*` | limitada | Corrección urgente de producción. Sale de `main`, vuelve a `main` (+ tag) y a `develop`. |

## Comandos (script de ayuda)

El script `git-flow.ps1` envuelve los comandos:

```powershell
# Feature
.\git-flow.ps1 feature start mi-feature     # crea feature/mi-feature desde develop
.\git-flow.ps1 feature finish mi-feature    # merge --no-ff en develop + elimina la rama

# Release
.\git-flow.ps1 release start 0.2.0          # crea release-0.2.0 desde develop
.\git-flow.ps1 release finish 0.2.0         # merge en main + tag v0.2.0 + merge en develop

# Hotfix
.\git-flow.ps1 hotfix start 0.1.1           # crea hotfix-0.1.1 desde main
.\git-flow.ps1 hotfix finish 0.1.1          # merge en main + tag v0.1.1 + merge en develop
```

## Flujo de una feature

1. `.\git-flow.ps1 feature start mi-feature`
2. Desarrollar y commitear en `feature/mi-feature` (tests en verde).
3. `.\git-flow.ps1 feature finish mi-feature` → merge `--no-ff` en `develop`.

> Las features grandes o experimentales pueden no estar listas para el próximo
> release: se quedan en `develop` y entran en la siguiente release.

## Flujo de una release

1. Cuando `develop` refleja el estado deseado del próximo release:
   `.\git-flow.ps1 release start 0.2.0`.
2. En `release-0.2.0`: actualizar la versión en `pyproject.toml` y la
   entrada en `CHANGELOG.md` (ver formato abajo), y commitear. **No añadir
   features grandes aquí** (solo bug fixes y metadatos).
3. Cuando la release está lista: `.\git-flow.ps1 release finish 0.2.0`.
   - Merge `--no-ff` en `main` + tag `v0.2.0`.
   - Merge `--no-ff` en `develop` (para que el fix/meta entre en el siguiente release).
   - Elimina la rama `release-0.2.0`.

## Flujo de un hotfix

1. Bug crítico en producción: `.\git-flow.ps1 hotfix start 0.1.1` (desde `main`).
2. Corregir, actualizar la versión (patch), la entrada en `CHANGELOG.md` y
   commitear.
3. `.\git-flow.ps1 hotfix finish 0.1.1`.
   - Merge `--no-ff` en `main` + tag `v0.1.1`.
   - Merge `--no-ff` en `develop` (para que el fix entre en el siguiente release).
   - **Excepción:** si hay una rama `release-*` activa, el hotfix se mergea primero
     en esa rama (y luego, al cerrarla, en `develop`).
   - Elimina la rama `hotfix-0.1.1`.

## Convenciones

- **Versionado:** `vX.Y.Z` (semver). El bump de versión se hace **en la rama
  release/hotfix**, no antes.
- **Merges:** siempre `--no-ff` en `develop`/`main` (conserva el historial de la
  rama de soporte y permite revertir una feature completa).
- **Tests:** cada feature/release/hotfix debe dejar la suite en verde antes de
  mergear.
- **Tags:** anotados (`git tag -a`), prefijo `v`.
- **Changelog:** `CHANGELOG.md` (formato [Keep a Changelog](https://keepachangelog.com/es/1.1.0/)).
  Cada release/hotfix añade su entrada **en la rama de soporte** (`release-*`/
  `hotfix-*`), junto con el bump de versión. Los cambios en `develop` que aún
  no están en una release se acumulan bajo `[Unreleased]`.

## Estado actual

- `main` = `v0.1.0` (framework completo, hitos 0–7).
- `develop` = `v0.1.0` + reorganización de documentación y `CHANGELOG.md`
  (listo para nuevas features).
