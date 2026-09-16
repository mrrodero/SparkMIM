# git-flow.ps1 — ayuda para el modelo git flow (nvie.com).
#
# Uso:
#   .\git-flow.ps1 feature start <nombre>
#   .\git-flow.ps1 feature finish <nombre>
#   .\git-flow.ps1 release start <version>
#   .\git-flow.ps1 release finish <version>
#   .\git-flow.ps1 hotfix start <version>
#   .\git-flow.ps1 hotfix finish <version>
#
# Ramas principales (vida infinita): main (producción), develop (integración).
# Ramas de soporte (vida limitada): feature-*, release-*, hotfix-*.
#
# Convenciones:
#   - feature: desde develop -> merge en develop (--no-ff).
#   - release: desde develop -> merge en main + tag v<version> + merge en develop.
#   - hotfix:  desde main    -> merge en main + tag v<version> + merge en develop.

$ErrorActionPreference = "Stop"
$main = "main"
$develop = "develop"
$scriptArgs = $args

function Run-Git([string[]]$gitArgs) {
    & git @gitArgs
    if ($LASTEXITCODE -ne 0) { throw "git $($gitArgs -join ' ') falló (exit $LASTEXITCODE)" }
}

function Require-Args([int]$n, [string]$cmd) {
    if ($scriptArgs.Count -lt $n) { throw "Uso: $cmd (faltan argumentos)" }
}

if ($scriptArgs.Count -lt 2) {
    Write-Host "Uso: .\git-flow.ps1 <feature|release|hotfix> <start|finish> <nombre|version>"
    exit 1
}

$tipo = $scriptArgs[0]
$accion = $scriptArgs[1]

switch ($tipo) {
    "feature" {
        switch ($accion) {
            "start" {
                Require-Args 3 "feature start <nombre>"
                $nombre = $scriptArgs[2]
                Run-Git @("checkout", "-b", "feature/$nombre", $develop)
                Write-Host "Rama feature/$nombre creada desde $develop."
            }
            "finish" {
                Require-Args 3 "feature finish <nombre>"
                $nombre = $scriptArgs[2]
                Run-Git @("checkout", $develop)
                Run-Git @("merge", "--no-ff", "feature/$nombre", "-m", "Merge feature/$nombre en ${develop}")
                Run-Git @("branch", "-d", "feature/$nombre")
                Write-Host "feature/$nombre fusionada en ${develop} y eliminada."
            }
            default { throw "Accion desconocida: $accion (usa start o finish)" }
        }
    }
    "release" {
        switch ($accion) {
            "start" {
                Require-Args 3 "release start <version>"
                $version = $scriptArgs[2]
                Run-Git @("checkout", "-b", "release-$version", $develop)
                Write-Host "Rama release-$version creada desde ${develop}. Recuerda actualizar la version (pyproject.toml) y commitar."
            }
            "finish" {
                Require-Args 3 "release finish <version>"
                $version = $scriptArgs[2]
                $rb = "release-$version"
                Run-Git @("checkout", $main)
                Run-Git @("merge", "--no-ff", $rb, "-m", "Merge $rb en ${main} (release v$version)")
                Run-Git @("tag", "-a", "v$version", "-m", "Release v$version")
                Run-Git @("checkout", $develop)
                Run-Git @("merge", "--no-ff", $rb, "-m", "Merge $rb en ${develop}")
                Run-Git @("branch", "-d", $rb)
                Write-Host "Release v${version}: fusionada en ${main} (tag v${version}) y ${develop}; rama eliminada."
            }
            default { throw "Accion desconocida: $accion (usa start o finish)" }
        }
    }
    "hotfix" {
        switch ($accion) {
            "start" {
                Require-Args 3 "hotfix start <version>"
                $version = $scriptArgs[2]
                Run-Git @("checkout", "-b", "hotfix-$version", $main)
                Write-Host "Rama hotfix-$version creada desde ${main}. Recuerda actualizar la version y commitar la correccion."
            }
            "finish" {
                Require-Args 3 "hotfix finish <version>"
                $version = $scriptArgs[2]
                $hb = "hotfix-$version"
                Run-Git @("checkout", $main)
                Run-Git @("merge", "--no-ff", $hb, "-m", "Merge $hb en ${main} (hotfix v$version)")
                Run-Git @("tag", "-a", "v$version", "-m", "Hotfix v$version")
                Run-Git @("checkout", $develop)
                Run-Git @("merge", "--no-ff", $hb, "-m", "Merge $hb en ${develop}")
                Run-Git @("branch", "-d", $hb)
                Write-Host "Hotfix v${version}: fusionada en ${main} (tag v${version}) y ${develop}; rama eliminada."
            }
            default { throw "Accion desconocida: $accion (usa start o finish)" }
        }
    }
    default { throw "Tipo desconocido: $tipo (usa feature, release o hotfix)" }
}
