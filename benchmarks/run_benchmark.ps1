# SparkMIM — Benchmark de escala (doble clic para lanzar).
# Corre la rejilla completa n × N (hasta 10M filas × 1000 features),
# un JVM nuevo por punto, y abre el informe HTML al terminar.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"

# JDK 17 (requisito de PySpark para Arrow)
$env:JAVA_HOME = Join-Path $root "jdk17"
$env:PATH = (Join-Path $env:JAVA_HOME "bin") + ";" + $env:PATH

# TEMP local al workspace: el launcher de Spark escribe ficheros temporales
# en %TEMP%; fuera del workspace algunos entornos lo deniegan.
$tmp = Join-Path $PSScriptRoot ".tmp"
if (-not (Test-Path $tmp)) { New-Item -ItemType Directory -Path $tmp | Out-Null }
$env:TEMP = $tmp
$env:TMP = $tmp

# Python del venv para los workers de Spark (evita el alias de Microsoft Store)
$env:PYSPARK_PYTHON = $py

Set-Location $PSScriptRoot
Write-Host ""
Write-Host "SparkMIM benchmark — rejilla completa (10M x 1000)" -ForegroundColor Cyan
Write-Host "Salidas: bench_grid.json / bench_grid_report.html / bench_grid.log" -ForegroundColor DarkGray
Write-Host ""

& $py "bench_grid.py"
$code = $LASTEXITCODE

$report = Join-Path $PSScriptRoot "bench_grid_report.html"
if (Test-Path $report) {
    Start-Process $report
}

Write-Host ""
if ($code -eq 0) {
    Write-Host "Benchmark finalizado. Informe abierto en el navegador." -ForegroundColor Green
} else {
    Write-Host "El benchmark termino con errores (ver bench_grid.log)." -ForegroundColor Red
}
Read-Host "Pulsa Enter para cerrar"
