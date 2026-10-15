$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$sharedPython = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'work/web_venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $taskPython) -and (Test-Path -LiteralPath $sharedPython)) {
    $taskPython = $sharedPython
}
try {
    $running = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/status' -TimeoutSec 2
    if ($running.ready) {
        Write-Host 'Tynys іске қосылған. Браузерде http://127.0.0.1:7860 аш.'
        exit 0
    }
} catch {}
if (!(Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 орнатылғанын тексер.' }
    & $taskPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Кітапханалар орнатылмады. Интернетті тексер.' }
}
Write-Host 'Tynys: браузерде http://127.0.0.1:7860 аш. Тоқтату: Ctrl+C.'
& $taskPython app.py
