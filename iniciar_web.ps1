Set-Location -Path $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error "No se encontro .venv. Crea el entorno con Python 3.10, 3.11, 3.12 o 3.13; Python 3.14 no sirve para TensorFlow en Windows."
    exit 1
}

& $VenvPython --version > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "El entorno .venv esta roto o apunta a un Python que ya no existe. Borra .venv y crealo de nuevo con Python 3.10, 3.11, 3.12 o 3.13."
    exit 1
}

& $VenvPython -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "La plataforma necesita Python 3.10, 3.11, 3.12 o 3.13 porque TensorFlow no tiene paquete compatible para Python 3.14 en Windows."
    exit 1
}

& $VenvPython -c "import uvicorn" > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Faltan dependencias. Ejecuta: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

& $VenvPython -m uvicorn web_app.app:app --host 127.0.0.1 --port 8000
