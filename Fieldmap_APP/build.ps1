# Build FieldMapDAQ using the local venv
$ErrorActionPreference = "Stop"

if (!(Test-Path ".\.venv\Scripts\activate.ps1")) {
  Write-Host "No .venv found. Create it first:"
  Write-Host "  python -m venv .venv"
  exit 1
}

.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pyinstaller

# Clean old builds
if (Test-Path ".\build") { Remove-Item -Recurse -Force .\build }
if (Test-Path ".\dist")  { Remove-Item -Recurse -Force .\dist }

pyinstaller .\build.spec
Write-Host "Build done. Output is in .\dist\FieldMapDAQ\FieldMapDAQ.exe"
