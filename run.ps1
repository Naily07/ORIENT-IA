# Fichier de lancement Windows (équivalent de run.sh).
# Démarre l'API FastAPI en arrière-plan puis un frontend (Next.js par défaut,
# Streamlit en option via -Frontend streamlit) ; l'API est arrêtée quand on
# quitte le frontend.
param(
    [ValidateSet("next", "streamlit")]
    [string]$Frontend = "next"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Write-Error "`.env` absent. Copier .env.example vers .env et y renseigner GEMINI_API_KEY (https://aistudio.google.com/apikey)."
    exit 1
}

# Charge les variables du .env racine dans l'environnement du process courant
# (API_URL/ORIENTIA_ADMIN_CODE/SESSION_SECRET notamment) — jusqu'ici seul
# pydantic-settings les lisait côté backend, ni ce script ni noyau.py ne les
# exportaient pour un frontend.
Get-Content ".env" | ForEach-Object {
    $ligne = $_.Trim()
    if ($ligne -and -not $ligne.StartsWith("#") -and $ligne -match '^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}

$racine = $PSScriptRoot
$python = Join-Path $racine ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# `src` n'est importable que depuis backend/ (package installé en editable
# avec package-dir = backend) : lancer depuis la racine échoue en
# ModuleNotFoundError.
$api = Start-Process -FilePath $python -ArgumentList "-m","uvicorn","src.api:app","--port","8000" `
    -WorkingDirectory (Join-Path $racine "backend") -PassThru -NoNewWindow
try {
    Start-Sleep -Seconds 2
    if (-not $env:API_URL) { $env:API_URL = "http://localhost:8000" }

    if ($Frontend -eq "streamlit") {
        & $python -m streamlit run (Join-Path $racine "frontend\app.py") --server.port 8501
    }
    else {
        $frontendNext = Join-Path $racine "frontend-next"
        Set-Location $frontendNext
        if (-not (Test-Path "node_modules")) { & npm install }
        & npm run dev
    }
}
finally {
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}
