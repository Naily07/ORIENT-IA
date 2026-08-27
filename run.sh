#!/bin/bash
# Fichier de lancement : démarre l'API FastAPI puis un frontend (Next.js par
# défaut, Streamlit en option via --frontend streamlit), et arrête proprement
# les processus à la sortie (Ctrl-C).
set -e

cd "$(dirname "$0")"

FRONTEND="next"
if [ "$1" = "--frontend" ]; then
    FRONTEND="$2"
fi

if [ ! -f .env ]; then
    echo "ERREUR : .env absent. Copier .env.example vers .env et y renseigner GEMINI_API_KEY."
    echo "Clé à générer sur https://aistudio.google.com/apikey"
    exit 1
fi

# Charge les variables du .env racine dans l'environnement du process courant
# (API_URL/ORIENTIA_ADMIN_CODE/SESSION_SECRET notamment) — jusqu'ici seul
# pydantic-settings les lisait côté backend, ni ce script ni noyau.py ne les
# exportaient pour un frontend.
set -a
source .env
set +a

RACINE="$(pwd)"
PYTHON="$RACINE/.venv/Scripts/python.exe"
[ -x "$PYTHON" ] || PYTHON="$RACINE/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python"

# `src` n'est importable que depuis backend/ (package installé en editable
# avec package-dir = backend) : lancer depuis la racine échoue en
# ModuleNotFoundError.
(cd "$RACINE/backend" && "$PYTHON" -m uvicorn src.api:app --port 8000) &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

sleep 2
export API_URL="${API_URL:-http://localhost:8000}"

if [ "$FRONTEND" = "streamlit" ]; then
    "$PYTHON" -m streamlit run "$RACINE/frontend/app.py" --server.port 8501
else
    cd "$RACINE/frontend-next"
    [ -d node_modules ] || npm install
    npm run dev
fi
