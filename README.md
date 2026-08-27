# ORIENT'IA

ORIENT'IA est un assistant d'aide à l'orientation vers les 16 parcours de l'ISPM. Il combine un modèle de classement supervisé, des règles d'admission, un corpus sourcé, une recherche RAG et un agent conversationnel. Les recommandations restent indicatives et doivent être confirmées par un conseiller pédagogique ou l'administration.

## Version à jour et déploiements

La branche principale contenant la version à jour du projet est **`develop`**.

- Backend (documentation de l'API) : [https://orient-ia-production.up.railway.app/docs](https://orient-ia-production.up.railway.app/docs)
- Frontend : [https://x-project-orient-ia.vercel.app/chat](https://x-project-orient-ia.vercel.app/chat)

## Démarrage rapide

### Prérequis

- Python 3.11 ou supérieur ;
- Node.js 20 ou supérieur et npm ;
- une clé Google AI Studio pour les réponses conversationnelles (`GEMINI_API_KEY`).

### Installation

Sous Windows PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
cd frontend-next
npm ci
cd ..
```

Sous Linux ou macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
cp .env.example .env
cd frontend-next && npm ci && cd ..
```

Renseigner ensuite `GEMINI_API_KEY` dans `.env`. `ORIENTIA_ADMIN_CODE` et `SESSION_SECRET` sont facultatifs pour une démonstration locale ; ne pas laisser le backoffice ouvert en production.

### Exécution

```powershell
.\run.ps1
```

ou :

```bash
./run.sh
```

Le frontend candidat est disponible sur <http://localhost:3000/chat>, le backoffice sur <http://localhost:3000/admin>, l'API sur <http://localhost:8000> et son état sur <http://localhost:8000/health>.

Pour utiliser l'ancienne interface Streamlit : `./run.sh --frontend streamlit` ou `.\run.ps1 -Frontend streamlit`, puis ouvrir <http://localhost:8501>.

## Vérification

La suite hors appels LLM réels :

```bash
pytest
ruff check backend frontend
cd frontend-next && npm run lint && npx tsc --noEmit
```

Le manifeste de remise et les JSON/notebooks :

```bash
python backend/scripts/verifier_livrables.py
```

Les évaluations reproductibles :

```bash
cd backend
python -m tests.eval_ml
python -m tests.eval_ontologie
python -m tests.eval_rag
python -m tests.eval_system
```

`eval_system` appelle réellement Gemini et consomme du quota. Les tests marqués `reseau` ou `index` sont exclus de la commande `pytest` par défaut.

## Reproduire les données et le modèle

Depuis `backend/` :

```bash
# Jeu d'entraînement déterministe (graine 42)
python -m src.ml.donnees_synthetiques --seed 42

# Variante calée sur la distribution observée dans les enquêtes
python -m src.ml.donnees_synthetiques --cale-sur-enquete --n-total 800 --seed 42

# Modèle de production sérialisé (le .joblib est volontairement ignoré par Git)
python -m src.ml.entrainement

# Corpus RAG dérivé des fichiers structurés
python -m scripts.generer_corpus_rag
```

L'import reproductible d'un nouvel export d'enquête est documenté dans `backend/scripts/preparer_jeu_test_reel.py`. Le fichier brut livré ne doit être rediffusé qu'en accord avec le consentement et la note de risque de ré-identification.

## Livrables du hackathon

La correspondance exhaustive entre les 14 exigences et les fichiers du dépôt est dans [DOCS/LIVRABLES.md](DOCS/LIVRABLES.md). Les documents centraux sont :

- [architecture](DOCS/ARCHITECTURE.md) ;
- [limites, biais et risques](DOCS/LIMITES_BIAIS_RISQUES.md) ;
- [scénario de vidéo et démonstration](DOCS/VIDEO_DEMONSTRATION.md) ;
- [registre des sources](backend/data/registre_sources.json) ;
- [analyse des évaluations](backend/tests/eval_analyse.md).

## Structure

```text
backend/src/          API, agent, RAG, règles, sécurité et ML
backend/data/         corpus, sources, enquêtes et jeux ML
backend/notebooks/    analyse exploratoire, entraînement et évaluation
backend/tests/        tests, jeux d'évaluation et résultats mesurés
frontend-next/        interface candidat et backoffice principal
frontend/             interface Streamlit de secours
DOCS/                 dossier de remise et documentation transversale
```

## Confidentialité et portée

Ne jamais committer `.env`, les journaux, l'index Chroma ou un export contenant des identifiants. Les données d'enquête publiées ont été anonymisées, mais les petits effectifs conservent un risque de ré-identification indirecte. Le modèle apprend principalement sur des profils synthétiques : ses scores mesurent d'abord la cohérence avec les hypothèses de génération, pas la réussite future d'une personne.
