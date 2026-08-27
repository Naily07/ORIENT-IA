# ORIENT'IA — frontend Next.js

Frontend candidat (chat) et backoffice admin, sur le même modèle que `frontend/`
(Streamlit) qu'il ne remplace pas — les deux coexistent. Voir `run.ps1`/`run.sh`
à la racine du dépôt pour lancer l'API + ce frontend ensemble.

## Lancement autonome

L'API FastAPI (`backend/`) doit tourner sur `http://localhost:8000` (ou l'URL
indiquée par `API_URL`).

```bash
cp .env.example .env.local   # ajuster API_URL / ORIENTIA_ADMIN_CODE / SESSION_SECRET
npm install
npm run dev
```

Ouvrir [http://localhost:3000](http://localhost:3000) — redirige vers `/chat`
(espace candidat). Le backoffice est sous `/admin` (protégé par
`ORIENTIA_ADMIN_CODE` s'il est défini, ouvert sinon).

## Structure

- `app/chat/` — espace candidat (chatbot).
- `app/admin/` — backoffice (login + tableau de bord, observabilité, qualité
  des données, corpus & graphe, mesures), protégé par `proxy.ts` + `lib/dal.ts`.
- `app/api/orientation/route.ts` — seul point de passage vers
  `POST /orientation/traiter` du backend (BFF).
- `lib/` — client API, types (miroir de `backend/src/schemas.py`), session
  admin, logique partagée.
- `components/chat/`, `components/admin/` — composants d'UI par espace.

## Vérification

```bash
npm run lint
npx tsc --noEmit
```
