# Manifeste des 14 livrables

État au 27 août 2026. Ce fichier est le point d'entrée de la remise ; `python backend/scripts/verifier_livrables.py` contrôle automatiquement les fichiers vérifiables.

| # | Exigence | Preuve dans le dépôt | État |
|---:|---|---|---|
| 1 | Code source complet | `backend/src/`, `backend/scripts/`, `frontend-next/`, `frontend/` | Prêt |
| 2 | Instructions d'installation et d'exécution | `README.md`, `.env.example`, `run.ps1`, `run.sh` | Prêt |
| 3 | Corpus ou collecte reproductible | `backend/data/corpus.json`, `corpus_genere.json`, fichiers structurés et `backend/scripts/generer_corpus_rag.py` | Prêt |
| 4 | Registre des sources | `backend/data/registre_sources.json` (statut, URL, date, données extraites, limites) | Prêt |
| 5 | Jeu de données ML | `backend/data/ml/profils_synthetiques.json` et variante `profils_synthetiques_realistes.json` | Prêt |
| 6 | Questionnaire, registre et réponses anonymisées | `backend/data/enquete/questionnaire.md`, `registre_collecte.json`, `registre_collecte.md`, `reponses_orientia.json` ; jeu secondaire anonymisé dans `backend/data/ml/jeu_test_reel.json` | Prêt, avec limites documentées |
| 7 | Notebooks d'analyse et d'entraînement | `backend/notebooks/01_analyse_exploratoire.ipynb`, `02_entrainement_et_evaluation.ipynb` | Prêt |
| 8 | Modèle ou script reproductible | `python -m src.ml.entrainement` depuis `backend/` ; produit `backend/data/ml/modele_recommandation.joblib` | Prêt par script |
| 9 | Jeu d'évaluation | `backend/tests/eval_dataset.json`, `eval_rag.json`, `backend/data/ml/jeu_test_reel.json` | Prêt |
| 10 | Résultats d'évaluation | `backend/tests/eval_results*.json` et `eval_analyse.md` | Prêt |
| 11 | Schéma d'architecture | `DOCS/ARCHITECTURE.md` | Prêt |
| 12 | Limites, biais et risques | `DOCS/LIMITES_BIAIS_RISQUES.md` et registres de collecte | Prêt |
| 13 | Vidéo fonctionnelle de 3 à 5 minutes | scénario et checklist dans `DOCS/VIDEO_DEMONSTRATION.md` | À enregistrer et joindre par l'équipe |
| 14 | Démonstration fonctionnelle | application lancée par `run.ps1`/`run.sh`, parcours de démo documenté | Prêt sous réserve de clé et réseau |

## Deux collectes à ne pas confondre

Le dépôt conserve deux ensembles anonymisés : notre formulaire ORIENT'IA (15 réponses reçues, 14 retenues), décrit dans `registre_collecte.json`, et un export secondaire plus large (86 réponses), transformé en `jeu_test_reel.json` et décrit dans `registre_collecte.md`. Les chiffres diffèrent donc légitimement. Cette distinction doit être dite au jury ; aucun des deux ensembles ne constitue un échantillon représentatif de tous les candidats.

## Checklist avant envoi

- Relancer le vérificateur, `pytest` et les contrôles frontend.
- Vérifier que `.env`, `backend/logs/` et `backend/chroma_db/` ne figurent pas dans l'archive.
- Ouvrir les deux notebooks et vérifier leur rendu.
- Enregistrer la vidéo avec l'application réellement manipulée, puis ajouter son fichier ou son lien à la remise.
- Faire une répétition avec la même machine, la même connexion et la même clé que le jour de la démonstration.

