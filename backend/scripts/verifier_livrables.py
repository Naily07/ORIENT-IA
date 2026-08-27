"""Vérifie les artefacts statiques de la remise ORIENT'IA.

Usage depuis la racine : ``python backend/scripts/verifier_livrables.py``.
La vidéo n'est pas vérifiable dans Git : le script rappelle qu'elle reste une
action humaine jusqu'à l'ajout d'un MP4 ou d'un lien dans la remise finale.
"""

from __future__ import annotations

import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]

FICHIERS_REQUIS = {
    "README": "README.md",
    "corpus": "backend/data/corpus.json",
    "mécanisme de corpus": "backend/scripts/generer_corpus_rag.py",
    "registre des sources": "backend/data/registre_sources.json",
    "jeu ML": "backend/data/ml/profils_synthetiques.json",
    "questionnaire": "backend/data/enquete/questionnaire.md",
    "registre de collecte JSON": "backend/data/enquete/registre_collecte.json",
    "registre de collecte lisible": "backend/data/enquete/registre_collecte.md",
    "réponses anonymisées": "backend/data/enquete/reponses_orientia.json",
    "notebook d'analyse": "backend/notebooks/01_analyse_exploratoire.ipynb",
    "notebook d'entraînement": "backend/notebooks/02_entrainement_et_evaluation.ipynb",
    "script d'entraînement": "backend/src/ml/entrainement.py",
    "jeu d'évaluation système": "backend/tests/eval_dataset.json",
    "jeu d'évaluation réel": "backend/data/ml/jeu_test_reel.json",
    "résultats système": "backend/tests/eval_results.json",
    "résultats ML": "backend/tests/eval_results_ml.json",
    "architecture": "DOCS/ARCHITECTURE.md",
    "limites, biais et risques": "DOCS/LIMITES_BIAIS_RISQUES.md",
    "conducteur vidéo": "DOCS/VIDEO_DEMONSTRATION.md",
    "manifeste": "DOCS/LIVRABLES.md",
}


def main() -> int:
    erreurs: list[str] = []
    for libelle, relatif in FICHIERS_REQUIS.items():
        chemin = RACINE / relatif
        if not chemin.is_file() or chemin.stat().st_size == 0:
            erreurs.append(f"{libelle}: fichier absent ou vide ({relatif})")
            continue
        if chemin.suffix in {".json", ".ipynb"}:
            try:
                with chemin.open(encoding="utf-8") as fichier:
                    json.load(fichier)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                erreurs.append(f"{libelle}: JSON illisible ({relatif}): {exc}")

    if erreurs:
        print("ÉCHEC — livrables statiques incomplets :")
        for erreur in erreurs:
            print(f"- {erreur}")
        return 1

    print(f"OK — {len(FICHIERS_REQUIS)} artefacts statiques présents et lisibles.")
    print("ACTION HUMAINE — enregistrer et joindre la vidéo fonctionnelle de 3 à 5 minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
