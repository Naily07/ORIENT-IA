"""Évaluation consolidée du bloc ML — produit `eval_results_ml.json` (livrable
« résultats de l'évaluation », §9.6 du sujet).

Compare baseline (régression logistique) et modèle retenu (forêt aléatoire)
sur le même split, plus une mesure de stabilité sur plusieurs seeds.

    python -m backend.tests.eval_ml
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.ml.entrainement import (
    entrainer_baseline,
    entrainer_foret,
    preparer_jeu_entrainement,
    separer_train_test,
)
from src.ml.evaluation import analyser_erreurs, evaluer_modele, mesurer_stabilite


def evaluer_ml() -> dict:
    X, y = preparer_jeu_entrainement()
    if X.size == 0:
        raise RuntimeError(
            "Aucun jeu de données ML disponible : lancer "
            "`python -m src.ml.donnees_synthetiques` pour le générer."
        )

    X_train, X_test, y_train, y_test = separer_train_test(X, y)

    baseline = entrainer_baseline(X_train, y_train)
    foret = entrainer_foret(X_train, y_train)

    return {
        "date": datetime.now(UTC).isoformat(),
        "taille_jeu_de_donnees": len(y),
        "taille_test": len(y_test),
        "avertissement": (
            "Évaluation sur données synthétiques uniquement (voir "
            "donnees_synthetiques.py) : mesure la capacité du modèle à "
            "retrouver les hypothèses de génération, pas sa capacité à "
            "orienter de vrais candidats. La validation sur l'enquête réelle "
            "(ML-7) reste à faire une fois DATA-4/DATA-7 disponibles."
        ),
        "baseline_regression_logistique": evaluer_modele(baseline, X_test, y_test),
        "modele_foret_aleatoire": evaluer_modele(foret, X_test, y_test),
        "erreurs_frequentes_foret_aleatoire": analyser_erreurs(foret, X_test, y_test),
        "stabilite_foret_aleatoire": mesurer_stabilite(entrainer_foret, X, y),
    }


def sauvegarder(resultats: dict, chemin: Path | None = None) -> Path:
    chemin = chemin or (Path(__file__).parent / "eval_results_ml.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    return chemin


if __name__ == "__main__":
    resultats = evaluer_ml()
    chemin = sauvegarder(resultats)
    print(json.dumps(resultats, indent=2, ensure_ascii=False))
    print(f"\nRésultats écrits dans {chemin}")
