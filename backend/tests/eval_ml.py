"""Évaluation consolidée du bloc ML — produit `eval_results_ml.json` (livrable
« résultats de l'évaluation », §9.6 du sujet).

Deux niveaux de mesure, volontairement distincts :

1. **Les estimateurs** — baseline (régression logistique) contre forêt
   aléatoire, sur le même découpage : classification, classement (MRR, NDCG,
   PR-AUC), calibration (ECE, Brier), matrice de confusion, stabilité entre
   graines.
2. **Le chemin réellement servi** — `ml.outils.analyser_profil`, qui ajoute à
   l'estimateur la résolution du vocabulaire ouvert, le garde-fou
   d'exploitabilité et les règles d'admission du volet hybride. Ces trois
   étages **réordonnent** le classement : mesurer le seul estimateur
   reviendrait à publier les chiffres d'un modèle que personne n'exécute, ce
   que le §8 interdit (« le modèle ne devra pas rester isolé ») et que le §14
   contrôle (« cohérence entre le modèle ML et la réponse finale »).

    cd backend && python -m tests.eval_ml
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import config
from src.ml import outils
from src.ml.donnees_synthetiques import charger_jeu_de_donnees
from src.ml.entrainement import (
    entrainer_baseline,
    entrainer_baseline_calibree,
    entrainer_foret,
    preparer_jeu_entrainement,
    separer_indices,
    separer_train_test,
)
from src.ml.evaluation import (
    analyser_erreurs,
    evaluer_chemin_de_production,
    evaluer_modele,
    mesurer_stabilite,
    mesurer_stabilite_des_recommandations,
)
from src.schemas import ProfilCandidat


def _mesurer_chemin_de_production(exemples_test: list[dict], modele_train) -> dict:
    """Mesure `analyser_profil` avec un modèle entraîné sur le seul train.

    Sans cette substitution, `analyser_profil` utiliserait le modèle de
    production, entraîné sur **tout** le jeu de données : les profils de test
    auraient été vus à l'entraînement et les chiffres n'auraient aucune valeur.
    """
    avec_serie_bac = sum(1 for e in exemples_test if (e["profil"].serie_bac or "").strip())

    outils.imposer_modele_pour_evaluation(modele_train)
    try:
        return {
            "classement": evaluer_chemin_de_production(exemples_test, outils.analyser_profil),
            "stabilite_des_recommandations": mesurer_stabilite_des_recommandations(
                exemples_test, outils.analyser_profil
            ),
            "profils_de_test_avec_serie_bac_declaree": avec_serie_bac,
            "lecture": (
                "Mesuré à travers ml.outils.analyser_profil : vocabulaire ouvert, "
                "garde-fou d'exploitabilité et règles d'admission compris. "
                f"{avec_serie_bac} profil(s) de test sur {len(exemples_test)} déclarent "
                "une série de baccalauréat : les règles d'admission (ml/hybride.py) ne "
                "s'appliquent qu'à ceux-là et sont donc inertes sur ce jeu, faute d'une "
                "série générée par donnees_synthetiques.py. L'écart attendu entre "
                "estimateur et chemin de production ne sera mesurable qu'une fois "
                "l'enquête réelle disponible (DATA-4, ML-7) — c'est une limite du jeu "
                "synthétique, pas une absence d'effet des règles, dont l'effet est "
                "démontré à part dans backend/tests/ml/test_hybride.py."
            ),
        }
    finally:
        outils.imposer_modele_pour_evaluation(None)


def _charger_jeu_test_reel() -> list[dict]:
    """Profils réels étiquetés (DATA-5/DATA-7), filtrés sur ceux dont
    l'étiquette de parcours a pu être résolue (`usable_pour_eval`) — un
    répondant non rattachable à un parcours connu n'a rien à mesurer contre."""
    chemin = config.dossier_data / "ml" / "jeu_test_reel.json"
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        brut = json.load(f)
    return [
        {"profil": ProfilCandidat.model_validate(e["profil"]), "parcours_id": e["parcours_id"]}
        for e in brut
        if e.get("usable_pour_eval")
    ]


def _mesurer_generalisation_reelle() -> dict:
    """ML-7 : le modèle de production mesuré sur des profils réels qu'il n'a
    jamais vus.

    Contrairement à `_mesurer_chemin_de_production`, aucune substitution de
    modèle n'est nécessaire : ces profils ne font partie d'aucun jeu
    d'entraînement synthétique (DATA-6), donc pas de fuite train/test à
    éviter — `outils.analyser_profil` est appelé tel quel, avec le modèle
    réellement servi à un candidat.
    """
    exemples = _charger_jeu_test_reel()
    if not exemples:
        return {
            "disponible": False,
            "lecture": (
                "backend/data/ml/jeu_test_reel.json absent ou vide — lancer "
                "backend/scripts/preparer_jeu_test_reel.py --source <export.csv> "
                "(DATA-5/DATA-7) pour le produire."
            ),
        }
    return {
        "disponible": True,
        "taille": len(exemples),
        "classement": evaluer_chemin_de_production(exemples, outils.analyser_profil),
        "stabilite_des_recommandations": mesurer_stabilite_des_recommandations(
            exemples, outils.analyser_profil
        ),
        "lecture": (
            f"{len(exemples)} profils réels (enquête de terrain), étiquette de parcours "
            "reconnue, jamais vus à l'entraînement — mesuré avec le modèle de production "
            "réellement servi (ml.outils.analyser_profil), sans substitution. L'étiquette "
            "est le parcours effectivement suivi par le répondant, indépendamment de sa "
            "satisfaction déclarée : un score bas ici peut aussi bien révéler une limite du "
            "modèle qu'un candidat qui s'est réorienté après coup. Échantillon de taille "
            "réduite et concentré sur quelques mentions — voir "
            "backend/data/enquete/registre_collecte.md pour les biais constatés avant de "
            "sur-interpréter un écart avec les chiffres synthétiques ci-dessus."
        ),
    }


def evaluer_ml() -> dict:
    exemples = charger_jeu_de_donnees()
    X, y = preparer_jeu_entrainement(exemples)
    if X.size == 0:
        raise RuntimeError(
            "Aucun jeu de données ML disponible : lancer "
            "`python -m src.ml.donnees_synthetiques` pour le générer."
        )

    X_train, X_test, y_train, y_test = separer_train_test(X, y)
    # Même partition, appliquée aux profils d'origine : le chemin de production
    # prend un ProfilCandidat, pas un vecteur.
    _, indices_test = separer_indices(y)
    exemples_test = [
        {
            "profil": ProfilCandidat.model_validate(exemples[i]["profil"]),
            "parcours_id": exemples[i]["parcours_id"],
        }
        for i in indices_test
    ]

    baseline = entrainer_baseline(X_train, y_train)
    baseline_calibree = entrainer_baseline_calibree(X_train, y_train)
    foret = entrainer_foret(X_train, y_train)

    return {
        "date": datetime.now(UTC).isoformat(),
        "taille_jeu_de_donnees": len(y),
        "taille_test": len(y_test),
        "avertissement": (
            "Les sections ci-dessous (hors `generalisation_reelle_ml_7`) évaluent sur "
            "données synthétiques uniquement (voir donnees_synthetiques.py) : elles "
            "mesurent la capacité du modèle à retrouver les hypothèses de génération, pas "
            "sa capacité à orienter de vrais candidats. `generalisation_reelle_ml_7` "
            "apporte cette seconde mesure, sur un échantillon réel mais réduit — lire son "
            "propre avertissement avant de la traiter comme définitive."
        ),
        # **Le modèle réellement servi** : régression logistique calibrée.
        # C'est cette section qu'il faut lire pour juger le système ; les deux
        # suivantes sont des points de comparaison.
        "modele_de_production_calibre": evaluer_modele(baseline_calibree, X_test, y_test),
        "baseline_regression_logistique": evaluer_modele(baseline, X_test, y_test),
        "modele_foret_aleatoire": evaluer_modele(foret, X_test, y_test),
        "erreurs_frequentes_foret_aleatoire": analyser_erreurs(foret, X_test, y_test),
        "stabilite_foret_aleatoire": mesurer_stabilite(entrainer_foret, X, y),
        "apport_de_la_calibration": {
            "lecture": (
                "La régression logistique brute est nettement sous-confiante sur ce "
                "jeu : elle réussit plus souvent qu'elle ne l'annonce, donc le score "
                "montré à un candidat ne correspond à aucune fréquence réelle. La "
                "calibration isotonique (bornée, voir ml/entrainement.ModeleBorne) "
                "corrige cet écart sans toucher à l'exactitude ni neutraliser le "
                "garde-fou d'escalade."
            ),
            "ece_avant": evaluer_modele(baseline, X_test, y_test)["calibration"]["ece"],
            "ece_apres": evaluer_modele(baseline_calibree, X_test, y_test)["calibration"][
                "ece"
            ],
        },
        # Le chemin de production est mesuré avec le modèle **calibré**, celui
        # que `ml.outils` sert réellement : y injecter la baseline brute
        # recréerait l'écart « modèle évalué ≠ modèle servi » que le §8 interdit.
        "chemin_de_production": _mesurer_chemin_de_production(
            exemples_test, baseline_calibree
        ),
        "generalisation_reelle_ml_7": _mesurer_generalisation_reelle(),
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
