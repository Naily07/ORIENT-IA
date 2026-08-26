"""Tests des métriques d'évaluation (ML-5, ML-6)."""

from src.ml.entrainement import entrainer_baseline, preparer_jeu_entrainement, separer_train_test
from src.ml.evaluation import analyser_erreurs, evaluer_modele, mesurer_stabilite


def _jeu_a_trois_classes() -> list[dict]:
    exemples = []
    for _ in range(20):
        exemples.append(
            {"profil": {"matieres_preferees": ["informatique"]}, "parcours_id": "IGGLIA"}
        )
        exemples.append({"profil": {"matieres_preferees": ["droit"]}, "parcours_id": "DTJA"})
        exemples.append({"profil": {"matieres_preferees": ["biologie"]}, "parcours_id": "PIP"})
    return exemples


def test_evaluer_modele_retourne_des_metriques_dans_des_bornes_valides():
    X, y = preparer_jeu_entrainement(_jeu_a_trois_classes())
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.5)
    modele = entrainer_baseline(X_train, y_train)

    resultats = evaluer_modele(modele, X_test, y_test)

    assert 0.0 <= resultats["exactitude"] <= 1.0
    assert 0.0 <= resultats["precision_macro"] <= 1.0
    assert 0.0 <= resultats["f1_macro"] <= 1.0
    assert set(resultats["par_classe"]) == {"IGGLIA", "DTJA", "PIP"}
    assert 0.0 <= resultats["top_3_accuracy"] <= 1.0


def test_evaluer_modele_sur_un_jeu_parfaitement_separable_donne_l_exactitude_maximale():
    X, y = preparer_jeu_entrainement(_jeu_a_trois_classes())
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.5)
    modele = entrainer_baseline(X_train, y_train)
    resultats = evaluer_modele(modele, X_test, y_test)
    assert resultats["exactitude"] == 1.0
    # Aucune erreur : la confiance moyenne sur les erreurs doit rester absente.
    assert resultats["calibration"]["confiance_moyenne_predictions_erronees"] is None


def test_analyser_erreurs_ne_compte_que_les_confusions():
    X, y = preparer_jeu_entrainement(_jeu_a_trois_classes())
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.5)
    modele = entrainer_baseline(X_train, y_train)
    erreurs = analyser_erreurs(modele, X_test, y_test)
    assert erreurs == []  # jeu parfaitement séparable : aucune confusion
    assert all(e["vrai"] != e["predit"] for e in erreurs)


def test_mesurer_stabilite_retourne_une_exactitude_par_seed():
    X, y = preparer_jeu_entrainement(_jeu_a_trois_classes())
    resultats = mesurer_stabilite(entrainer_baseline, X, y, seeds=(0, 1, 2))
    assert len(resultats["exactitudes"]) == 3
    assert resultats["exactitude_ecart_type"] >= 0.0
