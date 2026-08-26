"""Tests du pipeline d'entraînement (ML-3, ML-4)."""

import numpy as np

from src.ml.entrainement import (
    entrainer_baseline,
    entrainer_foret,
    preparer_jeu_entrainement,
    separer_train_test,
)


def _exemples_jouets() -> list[dict]:
    """Un jeu minimal mais séparable, pour des tests rapides et déterministes."""
    exemples = []
    for _ in range(20):
        exemples.append(
            {
                "profil": {"matieres_preferees": ["informatique"], "resultats_scolaires": {}},
                "parcours_id": "IGGLIA",
            }
        )
        exemples.append(
            {
                "profil": {"matieres_preferees": ["droit"], "resultats_scolaires": {}},
                "parcours_id": "DTJA",
            }
        )
    return exemples


def test_preparer_jeu_entrainement_sans_donnees_retourne_des_tableaux_vides(monkeypatch):
    import src.ml.entrainement as module

    monkeypatch.setattr(module, "charger_jeu_de_donnees", lambda: [])
    X, y = preparer_jeu_entrainement()
    assert X.size == 0
    assert y.size == 0


def test_preparer_jeu_entrainement_vectorise_les_exemples():
    X, y = preparer_jeu_entrainement(_exemples_jouets())
    assert X.shape[0] == 40
    assert set(y) == {"IGGLIA", "DTJA"}


def test_separer_train_test_est_stratifie():
    X, y = preparer_jeu_entrainement(_exemples_jouets())
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.5)
    assert set(y_test) == {"IGGLIA", "DTJA"}
    assert len(y_train) == len(y_test) == 20


def test_baseline_apprend_un_jeu_separable():
    X, y = preparer_jeu_entrainement(_exemples_jouets())
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.5)
    modele = entrainer_baseline(X_train, y_train)
    assert np.mean(modele.predict(X_test) == y_test) == 1.0


def test_foret_apprend_un_jeu_separable():
    X, y = preparer_jeu_entrainement(_exemples_jouets())
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.5)
    modele = entrainer_foret(X_train, y_train)
    assert np.mean(modele.predict(X_test) == y_test) == 1.0
