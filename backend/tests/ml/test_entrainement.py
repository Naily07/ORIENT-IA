"""Tests du pipeline d'entraînement (ML-3, ML-4)."""

import numpy as np

from src.ml.entrainement import (
    entrainer_baseline,
    entrainer_baseline_calibree,
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


# --- Calibration (correctif : sous-confiance mesurée à −0,12) ------------------


def test_la_calibration_reduit_l_ecart_entre_confiance_et_exactitude():
    """Non-régression du correctif. La régression logistique brute réussissait
    99,5 % du temps en n'annonçant que 87,9 % de confiance : le score montré à
    un candidat ne correspondait à aucune fréquence réelle."""
    from src.ml.evaluation import evaluer_modele

    X, y = preparer_jeu_entrainement()
    X_train, X_test, y_train, y_test = separer_train_test(X, y)

    brut = evaluer_modele(entrainer_baseline(X_train, y_train), X_test, y_test)
    calibre = evaluer_modele(entrainer_baseline_calibree(X_train, y_train), X_test, y_test)

    assert calibre["calibration"]["ece"] < brut["calibration"]["ece"] / 2
    # L'exactitude ne doit pas être payée pour la calibration : l'isotonique
    # est monotone, elle ne change aucune décision.
    assert calibre["exactitude"] == brut["exactitude"]


def test_le_modele_calibre_ne_produit_jamais_une_certitude_absolue():
    """Une probabilité de 1 exactement n'est pas une mesure : c'est la dernière
    marche de l'isotonique sur une tâche presque séparable. 68 % des profils
    ressortaient à 100 % — inacceptable pour une recommandation d'orientation."""
    X, y = preparer_jeu_entrainement()
    X_train, X_test, y_train, _ = separer_train_test(X, y)
    probabilites = entrainer_baseline_calibree(X_train, y_train).predict_proba(X_test)

    assert probabilites.max() < 1.0
    assert probabilites.min() > 0.0


def test_les_probabilites_bornees_somment_toujours_a_un():
    """Un simple plancher par classe casserait cette propriété : avec 16
    classes, il forcerait 7,5 % de masse dans la queue et écraserait le sommet."""
    import numpy as np

    X, y = preparer_jeu_entrainement()
    X_train, X_test, y_train, _ = separer_train_test(X, y)
    probabilites = entrainer_baseline_calibree(X_train, y_train).predict_proba(X_test)

    assert np.allclose(probabilites.sum(axis=1), 1.0)


def test_le_bornage_ne_change_aucune_decision():
    """Le mélange avec l'uniforme est monotone et uniforme : il déplace les
    valeurs, jamais l'ordre des classes."""
    import numpy as np

    X, y = preparer_jeu_entrainement()
    X_train, X_test, y_train, _ = separer_train_test(X, y)
    modele = entrainer_baseline_calibree(X_train, y_train)
    predites = np.array(modele.classes_)[np.argmax(modele.predict_proba(X_test), axis=1)]

    assert list(predites) == list(modele.predict(X_test))
