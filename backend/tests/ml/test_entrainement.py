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


# --- Réparation du classement quand la calibration devient muette -------------


class _CalibrationPlate:
    """Calibrateur qui renvoie la même valeur pour toutes les classes.

    Reproduit le défaut réel : chaque calibrateur isotonique un-contre-tous
    projette le score brut sur une marche, et quand les scores d'un profil
    tombent tous dans la même bande basse, les 16 renvoient la même valeur.
    """

    classes_ = np.array(["AEE", "DTJA", "IGGLIA"])

    def predict_proba(self, X):
        return np.full((len(X), 3), 1 / 3)

    def predict(self, X):
        return np.array([self.classes_[0]] * len(X))


class _SecoursOrdonne:
    """Modèle brut qui, lui, sait départager : IGGLIA nettement en tête."""

    classes_ = _CalibrationPlate.classes_

    def predict_proba(self, X):
        return np.tile(np.array([0.1, 0.2, 0.7]), (len(X), 1))


def test_une_calibration_plate_est_reparee_par_le_modele_brut():
    """Sans réparation, l'`argmax` d'une distribution uniforme tombe sur le
    premier parcours par ordre alphabétique — AEE pour tout le monde, quel que
    soit le profil. Mesuré sur 4 réponses d'enquête réelles sur 14."""
    from src.ml.entrainement import ModeleBorne

    modele = ModeleBorne(_CalibrationPlate(), n_entrainement=600, secours=_SecoursOrdonne())
    probabilites = modele.predict_proba(np.zeros((2, 4)))

    assert list(modele.classes_[probabilites.argmax(axis=1)]) == ["IGGLIA", "IGGLIA"]


def test_la_reparation_ne_touche_pas_une_calibration_informative():
    """La réparation ne doit s'appliquer qu'au cas dégénéré : ailleurs, c'est
    la calibration mesurée (ECE 0,033 contre 0,120 brut) qui fait foi."""
    X, y = preparer_jeu_entrainement()
    X_train, X_test, y_train, _ = separer_train_test(X, y)
    modele = entrainer_baseline_calibree(X_train, y_train)

    brutes = modele._modele.predict_proba(X_test)

    assert not modele._rangs_perdus(brutes).any()


def test_predict_suit_les_probabilites_apres_reparation():
    """Déléguer `predict` au modèle calibré servirait le parcours alphabétique
    que la réparation vient d'écarter : le système afficherait des scores en
    contradiction avec sa propre décision."""
    from src.ml.entrainement import ModeleBorne

    modele = ModeleBorne(_CalibrationPlate(), n_entrainement=600, secours=_SecoursOrdonne())
    X = np.zeros((3, 4))

    attendu = modele.classes_[modele.predict_proba(X).argmax(axis=1)]
    assert list(modele.predict(X)) == list(attendu)


def test_aucun_profil_reel_ne_produit_de_distribution_uniforme():
    """Non-régression sur les données qui ont révélé le défaut : le jeu
    synthétique ne le reproduit pas (0/200), seules les vraies réponses le
    déclenchent."""
    import pytest

    from src.enquete import jeu_evaluation
    from src.ml.outils import analyser_profil

    jeu = jeu_evaluation()
    if not jeu:
        pytest.skip("réponses d'enquête non importées")

    for reponse in jeu:
        scores = [p.score_adequation for p in analyser_profil(reponse.profil).parcours_candidats]
        assert max(scores) - min(scores) > 1e-6, (
            f"{reponse.id} : distribution uniforme, le classement ne veut rien dire"
        )
