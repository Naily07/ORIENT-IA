"""Test de non-régression sur le vrai jeu de données synthétique livré avec
le projet (`backend/data/ml/profils_synthetiques.json`, généré par
`donnees_synthetiques.py`).

Sert de garde-fou : si un futur changement du générateur ou des features
dégrade silencieusement la qualité du jeu (ex. réintroduit la fuite sur
`environnement_travail_recherche` documentée dans `donnees_synthetiques.py`),
ce test l'attrape sous forme d'un score trop haut ou trop bas, plutôt que de
laisser passer une régression invisible en CI.
"""

from src.ml.entrainement import entrainer_baseline, preparer_jeu_entrainement, separer_train_test
from src.ml.evaluation import evaluer_modele


def test_le_jeu_de_donnees_reel_n_est_pas_vide():
    X, y = preparer_jeu_entrainement()
    assert X.shape[0] > 0
    assert len(set(y)) == 16  # les 16 parcours réels de l'ISPM


def test_la_baseline_generalise_sans_etre_triviale():
    """Ni trop parfaite (jeu trivialement séparable — la fuite déjà
    rencontrée sur `environnement_travail_recherche`), ni trop mauvaise
    (générateur ou features cassés)."""
    X, y = preparer_jeu_entrainement()
    X_train, X_test, y_train, y_test = separer_train_test(X, y)
    modele = entrainer_baseline(X_train, y_train)
    resultats = evaluer_modele(modele, X_test, y_test)

    assert 0.85 <= resultats["exactitude"] < 1.0
    assert resultats["top_3_accuracy"] >= resultats["exactitude"]
