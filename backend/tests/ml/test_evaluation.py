"""Tests des métriques d'évaluation (ML-5, ML-6)."""

from src.ml.entrainement import entrainer_baseline, preparer_jeu_entrainement, separer_train_test
from src.ml.evaluation import (
    analyser_erreurs,
    evaluer_chemin_de_production,
    evaluer_modele,
    mesurer_stabilite,
    mesurer_stabilite_des_recommandations,
)
from src.ml.outils import analyser_profil
from src.schemas import ProfilCandidat


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


# --- Métriques de classement et de calibration (correctif d'audit) ------------


def _modele_a_trois_classes():
    X, y = preparer_jeu_entrainement(_jeu_a_trois_classes())
    X_train, X_test, y_train, y_test = separer_train_test(X, y, test_size=0.5)
    return entrainer_baseline(X_train, y_train), X_test, y_test


def test_les_metriques_de_classement_sont_presentes_et_bornees():
    """§14 exige « la qualité du classement ou de la recommandation » : le
    système propose des parcours ordonnés, l'exactitude au rang 1 n'en dit rien."""
    modele, X_test, y_test = _modele_a_trois_classes()
    r = evaluer_modele(modele, X_test, y_test)

    assert 0.0 <= r["mrr"] <= 1.0
    assert 0.0 <= r["ndcg_3"] <= 1.0
    assert r["rang_median_bonne_classe"] >= 1.0
    assert 0.0 <= r["pr_auc_macro"] <= 1.0


def test_mrr_vaut_un_quand_la_bonne_classe_est_toujours_en_tete():
    modele, X_test, y_test = _modele_a_trois_classes()
    r = evaluer_modele(modele, X_test, y_test)
    # Jeu trivialement séparable : la bonne classe doit être au rang 1 partout.
    assert r["exactitude"] == 1.0
    assert r["mrr"] == 1.0
    assert r["rang_median_bonne_classe"] == 1.0


def test_la_calibration_expose_ece_et_brier():
    """La séparation de confiance seule dit si la confiance discrimine, pas si
    le « 90 % » affiché à un candidat correspond à 90 % de réussite réelle."""
    modele, X_test, y_test = _modele_a_trois_classes()
    calibration = evaluer_modele(modele, X_test, y_test)["calibration"]

    assert 0.0 <= calibration["ece"] <= 1.0
    assert calibration["score_de_brier"] >= 0.0
    assert calibration["tranches"]
    assert all(0 <= t["exactitude"] <= 1 for t in calibration["tranches"])


def test_la_matrice_de_confusion_est_complete_et_serialisable():
    modele, X_test, y_test = _modele_a_trois_classes()
    matrice = evaluer_modele(modele, X_test, y_test)["matrice_confusion"]

    labels = matrice["labels"]
    assert len(matrice["matrice"]) == len(labels)
    assert all(len(ligne) == len(labels) for ligne in matrice["matrice"])
    # Le total des cases doit couvrir tout le jeu de test, sans perte.
    assert sum(sum(ligne) for ligne in matrice["matrice"]) == len(y_test)


# --- Chemin de production et stabilité des recommandations --------------------


def test_evaluer_chemin_de_production_mesure_le_classement_reellement_servi():
    """§8/§14 : mesurer le seul estimateur reviendrait à publier les chiffres
    d'un modèle que personne n'exécute."""
    exemples = [
        {
            "profil": ProfilCandidat(
                matieres_preferees=["informatique"], competences_declarees=["programmation"]
            ),
            "parcours_id": "IGGLIA",
        }
    ]
    resultat = evaluer_chemin_de_production(exemples, analyser_profil)

    assert resultat["effectif"] == 1
    assert 0.0 <= resultat["top_1"] <= 1.0
    assert resultat["parcours_absents_du_classement"] == 0


def test_la_stabilite_des_recommandations_mesure_bien_une_perturbation():
    """À distinguer de `mesurer_stabilite` : ici on retire un trait déclaré et
    on regarde si la recommandation bouge — la propriété qui compte pour un
    candidat."""
    exemples = [
        {
            "profil": ProfilCandidat(
                matieres_preferees=["informatique", "mathematiques", "gestion"],
                competences_declarees=["programmation", "algorithmique"],
            ),
            "parcours_id": "IGGLIA",
        }
    ]
    resultat = mesurer_stabilite_des_recommandations(exemples, analyser_profil)

    assert resultat["profils_compares"] == 1
    assert 0.0 <= resultat["top_1_inchange"] <= 1.0
    # Métrique de référence : la stabilité de ce qui est réellement présenté.
    assert 0.0 <= resultat["selection_presentee_inchangee"] <= 1.0
    assert 0.0 <= resultat["top_3_fixe_inchange"] <= 1.0


def test_un_profil_non_perturbable_est_signale_pas_compte_a_tort():
    exemples = [
        {"profil": ProfilCandidat(matieres_preferees=["informatique"]), "parcours_id": "IGGLIA"}
    ]
    assert "avertissement" in mesurer_stabilite_des_recommandations(exemples, analyser_profil)
