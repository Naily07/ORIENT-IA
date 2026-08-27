"""Tests de l'analyse exploratoire (ML-1).

Le calcul de l'EDA vit dans `src/ml/exploration.py` plutôt que dans le notebook
précisément pour être testable ici : un notebook n'est pas rejouable en CI et
dépend de l'ordre d'exécution de ses cellules.
"""

from src.ml.exploration import (
    analyser,
    completude_des_champs,
    correlations_traits_classe,
    distribution_des_classes,
    pouvoir_discriminant_par_champ,
    traits_les_plus_frequents,
)


def _jeu_equilibre() -> list[dict]:
    exemples = []
    for _ in range(10):
        exemples.append(
            {
                "profil": {
                    "matieres_preferees": ["informatique", "mathematiques"],
                    "competences_declarees": ["programmation"],
                    "resultats_scolaires": {"informatique": 16.0},
                    "environnement_travail_recherche": "bureau_informatique",
                },
                "parcours_id": "IGGLIA",
            }
        )
        exemples.append(
            {
                "profil": {
                    "matieres_preferees": ["droit", "mathematiques"],
                    "competences_declarees": ["redaction"],
                    "environnement_travail_recherche": "bureau_cabinet",
                },
                "parcours_id": "DTJA",
            }
        )
    return exemples


# --- Distribution ------------------------------------------------------------


def test_distribution_detecte_un_jeu_equilibre():
    d = distribution_des_classes(_jeu_equilibre())
    assert d["nombre_de_classes"] == 2
    assert d["effectif_total"] == 20
    assert d["equilibre"] is True


def test_distribution_detecte_un_desequilibre():
    exemples = _jeu_equilibre() + [
        {"profil": {"matieres_preferees": ["biologie"]}, "parcours_id": "PIP"}
    ]
    d = distribution_des_classes(exemples)
    assert d["equilibre"] is False
    assert d["effectif_min"] == 1


def test_distribution_sur_jeu_vide_ne_plante_pas():
    d = distribution_des_classes([])
    assert d["effectif_total"] == 0
    assert d["nombre_de_classes"] == 0


# --- Complétude ---------------------------------------------------------------


def test_completude_signale_un_champ_jamais_renseigne():
    """Un champ à 0 % n'est pas une statistique anodine : c'est une capacité que
    le modèle ne peut pas apprendre. `serie_bac` et `activites_projets` sont
    dans ce cas sur le jeu synthétique réel."""
    c = completude_des_champs(_jeu_equilibre())
    assert c["matieres_preferees"]["taux"] == 1.0
    assert c["serie_bac"]["taux"] == 0.0
    assert c["activites_projets"]["taux"] == 0.0


def test_completude_compte_les_profils_pas_les_valeurs():
    c = completude_des_champs(_jeu_equilibre())
    assert c["matieres_preferees"]["renseigne"] == 20


# --- Traits fréquents ---------------------------------------------------------


def test_traits_les_plus_frequents_ordonne_par_occurrence():
    traits = traits_les_plus_frequents(_jeu_equilibre())["matieres_preferees"]
    assert traits[0]["trait"] == "mathematiques"  # présent dans les deux classes
    assert traits[0]["occurrences"] == 20


# --- Détection de fuite -------------------------------------------------------


def test_une_variable_qui_identifie_seule_la_classe_est_signalee():
    """Régression documentée : `environnement_travail_recherche` déterministe
    par archétype donnait 100 % d'exactitude à n'importe quel modèle. Ce
    contrôle existe pour que le prochain cas se voie sans ré-entraînement."""
    fuite = [
        {"profil": {"environnement_travail_recherche": "labo_A"}, "parcours_id": "A"},
        {"profil": {"environnement_travail_recherche": "labo_B"}, "parcours_id": "B"},
    ]
    stats = pouvoir_discriminant_par_champ(fuite)["environnement_travail_recherche"]
    assert stats["part_traits_exclusifs"] == 1.0
    assert stats["classes_moyennes_par_trait"] == 1.0


def test_un_trait_partage_entre_classes_n_est_pas_signale_comme_fuite():
    stats = pouvoir_discriminant_par_champ(_jeu_equilibre())["matieres_preferees"]
    # « mathematiques » est dans les deux classes, « informatique »/« droit » dans une seule.
    assert stats["part_traits_exclusifs"] < 1.0
    assert stats["classes_moyennes_par_trait"] > 1.0


def test_pouvoir_discriminant_sur_champ_vide_ne_plante_pas():
    exemples = [{"profil": {"matieres_preferees": []}, "parcours_id": "A"}]
    assert pouvoir_discriminant_par_champ(exemples)["matieres_preferees"]["traits"] == 0


# --- Corrélations -------------------------------------------------------------


def test_les_correlations_sont_bornees_et_triees_par_amplitude():
    correlations = correlations_traits_classe(_jeu_equilibre(), top_n=5)
    assert correlations
    assert all(-1.0 <= c["correlation"] <= 1.0 for c in correlations)
    amplitudes = [abs(c["correlation"]) for c in correlations]
    assert amplitudes == sorted(amplitudes, reverse=True)


def test_un_trait_propre_a_une_classe_est_fortement_correle():
    correlations = correlations_traits_classe(_jeu_equilibre(), top_n=40)
    droit = next(
        c for c in correlations if c["trait"] == "matiere:droit" and c["parcours"] == "DTJA"
    )
    assert droit["correlation"] > 0.9


# --- Rapport complet ----------------------------------------------------------


def test_analyser_produit_un_rapport_serialisable():
    import json

    rapport = analyser(_jeu_equilibre())
    assert set(rapport) == {
        "distribution_des_classes",
        "completude_des_champs",
        "traits_les_plus_frequents",
        "pouvoir_discriminant_par_champ",
        "correlations_traits_classe",
        "dimension_de_l_espace_de_features",
    }
    json.dumps(rapport)  # doit passer sans TypeError (livrable JSON)
