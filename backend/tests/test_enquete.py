"""Tests de l'import d'enquête et de la séparation réel / fabriqué.

Le comportement le plus important vérifié ici : **un champ fabriqué exclut
l'enregistrement du jeu d'évaluation.** Sans cette garantie, ML-7 mesurerait
la capacité du modèle à retrouver ses propres hypothèses sur des données
qu'on lui aurait soufflées.
"""

import pytest

from src.enquete import (
    CHAMPS_EXPLOITES_PAR_LE_MODELE,
    ReponseEnquete,
    charger_reponses,
    jeu_evaluation,
)
from src.enquete_import import (
    _resoudre_parcours,
    _series_admissibles,
    _tirer_serie_bac,
)
from src.ml.archetypes import PARCOURS_CONNUS
from src.schemas import ProfilCandidat

# --- Résolution d'étiquette ---------------------------------------------------


@pytest.mark.parametrize(
    "brut,attendu",
    [
        ("IGGLIA", "IGGLIA"),
        ("Igglia 2014-2019", "IGGLIA"),
        ("ESIIA - 2017", "ESIIA"),
        ("IGGLIA + 2023", "IGGLIA"),
        ("ISPM ESIIA Master (2010)", "ESIIA"),
    ],
)
def test_sigle_extrait_d_une_reponse_libre(brut, attendu):
    assert _resoudre_parcours(brut) == attendu


@pytest.mark.parametrize(
    "brut",
    [
        "",
        "3 année",
        "MISA+BACC+4",
        # Une *mention* n'est pas un parcours : elle en contient plusieurs.
        # La rattacher à l'un d'eux serait inventer l'étiquette.
        "Biotechnologie et Agronomie 2022",
        "Informatique et télécommunications 2025",
    ],
)
def test_reponse_non_resolvable_retourne_none(brut):
    assert _resoudre_parcours(brut) is None


# --- Génération de la série de bac, ancrée sur les prérequis officiels --------


def test_series_admissibles_suit_les_prerequis():
    prerequis = {"IGGLIA": ["Baccalauréat série C, D, S, ou série techniques industrielles"]}
    assert set(_series_admissibles("IGGLIA", prerequis)) == {"C", "D", "S"}


def test_toute_serie_ouvre_toutes_les_series():
    prerequis = {"TEH": ["Baccalauréat toute série"]}
    assert "A" in _series_admissibles("TEH", prerequis)


def test_serie_generee_est_toujours_admissible():
    """Une série fabriquée qui rendrait le candidat inadmissible à son propre
    parcours produirait un profil incohérent — et ferait échouer la règle
    d'admission (`src.admission`) sur des données qu'on a nous-mêmes créées."""
    import random

    prerequis = {"IGGLIA": ["Baccalauréat série C, D, S, ou série techniques industrielles"]}
    for graine in range(30):
        serie = _tirer_serie_bac("IGGLIA", prerequis, random.Random(graine))
        assert serie in {"C", "D", "S"}


def test_parcours_sans_prerequis_connu_ne_bloque_pas_la_generation():
    import random

    serie = _tirer_serie_bac("INCONNU", {}, random.Random(0))
    assert serie


# --- Séparation réel / fabriqué -----------------------------------------------


def _reponse(**overrides) -> ReponseEnquete:
    valeurs = {
        "id": "reponse_0001",
        "population": "etudiant",
        "parcours_declare": "IGGLIA",
        "profil": ProfilCandidat(matieres_preferees=["Mathématiques"]),
        "provenance": {"matieres_preferees": "declaree"},
        "utilisable_pour_evaluation": True,
    }
    valeurs.update(overrides)
    return ReponseEnquete(**valeurs)


def test_champs_generes_sont_listes():
    reponse = _reponse(
        provenance={"matieres_preferees": "declaree", "serie_bac": "generee"}
    )
    assert reponse.champs_generes == ["serie_bac"]


def test_jeu_evaluation_ne_garde_que_les_utilisables():
    reponses = [
        _reponse(id="a", utilisable_pour_evaluation=True),
        _reponse(id="b", utilisable_pour_evaluation=False),
    ]
    assert [r.id for r in jeu_evaluation(reponses)] == ["a"]


def test_serie_bac_generee_ne_sort_pas_du_jeu_d_evaluation():
    """`serie_bac` n'appartient pas à l'espace de features du modèle : la
    fabriquer ne contamine pas la mesure, contrairement aux traits issus des
    archétypes."""
    assert "serie_bac" not in CHAMPS_EXPLOITES_PAR_LE_MODELE


@pytest.mark.parametrize("champ", CHAMPS_EXPLOITES_PAR_LE_MODELE)
def test_chaque_champ_du_modele_est_un_champ_de_profil(champ):
    """Non-régression : un champ renommé dans `ProfilCandidat` sans être mis à
    jour ici ferait passer une contamination inaperçue."""
    assert champ in ProfilCandidat.model_fields


# --- Données réellement livrées ------------------------------------------------


def test_le_jeu_d_evaluation_livre_est_exploitable():
    """Sur les données réelles versionnées : l'étiquette et les traits doivent
    être déclarés, jamais fabriqués."""
    reponses = charger_reponses()
    if not reponses:
        pytest.skip("réponses d'enquête non importées")

    evaluables = jeu_evaluation(reponses)
    assert evaluables, "le jeu d'évaluation ne doit pas être vide"
    for reponse in evaluables:
        assert reponse.parcours_declare in PARCOURS_CONNUS
        assert not any(
            reponse.provenance.get(champ) == "generee"
            for champ in CHAMPS_EXPLOITES_PAR_LE_MODELE
        )


def test_les_profils_completes_sont_tous_exclus_de_l_evaluation():
    """Le fichier de démonstration ne doit jamais pouvoir servir à mesurer."""
    completes = charger_reponses("profils_completes.json")
    if not completes:
        pytest.skip("profils complétés non générés")
    assert jeu_evaluation(completes) == []
