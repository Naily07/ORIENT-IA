"""Tests de l'import de notre enquête (DATA-4, DATA-5, DATA-7).

Notre questionnaire demande tous les champs du profil : aucune valeur n'est
fabriquée, tout est `declaree`. La notion de provenance reste néanmoins portée
par le schéma — une collecte ultérieure pourrait dériver un champ (conversion
d'échelle, par exemple), et `jeu_evaluation()` doit continuer à filtrer.
"""

import pytest

from src.enquete import (
    CHAMPS_EXPLOITES_PAR_LE_MODELE,
    ReponseEnquete,
    charger_registres_collecte,
    charger_reponses,
    jeu_evaluation,
)
from src.enquete_import import (
    _resoudre_parcours,
    _sigle_depuis_choix,
    _valeurs_multiples,
)
from src.ml.archetypes import PARCOURS_CONNUS
from src.schemas import ProfilCandidat

# --- Résolution d'étiquette ---------------------------------------------------


def test_sigle_extrait_du_format_du_formulaire():
    assert _sigle_depuis_choix(
        "IGGLIA — Informatique de Gestion, Génie Logiciel et Intelligence Artificielle"
    ) == "IGGLIA"


def test_formation_hors_ispm_n_est_pas_une_etiquette():
    """Réponse réelle, mais hors du périmètre des 16 parcours du modèle."""
    assert _sigle_depuis_choix("Une formation hors ISPM") is None


@pytest.mark.parametrize(
    "brut,attendu",
    [("IGGLIA", "IGGLIA"), ("Igglia 2014-2019", "IGGLIA"), ("ESIIA - 2017", "ESIIA")],
)
def test_sigle_extrait_d_une_reponse_libre(brut, attendu):
    assert _resoudre_parcours(brut) == attendu


@pytest.mark.parametrize("brut", ["", "3 année", "Biotechnologie et Agronomie 2022"])
def test_reponse_non_resolvable_retourne_none(brut):
    """Le dernier cas désigne une *mention*, qui regroupe plusieurs parcours :
    la rattacher à l'un d'eux inventerait l'étiquette."""
    assert _resoudre_parcours(brut) is None


# --- Lecture des cases à cocher -----------------------------------------------


def test_aucune_competence_n_est_pas_comptee_comme_un_trait():
    """« Aucune en particulier » est une absence déclarée : la compter
    gonflerait artificiellement l'exploitabilité d'un profil."""
    assert _valeurs_multiples("Aucune en particulier") == []
    assert _valeurs_multiples("Programmation, Aucune en particulier") == ["Programmation"]


def test_valeurs_multiples_ignore_les_entrees_vides():
    assert _valeurs_multiples("Programmation, , Statistiques") == [
        "Programmation", "Statistiques"
    ]


# --- Provenance et jeu d'évaluation -------------------------------------------


def _reponse(**overrides) -> ReponseEnquete:
    valeurs = {
        "id": "orientia_0001",
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


@pytest.mark.parametrize("champ", CHAMPS_EXPLOITES_PAR_LE_MODELE)
def test_chaque_champ_du_modele_est_un_champ_de_profil(champ):
    """Non-régression : un champ renommé dans `ProfilCandidat` sans être mis à
    jour ici ferait passer une contamination inaperçue."""
    assert champ in ProfilCandidat.model_fields


# --- Données réellement livrées ------------------------------------------------


def test_nos_reponses_sont_toutes_declarees_jamais_fabriquees():
    """Notre questionnaire demande tous les champs : aucune complétion n'est
    nécessaire, donc aucune contamination possible du jeu d'évaluation."""
    reponses = charger_reponses()
    if not reponses:
        pytest.skip("réponses non importées")

    for reponse in reponses:
        assert reponse.champs_generes == [], f"{reponse.id} contient un champ fabriqué"


def test_le_jeu_d_evaluation_livre_est_exploitable():
    from src.ml.features import analyser_couverture

    reponses = charger_reponses()
    if not reponses:
        pytest.skip("réponses non importées")

    evaluables = jeu_evaluation(reponses)
    assert evaluables, "le jeu d'évaluation ne doit pas être vide"
    for reponse in evaluables:
        assert reponse.parcours_declare in PARCOURS_CONNUS
        assert analyser_couverture(reponse.profil).exploitable


# --- Registre de collecte (DATA-5) --------------------------------------------


def test_le_registre_est_charge_et_documente_ses_limites():
    registres = charger_registres_collecte()
    assert "notre_enquete" in registres
    registre = registres["notre_enquete"]
    assert registre.limites, "le registre doit nommer ses limites (§5 du sujet)"
    assert registre.texte_consentement


def test_les_chiffres_du_registre_correspondent_aux_donnees():
    """Un registre qui annoncerait d'autres chiffres que les fichiers livrés
    serait pire qu'absent."""
    registres = charger_registres_collecte()
    reponses = charger_reponses()
    if not reponses or "notre_enquete" not in registres:
        pytest.skip("données ou registre absents")

    registre = registres["notre_enquete"]
    assert registre.reponses_recues == len(reponses)
    assert registre.reponses_retenues == len(jeu_evaluation(reponses))
