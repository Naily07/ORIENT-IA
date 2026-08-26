"""Tests de la construction et des requêtes du graphe de connaissances
(ONTO-2 à ONTO-5), sur les corpus jouets partagés de `corpus_jouet.py`."""

import pytest

from src.graphe import (
    chemin_competence_parcours_metier,
    construire_graphe,
    detecter_incoherences,
    id_noeud,
    prerequis_du_parcours,
    type_et_id,
)
from src.models import CorpusFormations
from tests.corpus_jouet import corpus_avec_incoherences, corpus_coherent


@pytest.fixture
def corpus_sain() -> CorpusFormations:
    return corpus_coherent()


@pytest.fixture
def corpus_casse() -> CorpusFormations:
    return corpus_avec_incoherences()


def _types_signales_pour(incoherences: list[dict], competence: str) -> set[str]:
    return {i["type"] for i in incoherences if i.get("competence") == competence}


# --- id_noeud / type_et_id -----------------------------------------------


def test_id_noeud_et_type_et_id_sont_inverses():
    assert type_et_id(id_noeud("Parcours", "IGGLIA")) == ("Parcours", "IGGLIA")


def test_type_et_id_supporte_un_identifiant_contenant_un_deux_points():
    assert type_et_id(id_noeud("Parcours", "A:B")) == ("Parcours", "A:B")


# --- construire_graphe (ONTO-2) ------------------------------------------


def test_construire_graphe_relie_parcours_a_ses_entites(corpus_sain):
    graphe = construire_graphe(corpus_sain)
    noeud = id_noeud("Parcours", "IGGLIA")
    assert graphe.nodes[noeud]["nom"].startswith("Informatique")
    assert graphe[noeud][id_noeud("Matiere", "MAT-INFO")]["relation"] == "enseigne"
    assert graphe[noeud][id_noeud("Competence", "COMP-PROG")]["relation"] == "developpe"
    assert graphe[noeud][id_noeud("Metier", "METIER-DEV")]["relation"] == "prepareA"
    assert graphe[noeud][id_noeud("Prerequis", "PREREQ-SCIENTIFIQUE")]["relation"] == "necessite"


def test_construire_graphe_relie_parcours_a_sa_mention(corpus_sain):
    """La mention était absente du graphe tant qu'aucune relation du schéma ne
    la couvrait — `Parcours appartientA Mention` la rend représentable."""
    graphe = construire_graphe(corpus_sain)
    arete = graphe[id_noeud("Parcours", "IGGLIA")][id_noeud("Mention", "MENTION-INFO")]
    assert arete["relation"] == "appartientA"


def test_construire_graphe_relie_competence_a_metier(corpus_sain):
    graphe = construire_graphe(corpus_sain)
    arete = graphe[id_noeud("Competence", "COMP-PROG")][id_noeud("Metier", "METIER-DEV")]
    assert arete["relation"] == "estRequisePour"


def test_construire_graphe_ignore_silencieusement_une_reference_orpheline(corpus_casse):
    graphe = construire_graphe(corpus_casse)
    assert not graphe.has_node(id_noeud("Prerequis", "PREREQ-INEXISTANT"))
    assert prerequis_du_parcours(graphe, "TEH") == []


def test_construire_graphe_sur_corpus_vide_ne_plante_pas():
    assert len(construire_graphe(CorpusFormations()).nodes) == 0


# --- prerequis_du_parcours (ONTO-3) ---------------------------------------


def test_prerequis_du_parcours_via_graphe(corpus_sain):
    graphe = construire_graphe(corpus_sain)
    assert prerequis_du_parcours(graphe, "IGGLIA") == ["Baccalauréat série C, D, S"]


def test_prerequis_du_parcours_inconnu_donne_liste_vide(corpus_sain):
    graphe = construire_graphe(corpus_sain)
    assert prerequis_du_parcours(graphe, "PARCOURS-INEXISTANT") == []


# --- detecter_incoherences (ONTO-4) ---------------------------------------


def test_aucune_incoherence_sur_corpus_vide():
    assert detecter_incoherences(CorpusFormations(), construire_graphe(CorpusFormations())) == []


def test_corpus_coherent_ne_signale_que_la_donnee_manquante(corpus_sain):
    """Le corpus sain n'a qu'un défaut : TEH sans débouché, qui est une donnée
    non collectée et doit être marquée comme telle, pas comme une contradiction."""
    incoherences = detecter_incoherences(corpus_sain, construire_graphe(corpus_sain))
    assert [i["type"] for i in incoherences] == ["parcours_sans_debouche"]
    assert incoherences[0]["parcours"] == "TEH"
    assert incoherences[0]["donnee_manquante"] is True


def test_detecte_la_reference_orpheline_d_un_parcours(corpus_casse):
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    assert any(
        i["type"] == "reference_orpheline"
        and i["entite"] == "TEH"
        and i["champ"] == "prerequis"
        and i["id_reference"] == "PREREQ-INEXISTANT"
        for i in incoherences
    )


def test_detecte_la_reference_orpheline_de_metiers_requis(corpus_casse):
    """`Competence.metiers_requis` n'était couvert par aucun contrôle : l'arête
    était supprimée en silence et rien ne le signalait."""
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    assert any(
        i["type"] == "reference_orpheline"
        and i["entite"] == "COMP-ORPHELINE"
        and i["type_entite"] == "Competence"
        and i["champ"] == "metiers_requis"
        and i["id_reference"] == "METIER-X"
        for i in incoherences
    )


def test_detecte_la_reference_orpheline_d_une_mention(corpus_casse):
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    assert any(
        i["type"] == "reference_orpheline"
        and i["entite"] == "CAA"
        and i["champ"] == "mention_id"
        and i["id_reference"] == "MENTION-AFFAIRES"
        for i in incoherences
    )


def test_detecte_parcours_sans_debouche(corpus_casse):
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    sans_debouche = {i["parcours"] for i in incoherences if i["type"] == "parcours_sans_debouche"}
    assert sans_debouche == {"TEH", "CAA"}


def test_detecte_competence_requise_sans_parcours(corpus_casse):
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    assert "competence_requise_sans_parcours" in _types_signales_pour(incoherences, "COMP-ISOLEE")


def test_detecte_competence_requise_sans_prerequis_verifiable(corpus_casse):
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    assert "competence_requise_sans_prerequis_verifiable" in _types_signales_pour(
        incoherences, "COMP-NEGO"
    )


def test_competence_developpee_par_plusieurs_parcours_dont_un_avec_prerequis(corpus_casse):
    """Non-régression : COMP-MIXTE est développée par IGGLIA (prérequis connus)
    et par CAA (aucun). L'accès reste vérifiable via IGGLIA, donc rien ne doit
    être signalé — un `if sans_prerequis:` au lieu d'un `if len(...) == len(...)`
    inventait ici une incohérence que l'agent relayait à l'étudiant."""
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    assert _types_signales_pour(incoherences, "COMP-MIXTE") == set()


def test_competence_avec_parcours_et_prerequis_n_est_pas_signalee(corpus_casse):
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    assert _types_signales_pour(incoherences, "COMP-PROG") == set()


def test_les_messages_citent_les_noms_de_metier_pas_leurs_identifiants(corpus_casse):
    """L'identifiant interne ne doit pas remonter jusqu'au LLM puis à l'étudiant."""
    incoherences = detecter_incoherences(corpus_casse, construire_graphe(corpus_casse))
    message = next(
        i["message"]
        for i in incoherences
        if i.get("competence") == "COMP-NEGO"
        and i["type"] == "competence_requise_sans_prerequis_verifiable"
    )
    assert "Commercial" in message
    assert "METIER-COMMERCIAL" not in message


# --- chemin_competence_parcours_metier (ONTO-5) ---------------------------


def test_chemin_competence_parcours_metier(corpus_sain):
    graphe = construire_graphe(corpus_sain)
    assert chemin_competence_parcours_metier(graphe, "IGGLIA") == [
        {
            "parcours": "IGGLIA",
            "competence": "programmation",
            "metier": "Développeur logiciel",
            "chemin": [
                id_noeud("Parcours", "IGGLIA"),
                id_noeud("Competence", "COMP-PROG"),
                id_noeud("Metier", "METIER-DEV"),
            ],
        }
    ]


def test_chemin_vide_si_parcours_ne_developpe_aucune_competence_requise(corpus_sain):
    graphe = construire_graphe(corpus_sain)
    assert chemin_competence_parcours_metier(graphe, "TEH") == []


def test_chemin_vide_si_parcours_inconnu(corpus_sain):
    graphe = construire_graphe(corpus_sain)
    assert chemin_competence_parcours_metier(graphe, "PARCOURS-INEXISTANT") == []
