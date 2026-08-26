"""Tests des outils de l'agent (AGT-2/AGT-3), sur un corpus jouet isolé."""

import pytest

from src import tools
from src.models import (
    Competence,
    CorpusFormations,
    Matiere,
    Mention,
    Metier,
    Parcours,
    Prerequis,
)
from src.schemas import ProfilCandidat


@pytest.fixture
def corpus():
    c = CorpusFormations(
        mentions=[
            Mention(
                id="MENTION-INFO",
                nom="Informatique et Télécommunications",
                niveau="Licence",
            )
        ],
        parcours=[
            Parcours(
                id="IGGLIA",
                nom="Informatique de Gestion, Génie Logiciel et Intelligence Artificielle",
                mention_id="MENTION-INFO",
                matieres=["MAT-INFO"],
                competences=["COMP-PROG"],
                prerequis=["PREREQ-SCIENTIFIQUE"],
                debouches=["METIER-DEV"],
            ),
            Parcours(
                id="TEH",
                nom="Tourisme et Hôtellerie",
                mention_id="MENTION-TOURISME",
                prerequis=["PREREQ-TOUTE-SERIE"],
            ),
        ],
        matieres=[Matiere(id="MAT-INFO", nom="informatique")],
        competences=[Competence(id="COMP-PROG", nom="programmation")],
        prerequis=[
            Prerequis(id="PREREQ-SCIENTIFIQUE", description="Baccalauréat série C, D, S"),
            Prerequis(id="PREREQ-TOUTE-SERIE", description="Baccalauréat toute série"),
        ],
        metiers=[Metier(id="METIER-DEV", nom="Développeur logiciel")],
    )
    tools.initialiser_corpus(c)
    yield c
    tools.initialiser_corpus()  # reset sur le corpus réel pour les autres tests


# --- rechercher_formation -----------------------------------------------------


def test_rechercher_formation_par_sigle(corpus):
    resultat = tools.rechercher_formation("IGGLIA")
    assert resultat["statut"] == "trouve"
    assert resultat["parcours"][0]["id"] == "IGGLIA"


def test_rechercher_formation_par_nom_de_mention(corpus):
    resultat = tools.rechercher_formation("informatique")
    assert resultat["statut"] == "trouve"
    assert any(m["id"] == "MENTION-INFO" for m in resultat["mentions"])


def test_rechercher_formation_aucun_resultat(corpus):
    resultat = tools.rechercher_formation("astrophysique")
    assert resultat["statut"] == "aucun_resultat"


# --- comparer_parcours ---------------------------------------------------


def test_comparer_parcours_retourne_les_deux_fiches(corpus):
    resultat = tools.comparer_parcours("IGGLIA", "TEH")
    assert resultat["statut"] == "trouve"
    assert resultat["parcours_a"]["id"] == "IGGLIA"
    assert resultat["parcours_b"]["id"] == "TEH"
    assert resultat["parcours_a"]["mention"]["id"] == "MENTION-INFO"


def test_comparer_parcours_note_l_incompletude(corpus):
    resultat = tools.comparer_parcours("IGGLIA", "TEH")
    assert resultat["parcours_a"]["note_completude"] is None  # a des matières/compétences/débouchés
    assert resultat["parcours_b"]["note_completude"] is not None  # TEH n'en a aucun


def test_comparer_parcours_introuvable(corpus):
    resultat = tools.comparer_parcours("IGGLIA", "PARCOURS-INEXISTANT")
    assert resultat["statut"] == "aucun_resultat"


# --- verifier_prerequis ---------------------------------------------------


def test_verifier_prerequis_sans_serie_bac_declaree(corpus):
    tools.definir_profil_courant(ProfilCandidat())
    resultat = tools.verifier_prerequis("IGGLIA")
    assert resultat["statut"] == "information_manquante"
    assert resultat["compatible"] is None


def test_verifier_prerequis_serie_compatible(corpus):
    tools.definir_profil_courant(ProfilCandidat(serie_bac="D"))
    resultat = tools.verifier_prerequis("IGGLIA")
    assert resultat["statut"] == "trouve"
    assert resultat["compatible"] is True


def test_verifier_prerequis_serie_incompatible(corpus):
    tools.definir_profil_courant(ProfilCandidat(serie_bac="L"))
    resultat = tools.verifier_prerequis("IGGLIA")
    assert resultat["compatible"] is False


def test_verifier_prerequis_toute_serie_toujours_compatible(corpus):
    tools.definir_profil_courant(ProfilCandidat(serie_bac="L"))
    resultat = tools.verifier_prerequis("TEH")
    assert resultat["compatible"] is True


def test_verifier_prerequis_parcours_introuvable(corpus):
    tools.definir_profil_courant(ProfilCandidat(serie_bac="D"))
    resultat = tools.verifier_prerequis("PARCOURS-INEXISTANT")
    assert resultat["statut"] == "aucun_resultat"


# --- rechercher_competences / identifier_debouches ---------------------------


def test_rechercher_competences_trouve_le_parcours(corpus):
    resultat = tools.rechercher_competences("programmation")
    assert resultat["statut"] == "trouve"
    assert "IGGLIA" in resultat["parcours"]


def test_rechercher_competences_inconnue(corpus):
    resultat = tools.rechercher_competences("comptabilite_inexistante")
    assert resultat["statut"] == "aucun_resultat"


def test_identifier_debouches_connu(corpus):
    resultat = tools.identifier_debouches("IGGLIA")
    assert resultat["statut"] == "trouve"
    assert "Développeur logiciel" in resultat["debouches"]


def test_identifier_debouches_information_manquante(corpus):
    resultat = tools.identifier_debouches("TEH")
    assert resultat["statut"] == "information_manquante"


# --- executer_outil (AGT-3) ---------------------------------------------------


def test_executer_outil_inconnu(corpus):
    resultat = tools.executer_outil("outil_inexistant", {}, "trace-1")
    assert resultat["statut"] == "erreur"


def test_executer_outil_parametres_manquants(corpus):
    resultat = tools.executer_outil("comparer_parcours", {"parcours_a": "IGGLIA"}, "trace-1")
    assert resultat["statut"] == "erreur"
    assert "parcours_b" in resultat["message"]


def test_executer_outil_capture_outil_indisponible_sans_planter(corpus, monkeypatch):
    monkeypatch.setattr(tools, "_corpus", None)
    resultat = tools.executer_outil("verifier_prerequis", {"parcours": "IGGLIA"}, "trace-1")
    assert resultat["statut"] == "erreur"
    assert "non initialisé" in resultat["message"]


def test_corpus_vide_donne_aucun_resultat_pas_un_crash(corpus):
    tools.initialiser_corpus(CorpusFormations())
    resultat = tools.executer_outil("verifier_prerequis", {"parcours": "IGGLIA"}, "trace-1")
    assert resultat["statut"] == "succes"
    assert resultat["resultat"]["statut"] == "aucun_resultat"


def test_executer_outil_succes(corpus):
    resultat = tools.executer_outil("rechercher_formation", {"mot_cle": "IGGLIA"}, "trace-1")
    assert resultat["statut"] == "succes"
    assert resultat["resultat"]["statut"] == "trouve"


# --- Outils ML (analyser_profil_ml, calculer_score_adequation, expliquer_recommandation) --
# Ces outils passent par src.ml.outils (modèle réel entraîné sur le jeu synthétique) :
# indépendants du corpus jouet ci-dessus, ils utilisent les 16 vrais parcours ISPM.


def test_analyser_profil_ml_retourne_une_analyse_serialisee():
    tools.definir_profil_courant(ProfilCandidat(matieres_preferees=["informatique"]))
    resultat = tools.analyser_profil_ml()
    assert "parcours_candidats" in resultat
    assert len(resultat["parcours_candidats"]) == 16


def test_calculer_score_adequation_retourne_un_score_entre_0_et_1():
    tools.definir_profil_courant(ProfilCandidat(matieres_preferees=["droit"]))
    resultat = tools.calculer_score_adequation("DTJA")
    assert 0.0 <= resultat["score_adequation"] <= 1.0


def test_expliquer_recommandation_retourne_des_points_forts():
    tools.definir_profil_courant(ProfilCandidat(competences_declarees=["programmation"]))
    resultat = tools.expliquer_recommandation("IGGLIA")
    assert "points_forts" in resultat
    assert isinstance(resultat["points_forts"], list)
