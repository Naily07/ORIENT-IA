"""Tests des outils de l'agent (AGT-2/AGT-3), sur un corpus jouet isolé."""

import pytest

from src import tools
from src.models import CorpusFormations
from src.schemas import ProfilCandidat
from tests.corpus_jouet import corpus_coherent


@pytest.fixture
def corpus():
    c = corpus_coherent()
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


# --- Traçabilité des sources des outils structurés (AGT-6) -------------------


def test_fiche_parcours_remonte_le_source_id(corpus):
    """EVAL-17 : un outil structuré doit pouvoir citer sa source (`Parcours.source_id`,
    registre DATA-2), pas seulement le RAG."""
    resultat = tools.comparer_parcours("IGGLIA", "TEH")
    assert resultat["parcours_a"]["source_id"] == "FORM-IGGLIA-JOUET"
    assert resultat["parcours_b"]["source_id"] is None  # TEH n'en a pas dans le corpus jouet


def test_verifier_prerequis_remonte_le_source_id(corpus):
    tools.definir_profil_courant(ProfilCandidat(serie_bac="D"))
    resultat = tools.verifier_prerequis("IGGLIA")
    assert resultat["source_id"] == "FORM-IGGLIA-JOUET"


def test_identifier_debouches_remonte_le_source_id(corpus):
    resultat = tools.identifier_debouches("IGGLIA")
    assert resultat["source_id"] == "FORM-IGGLIA-JOUET"


def test_rechercher_formation_remonte_le_source_id(corpus):
    resultat = tools.rechercher_formation("IGGLIA")
    assert resultat["parcours"][0]["source_id"] == "FORM-IGGLIA-JOUET"


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


def test_verifier_prerequis_serie_bac_blanche_ne_confirme_rien(corpus):
    """Non-régression : une série faite d'espaces passait le test de vérité,
    puis `.strip()` vidait le motif et la regex `\\b\\b` matchait tout — l'outil
    confirmait l'admissibilité au lieu de réclamer l'information."""
    tools.definir_profil_courant(ProfilCandidat(serie_bac="   "))
    resultat = tools.verifier_prerequis("IGGLIA")
    assert resultat["statut"] == "information_manquante"
    assert resultat["compatible"] is None


def test_verifier_prerequis_et_comparer_parcours_donnent_les_memes_prerequis(corpus):
    """Une seule source de vérité : les deux outils lisent le graphe."""
    tools.definir_profil_courant(ProfilCandidat(serie_bac="D"))
    via_verification = tools.verifier_prerequis("IGGLIA")["prerequis"]
    via_comparaison = tools.comparer_parcours("IGGLIA", "TEH")["parcours_a"]["prerequis"]
    assert via_verification == via_comparaison == ["Baccalauréat série C, D, S"]


# --- detecter_incoherences (ONTO-4) -------------------------------------


def test_detecter_incoherences_signale_teh_sans_debouche(corpus):
    resultat = tools.detecter_incoherences()
    assert resultat["statut"] == "trouve"
    parcours_signales = {
        i["parcours"] for i in resultat["incoherences"] if i["type"] == "parcours_sans_debouche"
    }
    assert "TEH" in parcours_signales
    assert "IGGLIA" not in parcours_signales  # IGGLIA a un débouché renseigné


def test_detecter_incoherences_corpus_sain_reste_un_succes(corpus):
    """Non-régression : `aucun_resultat` signifie « je n'ai pas pu répondre »
    partout ailleurs dans tools.py. Un corpus sain est un résultat positif, et
    doit le rester pour que l'agent ne rapporte pas l'inverse."""
    tools.initialiser_corpus(CorpusFormations())
    resultat = tools.detecter_incoherences()
    assert resultat["statut"] == "trouve"
    assert resultat["nombre"] == 0
    assert "Aucune incohérence" in resultat["message"]


def test_detecter_incoherences_distingue_donnee_manquante_et_contradiction(corpus):
    """Le champ `donnee_manquante` et le message doivent empêcher l'agent de
    présenter un chantier de collecte (DATA-1) comme un défaut de fiabilité."""
    resultat = tools.detecter_incoherences()
    manquantes = [i for i in resultat["incoherences"] if i["type"] == "parcours_sans_debouche"]
    assert all(i["donnee_manquante"] is True for i in manquantes)
    assert "pas encore collectée" in resultat["message"]


def test_executer_outil_detecter_incoherences(corpus):
    resultat = tools.executer_outil("detecter_incoherences", {}, "trace-1")
    assert resultat["statut"] == "succes"
    assert resultat["resultat"]["nombre"] >= 1


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


def test_initialiser_corpus_ne_desynchronise_pas_corpus_et_graphe(corpus, monkeypatch):
    """Non-régression : `_corpus` était affecté avant la construction du graphe.
    Si celle-ci échouait, un corpus neuf se retrouvait face à un graphe périmé
    et `verifier_prerequis` croisait les deux pour répondre sur l'admissibilité."""
    monkeypatch.setattr(
        tools, "_construire_graphe", lambda _: (_ for _ in ()).throw(ValueError("boom"))
    )
    autre_corpus = CorpusFormations()

    with pytest.raises(ValueError):
        tools.initialiser_corpus(autre_corpus)

    # Les deux globales ont gardé leur valeur précédente, cohérente entre elles.
    assert tools._corpus is corpus
    tools.definir_profil_courant(ProfilCandidat(serie_bac="D"))
    assert tools.verifier_prerequis("IGGLIA")["prerequis"] == ["Baccalauréat série C, D, S"]


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


def test_expliquer_recommandation_retourne_des_points_forts(corpus):
    tools.definir_profil_courant(ProfilCandidat(competences_declarees=["programmation"]))
    resultat = tools.expliquer_recommandation("IGGLIA")
    assert "points_forts" in resultat
    assert isinstance(resultat["points_forts"], list)


def test_expliquer_recommandation_remonte_le_source_id(corpus):
    tools.definir_profil_courant(ProfilCandidat(competences_declarees=["programmation"]))
    resultat = tools.expliquer_recommandation("IGGLIA")
    assert resultat["source_id"] == "FORM-IGGLIA-JOUET"


def test_expliquer_recommandation_ajoute_le_raisonnement_du_graphe(corpus):
    """ONTO-5 : sur le corpus jouet (COMP-PROG développée par IGGLIA et
    requise pour METIER-DEV), le chemin Compétence → Métier doit apparaître."""
    tools.definir_profil_courant(ProfilCandidat(competences_declarees=["programmation"]))
    resultat = tools.expliquer_recommandation("IGGLIA")
    assert resultat["raisonnement_graphe"] == [
        {
            "parcours": "IGGLIA",
            "competence": "programmation",
            "metier": "Développeur logiciel",
            "chemin": ["Parcours:IGGLIA", "Competence:COMP-PROG", "Metier:METIER-DEV"],
        }
    ]


@pytest.mark.parametrize("saisie", ["IGGLIA", "igglia", "Informatique de Gestion"])
def test_expliquer_recommandation_resout_sigle_casse_et_nom(corpus, saisie):
    """Non-régression : sans résolution, « igglia » ou un nom de parcours
    donnaient un score de 0.0 et un raisonnement vide — indiscernables, côté
    agent, d'une absence réelle de données. Les autres outils résolvant déjà
    les noms, le LLM en passe."""
    tools.definir_profil_courant(ProfilCandidat(competences_declarees=["programmation"]))
    resultat = tools.expliquer_recommandation(saisie)
    assert resultat["statut"] == "trouve"
    assert resultat["parcours"] == "IGGLIA"
    assert len(resultat["raisonnement_graphe"]) == 1


def test_expliquer_recommandation_parcours_introuvable(corpus):
    tools.definir_profil_courant(ProfilCandidat())
    resultat = tools.expliquer_recommandation("PARCOURS-INEXISTANT")
    assert resultat["statut"] == "aucun_resultat"


def test_expliquer_recommandation_degrade_sans_graphe_initialise(corpus, monkeypatch):
    """L'explication ML reste utilisable même si le graphe n'a jamais été
    construit (ex. import isolé, avant tout appel à initialiser_corpus)."""
    monkeypatch.setattr(tools, "_graphe", None)
    tools.definir_profil_courant(ProfilCandidat(competences_declarees=["programmation"]))
    resultat = tools.expliquer_recommandation("IGGLIA")
    assert resultat["raisonnement_graphe"] == []
    assert resultat["score_adequation"] >= 0.0


def test_le_graphe_d_admission_suit_le_corpus_courant(corpus):
    """Non-régression d'alignement : `ml.outils` reconstruisait son propre
    graphe **depuis le disque** et le mettait en cache. Basculer le corpus ne
    rebâtissait que celui de `tools` — mesuré : 2 nœuds d'un côté, 517 de
    l'autre, pour un seul corpus censément courant. Les règles d'admission
    jugeaient donc d'après un corpus que plus personne ne servait, et
    silencieusement (un parcours inconnu n'a pas de prérequis, donc aucune
    rétrogradation) : un test croyait les exercer alors qu'elles ne
    s'appliquaient pas."""
    from src.ml import outils

    assert outils._graphe_courant() is tools._graphe

    tools.initialiser_corpus(CorpusFormations())
    assert outils._graphe_courant() is tools._graphe
    assert len(outils._graphe_courant().nodes) == 0
