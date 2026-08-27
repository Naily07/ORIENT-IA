"""Tests du volet hybride : apprentissage statistique + règles d'admission.

Le défaut corrigé, mesuré sur le corpus réel : le modèle ML ne voit jamais la
série de baccalauréat (absente de l'espace de features), il peut donc classer
en tête un parcours auquel le candidat n'est pas admissible.
"""

import networkx as nx
import pytest

from src.admission import serie_satisfait_prerequis
from src.graphe import construire_graphe
from src.ml.hybride import appliquer_regles_admission, evaluer_admissibilite
from src.ml.outils import classer_parcours
from src.models import charger_corpus_formations
from src.schemas import ProfilCandidat, RecommandationParcours

# --- La règle d'admission partagée (src/admission.py) ------------------------

SCIENTIFIQUE = ["Baccalauréat série C, D, S, ou série techniques industrielles"]
TOUTE_SERIE = ["Baccalauréat toute série"]


def test_serie_listee_est_admissible():
    assert serie_satisfait_prerequis("D", SCIENTIFIQUE) is True


def test_serie_absente_est_refusee():
    assert serie_satisfait_prerequis("A", SCIENTIFIQUE) is False


def test_toute_serie_accepte_n_importe_quelle_serie():
    assert serie_satisfait_prerequis("A", TOUTE_SERIE) is True


def test_lettre_isolee_ne_matche_pas_au_milieu_d_un_mot():
    """« L » ne doit pas correspondre au « l » de « baccaLauréat » — le piège
    déjà rencontré et corrigé, verrouillé ici contre toute régression."""
    assert serie_satisfait_prerequis("L", SCIENTIFIQUE) is False


def test_serie_non_declaree_est_indeterminable():
    """`None` n'est pas un refus : c'est une information manquante."""
    assert serie_satisfait_prerequis(None, SCIENTIFIQUE) is None
    assert serie_satisfait_prerequis("", SCIENTIFIQUE) is None


def test_serie_composee_uniquement_d_espaces_est_indeterminable():
    """Sans le `.strip()` en amont, le motif deviendrait vide et `\\b\\b`
    correspondrait à n'importe quel texte — l'admissibilité serait confirmée
    à tort."""
    assert serie_satisfait_prerequis("   ", SCIENTIFIQUE) is None


def test_aucun_prerequis_connu_est_indeterminable():
    assert serie_satisfait_prerequis("D", []) is None


# --- Rétrogradation des parcours inadmissibles -------------------------------


@pytest.fixture
def graphe():
    return construire_graphe(charger_corpus_formations())


def _candidat(parcours: str, score: float) -> RecommandationParcours:
    return RecommandationParcours(
        parcours=parcours, score_adequation=score, justification="score du modèle"
    )


def test_parcours_inadmissible_passe_derriere_les_accessibles(graphe):
    """IGGLIA exige un Bac C/D/S ; DTJA accepte toute série. Un candidat Bac A
    doit voir DTJA remonter, quel que soit le score du modèle."""
    candidats = [_candidat("IGGLIA", 0.54), _candidat("DTJA", 0.03)]
    profil = ProfilCandidat(serie_bac="A")

    resultat = appliquer_regles_admission(candidats, profil, graphe)

    assert [c.parcours for c in resultat] == ["DTJA", "IGGLIA"]


def test_les_scores_ne_sont_jamais_modifies(graphe):
    """Seul l'ordre change : un score affiché doit rester celui que le modèle
    a réellement produit (§6, distinguer résultat du modèle et règle)."""
    candidats = [_candidat("IGGLIA", 0.54), _candidat("DTJA", 0.03)]
    resultat = appliquer_regles_admission(candidats, ProfilCandidat(serie_bac="A"), graphe)

    scores = {c.parcours: c.score_adequation for c in resultat}
    assert scores == {"IGGLIA": 0.54, "DTJA": 0.03}


def test_parcours_ecarte_est_annote_et_non_masque(graphe):
    """Rétrograder, pas cacher : le candidat garde l'information et la raison."""
    candidats = [_candidat("IGGLIA", 0.54), _candidat("DTJA", 0.03)]
    resultat = appliquer_regles_admission(candidats, ProfilCandidat(serie_bac="A"), graphe)

    igglia = next(c for c in resultat if c.parcours == "IGGLIA")
    assert len(resultat) == 2  # rien n'a disparu
    assert "[Règle d'admission]" in igglia.justification
    assert "administration" in igglia.justification


def test_candidat_admissible_conserve_l_ordre_du_modele(graphe):
    candidats = [_candidat("IGGLIA", 0.54), _candidat("DTJA", 0.03)]
    resultat = appliquer_regles_admission(candidats, ProfilCandidat(serie_bac="D"), graphe)

    assert [c.parcours for c in resultat] == ["IGGLIA", "DTJA"]
    assert "[Règle d'admission]" not in resultat[0].justification


def test_serie_non_declaree_laisse_le_classement_intact(graphe):
    """L'incertitude ne doit pas se transformer en pénalité silencieuse."""
    candidats = [_candidat("IGGLIA", 0.54), _candidat("DTJA", 0.03)]
    resultat = appliquer_regles_admission(candidats, ProfilCandidat(), graphe)

    assert [c.parcours for c in resultat] == ["IGGLIA", "DTJA"]


def test_graphe_absent_laisse_le_classement_intact():
    """L'enrichissement symbolique est un bonus, jamais une condition pour
    répondre."""
    candidats = [_candidat("IGGLIA", 0.54), _candidat("DTJA", 0.03)]
    resultat = appliquer_regles_admission(candidats, ProfilCandidat(serie_bac="A"), None)

    assert [c.parcours for c in resultat] == ["IGGLIA", "DTJA"]


def test_graphe_vide_ne_retrograde_rien():
    """Sans prérequis connus, l'admissibilité est indéterminable — donc pas de
    rétrogradation."""
    candidats = [_candidat("IGGLIA", 0.54)]
    resultat = appliquer_regles_admission(
        candidats, ProfilCandidat(serie_bac="A"), nx.DiGraph()
    )

    assert [c.parcours for c in resultat] == ["IGGLIA"]


def test_evaluer_admissibilite_expose_les_prerequis(graphe):
    verdict = evaluer_admissibilite(graphe, "IGGLIA", "A")
    assert verdict.admissible is False
    assert verdict.inadmissible is True
    assert verdict.prerequis  # les prérequis réels sont remontés


def test_verdict_indetermine_n_est_pas_un_refus(graphe):
    verdict = evaluer_admissibilite(graphe, "IGGLIA", None)
    assert verdict.admissible is None
    assert verdict.inadmissible is False


# --- Intégration bout en bout dans le modèle ---------------------------------


def test_le_modele_ne_recommande_plus_de_parcours_inaccessible():
    """Le cas réel : un profil informatique avec un Bac A obtenait ses quatre
    premières recommandations parmi des parcours exigeant un Bac C/D/S."""
    profil = ProfilCandidat(
        matieres_preferees=["maths", "info"],
        competences_declarees=["Python"],
        centres_interet=["IA"],
        serie_bac="A",
    )
    graphe_reel = construire_graphe(charger_corpus_formations())

    for candidat in classer_parcours(profil, top_k=4):
        verdict = evaluer_admissibilite(graphe_reel, candidat.parcours, "A")
        assert not verdict.inadmissible, f"{candidat.parcours} n'est pas accessible en Bac A"


def test_le_meme_profil_admissible_garde_sa_meilleure_recommandation():
    """Contrôle négatif : la règle ne doit pas dégrader un candidat éligible."""
    profil = ProfilCandidat(
        matieres_preferees=["maths", "info"],
        competences_declarees=["Python"],
        centres_interet=["IA"],
        serie_bac="D",
    )
    assert classer_parcours(profil, top_k=1)[0].parcours == "IGGLIA"
