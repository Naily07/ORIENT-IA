"""Tests des outils ML exposés à l'agent (ML-8) — sur le vrai jeu de données
synthétique livré avec le projet (`backend/data/ml/profils_synthetiques.json`).
"""

from src.ml.archetypes import ARCHETYPES
from src.ml.donnees_synthetiques import generer_profil
from src.ml.outils import (
    analyser_profil,
    calculer_adequation,
    classer_parcours,
    identifier_points_forts,
)
from src.schemas import ProfilCandidat


def _profil_type_igglia() -> ProfilCandidat:
    import random

    return generer_profil("IGGLIA", random.Random(123))


def test_analyser_profil_retourne_tous_les_parcours_connus_tries():
    analyse = analyser_profil(_profil_type_igglia())
    assert len(analyse.parcours_candidats) == len(ARCHETYPES)
    scores = [c.score_adequation for c in analyse.parcours_candidats]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 <= analyse.confiance <= 1.0
    assert analyse.confiance == scores[0]


def test_classer_parcours_retourne_le_top_k():
    top = classer_parcours(_profil_type_igglia(), top_k=3)
    assert len(top) == 3
    assert top[0].score_adequation >= top[1].score_adequation >= top[2].score_adequation


def test_calculer_adequation_pour_un_parcours_inconnu_retourne_zero():
    assert calculer_adequation(_profil_type_igglia(), "PARCOURS-INEXISTANT") == 0.0


def test_calculer_adequation_correspond_au_score_de_analyser_profil():
    profil = _profil_type_igglia()
    analyse = analyser_profil(profil)
    meilleur = analyse.parcours_candidats[0]
    assert calculer_adequation(profil, meilleur.parcours) == meilleur.score_adequation


def test_identifier_points_forts_ne_depasse_pas_top_n():
    points_forts = identifier_points_forts(_profil_type_igglia(), top_n=3)
    assert len(points_forts) <= 3
    assert all(isinstance(p, str) for p in points_forts)


def test_identifier_points_forts_sur_profil_vide_ne_leve_pas():
    assert identifier_points_forts(ProfilCandidat()) == []
