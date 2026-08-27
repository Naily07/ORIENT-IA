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
    precharger,
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


def test_classer_parcours_ne_propose_que_les_parcours_significatifs():
    """Le contrat a changé : `classer_parcours` ne renvoie plus `top_k`
    éléments par principe, mais seulement ceux qui atteignent une fraction du
    score de tête. Un top-3 constant présentait du bruit — score médian 92 % au
    rang 1 contre 2,0 % au rang 2 — et c'est ce bruit qui rendait le classement
    instable (34 % des profils changeaient de top-3 au retrait d'un trait)."""
    top = classer_parcours(_profil_type_igglia(), top_k=3)

    assert 1 <= len(top) <= 3
    assert top == sorted(top, key=lambda c: c.score_adequation, reverse=True)
    # Sur un profil d'archétype pur, le modèle est net : une seule proposition.
    assert len(top) == 1


def test_classer_parcours_propose_plusieurs_options_quand_le_modele_hesite():
    """L'inverse doit rester vrai : restreindre ne doit pas revenir à toujours
    n'afficher qu'un seul parcours."""
    from src.ml.outils import selectionner_significatifs
    from src.schemas import RecommandationParcours

    candidats = [
        RecommandationParcours(parcours="A", score_adequation=0.40, justification="—"),
        RecommandationParcours(parcours="B", score_adequation=0.35, justification="—"),
        RecommandationParcours(parcours="C", score_adequation=0.20, justification="—"),
        RecommandationParcours(parcours="D", score_adequation=0.02, justification="—"),
    ]
    retenus = selectionner_significatifs(candidats)
    assert [c.parcours for c in retenus] == ["A", "B", "C"]


def test_selectionner_significatifs_ecarte_le_bruit():
    from src.ml.outils import selectionner_significatifs
    from src.schemas import RecommandationParcours

    candidats = [
        RecommandationParcours(parcours="A", score_adequation=0.92, justification="—"),
        RecommandationParcours(parcours="B", score_adequation=0.02, justification="—"),
        RecommandationParcours(parcours="C", score_adequation=0.01, justification="—"),
    ]
    assert [c.parcours for c in selectionner_significatifs(candidats)] == ["A"]


def test_selectionner_significatifs_renvoie_toujours_au_moins_un_parcours():
    """Une liste vide priverait l'appelant de la sortie du modèle sans rien
    dire de plus."""
    from src.ml.outils import selectionner_significatifs
    from src.schemas import RecommandationParcours

    assert selectionner_significatifs([]) == []
    plat = [RecommandationParcours(parcours="A", score_adequation=0.0, justification="—")]
    assert len(selectionner_significatifs(plat)) == 1


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


# --- Honnêteté sur un profil inexploitable (correctif d'audit) ----------------


def test_le_score_d_un_profil_inexploitable_porte_son_avertissement():
    """Non-régression : `analyser_profil` déclarait `confiance=0.0`, mais
    `classer_parcours` et `calculer_adequation` renvoyaient le score nu. Les deux
    outils que l'agent appelle le plus contournaient donc entièrement le garde-fou
    « refuse d'affirmer sans signal »."""
    profil = ProfilCandidat(matieres_preferees=["cuisine"])
    analyse = analyser_profil(profil)

    assert analyse.profil_exploitable is False
    assert analyse.confiance == 0.0
    assert "cuisine" in analyse.elements_non_reconnus
    # L'avertissement voyage avec chaque candidat, donc avec classer_parcours.
    assert all("non informatif" in c.justification for c in analyse.parcours_candidats)


def test_classer_parcours_propage_l_avertissement():
    profil = ProfilCandidat(matieres_preferees=["cuisine"])
    assert "non informatif" in classer_parcours(profil)[0].justification


def test_un_profil_exploitable_n_est_pas_annote():
    """Contrôle : le correctif ne doit pas polluer les profils normaux."""
    profil = ProfilCandidat(
        matieres_preferees=["informatique"], competences_declarees=["programmation"]
    )
    analyse = analyser_profil(profil)

    assert analyse.profil_exploitable is True
    assert analyse.confiance > 0.0
    assert all("non informatif" not in c.justification for c in analyse.parcours_candidats)


# --- Préchauffage (constat d'audit P3) --------------------------------------


def test_precharger_ne_leve_jamais():
    """Sur le jeu livré avec le dépôt, `precharger()` doit réussir sans
    exception — c'est la condition pour l'appeler depuis `api.lifespan()`
    sans jamais empêcher le serveur de démarrer."""
    precharger()  # ne doit pas lever


def test_precharger_peuple_le_cache_du_modele():
    """Après `precharger()`, `analyser_profil` ne doit plus payer le coût
    d'entraînement — vérifié indirectement via le cache `lru_cache` de
    `_modele`, dont l'info expose les appels réussis."""
    from src.ml import outils

    outils._modele.cache_clear()
    outils._modele_explicatif.cache_clear()

    precharger()

    assert outils._modele.cache_info().currsize == 1
    assert outils._modele_explicatif.cache_info().currsize == 1


def test_precharger_tolere_l_absence_de_jeu_de_donnees(monkeypatch):
    """Un dépôt fraîchement cloné, avant `python -m src.ml.donnees_synthetiques`,
    ne doit pas empêcher le serveur de démarrer."""
    from src.ml import outils

    outils._modele.cache_clear()
    outils._modele_explicatif.cache_clear()
    monkeypatch.setattr(outils, "_jeu_ou_erreur", lambda: (_ for _ in ()).throw(RuntimeError()))

    precharger()  # ne doit pas lever
