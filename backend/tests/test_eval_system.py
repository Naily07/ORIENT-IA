"""Tests de la logique de vérification du harnais d'évaluation système
(`_verifier_attendu`), sans appel réseau : le pipeline lui-même n'est pas
exercé ici (voir `eval_system.py`, lancé manuellement, pour l'évaluation
réelle des 32 cas)."""

from src.schemas import RecommandationDecision
from tests.eval_system import _verifier_attendu, charger_dataset


def _decision(**overrides) -> RecommandationDecision:
    valeurs = {
        "resume": "resume neutre",
        "parcours_recommandes": [],
        "confiance": 0.9,
        "informations_manquantes": [],
        "explication": "explication neutre",
        "sources": [],
        "outils_utilises": [],
        "action": "recommandation",
        "incertitude_declaree": False,
    }
    valeurs.update(overrides)
    return RecommandationDecision(**valeurs)


def test_aucun_attendu_reussit_toujours():
    assert _verifier_attendu(_decision(), {}) == []


def test_action_hors_liste_acceptable_echoue():
    echecs = _verifier_attendu(
        _decision(action="recommandation"), {"actions_acceptables": ["escalade_conseiller"]}
    )
    assert echecs


def test_action_dans_la_liste_acceptable_reussit():
    echecs = _verifier_attendu(
        _decision(action="demande_information"),
        {"actions_acceptables": ["demande_information", "escalade_conseiller"]},
    )
    assert echecs == []


def test_doit_escalader_echoue_si_action_differente():
    echecs = _verifier_attendu(_decision(action="recommandation"), {"doit_escalader": True})
    assert echecs


def test_doit_escalader_reussit_si_escalade():
    echecs = _verifier_attendu(_decision(action="escalade_conseiller"), {"doit_escalader": True})
    assert echecs == []


def test_aucun_outil_appele_echoue_si_des_outils_ont_ete_appeles():
    echecs = _verifier_attendu(
        _decision(outils_utilises=["rechercher_formation"]), {"aucun_outil_appele": True}
    )
    assert echecs


def test_sources_attendues_manquantes_echoue():
    echecs = _verifier_attendu(
        _decision(sources=["DOC-A"]), {"sources_attendues": ["DOC-A", "DOC-B"]}
    )
    assert echecs
    assert "DOC-B" in echecs[0]


def test_sources_attendues_toutes_presentes_reussit():
    echecs = _verifier_attendu(
        _decision(sources=["DOC-A", "DOC-B"]), {"sources_attendues": ["DOC-A", "DOC-B"]}
    )
    assert echecs == []


def test_doit_appeler_analyser_profil_ml_echoue_si_absent():
    echecs = _verifier_attendu(
        _decision(outils_utilises=["verifier_prerequis"]), {"doit_appeler_analyser_profil_ml": True}
    )
    assert echecs


def test_contenu_interdit_detecte_dans_l_explication():
    echecs = _verifier_attendu(
        _decision(explication="Une nouvelle filière de robotique existe."),
        {"ne_doit_pas_contenir": ["nouvelle filière de robotique existe"]},
    )
    assert echecs


def test_contenu_interdit_absent_reussit():
    echecs = _verifier_attendu(
        _decision(explication="Aucune formation de ce type n'a été trouvée."),
        {"ne_doit_pas_contenir": ["nouvelle filière de robotique existe"]},
    )
    assert echecs == []


def test_contenu_interdit_insensible_a_la_casse():
    echecs = _verifier_attendu(
        _decision(resume="UNE NOUVELLE FILIÈRE DE ROBOTIQUE EXISTE."),
        {"ne_doit_pas_contenir": ["nouvelle filière de robotique existe"]},
    )
    assert echecs


# --- Cohérence du jeu de données réel ----------------------------------------


def test_le_jeu_de_donnees_reel_respecte_les_minimums_du_sujet():
    """§13 du sujet : au moins 32 cas, répartis avec un minimum par catégorie."""
    minimums = {
        "questions_factuelles": 5,
        "comparaisons": 4,
        "profils_ml": 6,
        "multi_sources": 4,
        "information_absente": 3,
        "profils_ambigus": 3,
        "securite_injection": 3,
        "biais": 2,
        "provenance_profilage": 2,
    }
    dataset = charger_dataset()
    assert len(dataset) >= 32

    comptage: dict[str, int] = {}
    for cas in dataset:
        comptage[cas["categorie"]] = comptage.get(cas["categorie"], 0) + 1

    for categorie, minimum in minimums.items():
        assert comptage.get(categorie, 0) >= minimum, (
            f"catégorie « {categorie} » : {comptage.get(categorie, 0)} cas, "
            f"minimum {minimum} exigé par le sujet"
        )


def test_chaque_cas_a_un_identifiant_unique():
    dataset = charger_dataset()
    ids = [cas["id"] for cas in dataset]
    assert len(ids) == len(set(ids))
