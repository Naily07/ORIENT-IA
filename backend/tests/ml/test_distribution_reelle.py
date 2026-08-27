"""Tests de la mesure de distribution réelle et du lissage (DATA-6 bis)."""

import json

from src.ml.archetypes import PARCOURS_CONNUS
from src.ml.distribution_reelle import (
    DistributionReelle,
    effectifs_cibles,
    mesurer,
    prior_lisse,
)


def _ecrire(chemin, entrees):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(entrees, f)
    return chemin


def test_sans_aucun_effectif_le_prior_est_uniforme():
    """Un dépôt sans enquête collectée doit se comporter comme le générateur
    d'origine, pas échouer."""
    prior = prior_lisse({}, PARCOURS_CONNUS)
    assert len(prior) == len(PARCOURS_CONNUS)
    assert all(abs(p - 1 / len(PARCOURS_CONNUS)) < 1e-9 for p in prior.values())


def test_le_prior_somme_a_un():
    prior = prior_lisse({"IGGLIA": 40, "ESIIA": 10}, PARCOURS_CONNUS)
    assert abs(sum(prior.values()) - 1.0) < 1e-9


def test_la_classe_dominante_de_l_enquete_reste_dominante_apres_lissage():
    prior = prior_lisse({"IGGLIA": 40, "ESIIA": 10, "AEE": 1}, PARCOURS_CONNUS)
    assert prior["IGGLIA"] > prior["ESIIA"] > prior["AEE"]


def test_un_parcours_jamais_observe_garde_un_effectif_non_nul():
    """Recopier l'empirique tel quel laisserait EMP et TEE à zéro profil
    d'entraînement — le lissage existe précisément pour l'éviter."""
    prior = prior_lisse({"IGGLIA": 40}, PARCOURS_CONNUS)
    assert prior["EMP"] > 0
    effectifs = effectifs_cibles(prior, 800)
    assert effectifs["EMP"] > 0


def test_alpha_zero_redonne_exactement_l_uniforme():
    prior = prior_lisse({"IGGLIA": 40, "ESIIA": 10}, PARCOURS_CONNUS, alpha=0.0)
    assert all(abs(p - 1 / len(PARCOURS_CONNUS)) < 1e-9 for p in prior.values())


def test_alpha_plus_grand_accentue_le_desequilibre():
    faible = prior_lisse({"IGGLIA": 40, "ESIIA": 10}, PARCOURS_CONNUS, alpha=0.2)
    fort = prior_lisse({"IGGLIA": 40, "ESIIA": 10}, PARCOURS_CONNUS, alpha=0.9)
    assert fort["IGGLIA"] > faible["IGGLIA"]


def test_les_effectifs_cibles_somment_exactement_au_total_demande():
    """Arrondir chaque classe indépendamment ferait dériver le total ; la
    méthode des plus forts restes doit tomber juste."""
    prior = prior_lisse({"IGGLIA": 36, "ESIIA": 16, "ISAIA": 12}, PARCOURS_CONNUS)
    for total in (16, 100, 799, 800, 1001):
        effectifs = effectifs_cibles(prior, total)
        assert sum(effectifs.values()) == total


def test_effectifs_cibles_est_deterministe():
    prior = prior_lisse({"IGGLIA": 36, "ESIIA": 16}, PARCOURS_CONNUS)
    assert effectifs_cibles(prior, 800) == effectifs_cibles(prior, 800)


def test_mesurer_additionne_les_deux_enquetes(tmp_path):
    externe = _ecrire(
        tmp_path / "reel.json",
        [
            {"parcours_id": "IGGLIA", "profil": {"matieres_preferees": ["maths"]}},
            {"parcours_id": "IGGLIA", "profil": {"matieres_preferees": ["info"]}},
        ],
    )
    interne = _ecrire(
        tmp_path / "enquete.json",
        [{"parcours_declare": "ESIIA", "profil": {"matieres_preferees": ["physique"]}}],
    )

    d = mesurer(externe, interne)
    assert d.n_reponses == 3
    assert d.effectifs_parcours == {"IGGLIA": 2, "ESIIA": 1}
    assert d.sources == ("reel.json", "enquete.json")


def test_mesurer_tolere_des_fichiers_absents(tmp_path):
    d = mesurer(tmp_path / "absent.json", tmp_path / "absent2.json")
    assert d.n_reponses == 0
    assert d.effectifs_parcours == {}
    assert d.sources == ()


def test_mesurer_ignore_les_reponses_sans_etiquette(tmp_path):
    """Une réponse non rattachable à un parcours compte dans la complétude
    mais ne doit pas peser sur le prior des classes."""
    externe = _ecrire(
        tmp_path / "reel.json",
        [
            {"parcours_id": None, "profil": {"matieres_preferees": ["maths"]}},
            {"parcours_id": "IGGLIA", "profil": {"matieres_preferees": ["info"]}},
        ],
    )
    d = mesurer(externe, tmp_path / "absent.json")
    assert d.n_reponses == 2
    assert d.n_etiquetees == 1
    assert d.effectifs_parcours == {"IGGLIA": 1}


def test_la_completude_mesure_le_taux_de_presence_par_champ(tmp_path):
    externe = _ecrire(
        tmp_path / "reel.json",
        [
            {
                "parcours_id": "IGGLIA",
                "profil": {"matieres_preferees": ["a", "b"], "centres_interet": []},
            },
            {
                "parcours_id": "ESIIA",
                "profil": {"matieres_preferees": [], "centres_interet": ["x"]},
            },
        ],
    )
    d = mesurer(externe, tmp_path / "absent.json")
    assert d.completude["matieres_preferees"].taux_presence == 0.5
    assert d.completude["matieres_preferees"].tailles == (2,)
    assert d.completude["centres_interet"].taux_presence == 0.5


def test_taille_mediane_sans_observation_vaut_zero():
    d = DistributionReelle()
    assert d.prior(PARCOURS_CONNUS)["IGGLIA"] > 0
