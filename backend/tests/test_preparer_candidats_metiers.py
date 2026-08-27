"""Tests de la préparation de candidats de métiers depuis une source externe
non vérifiée (voir `scripts/preparer_candidats_metiers.py`)."""

from scripts.preparer_candidats_metiers import (
    construire_candidats_metiers,
    extraire_debouches_par_parcours,
)


def test_ignore_les_parcours_absents_du_corpus_reel():
    entrees = [
        {"code_filiere": "IGGLIA", "debouches": ["Développeur Web/Mobile"]},
        {"code_filiere": "PARCOURS-INCONNU", "debouches": ["Métier fantôme"]},
    ]
    resultat = extraire_debouches_par_parcours(entrees, ids_parcours_connus={"IGGLIA"})
    assert resultat == {"IGGLIA": ["Développeur Web/Mobile"]}


def test_ignore_les_entrees_sans_debouches():
    entrees = [{"code_filiere": "IGGLIA", "debouches": []}]
    assert extraire_debouches_par_parcours(entrees, {"IGGLIA"}) == {}


def test_deduplique_un_intitule_de_metier_partage_entre_parcours():
    debouches = {
        "IGGLIA": ["Développeur Web/Mobile", "Consultant IA"],
        "ESIIA": ["Consultant IA"],
    }
    metiers, par_parcours = construire_candidats_metiers(debouches, mention_par_parcours={})

    noms = [m["nom"] for m in metiers]
    assert noms.count("Consultant IA") == 1
    id_consultant_ia = next(m["id"] for m in metiers if m["nom"] == "Consultant IA")
    assert par_parcours["IGGLIA"][1] == id_consultant_ia
    assert par_parcours["ESIIA"] == [id_consultant_ia]


def test_ne_fusionne_jamais_deux_intitules_differents_meme_proches():
    """Pas de rapprochement flou : « Data Scientist » et « Data Analyst » restent
    deux métiers distincts, même s'ils partagent un mot."""
    debouches = {"ISAIA": ["Data Scientist / Data Analyst", "Data Scientist"]}
    metiers, _ = construire_candidats_metiers(debouches, mention_par_parcours={})
    assert len(metiers) == 2


def test_chaque_candidat_porte_le_secteur_et_la_source():
    debouches = {"GCA": ["Ingénieur Génie Civil"]}
    metiers, _ = construire_candidats_metiers(
        debouches,
        mention_par_parcours={"GCA": "Génie Civil et Architecture"},
        source_id="SRC-TEST",
    )
    assert metiers[0]["secteur"] == "Génie Civil et Architecture"
    assert metiers[0]["source_id"] == "SRC-TEST"
    assert metiers[0]["id"].startswith("MET-")
