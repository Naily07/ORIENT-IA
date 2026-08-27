"""Tests de la préparation du jeu de test ML réel depuis un export d'enquête
anonymisé (voir `scripts/preparer_jeu_test_reel.py`, DATA-5/DATA-7/DATA-8)."""

from scripts.preparer_jeu_test_reel import (
    COL_ADAPTATION,
    COL_MATIERES_PREFEREES,
    COL_METIER,
    COL_NOTE_DECLAREE,
    COL_PARCOURS_ACTUEL,
    COL_PARCOURS_SUIVI,
    COL_SATISFACTION,
    COL_TYPE,
    construire_enregistrement,
    convertir_echelle_vers_note20,
    extraire_termes,
    normaliser_metier_declare,
    parser_type,
    resoudre_parcours_ou_mention,
)

IDS_PARCOURS = {"IGGLIA", "ESIIA", "ISAIA", "AEE", "IAA", "PIP"}
NOMS_MENTIONS = {
    "MENTION-BIOTECH-AGRO": "Biotechnologie et Agronomie",
    "MENTION-INFO-TELECOM": "Informatique et Télécommunications",
}


def test_parser_type_reconnait_les_deux_populations():
    assert parser_type("Étudiant(e)") == "etudiant"
    assert parser_type("Professionnel(le) (diplômé ISPM)") == "professionnel"


def test_parser_type_valeur_inconnue_retourne_none():
    assert parser_type("autre chose") is None
    assert parser_type(None) is None


def test_convertir_echelle_est_lineaire_et_borne():
    assert convertir_echelle_vers_note20("1") == 4.0
    assert convertir_echelle_vers_note20("5") == 20.0
    assert convertir_echelle_vers_note20("3") == 12.0


def test_convertir_echelle_rejette_hors_domaine():
    assert convertir_echelle_vers_note20("0") is None
    assert convertir_echelle_vers_note20("6") is None
    assert convertir_echelle_vers_note20("") is None
    assert convertir_echelle_vers_note20(None) is None
    assert convertir_echelle_vers_note20("pas un chiffre") is None


def test_extraire_termes_deduplique_et_nettoie():
    assert extraire_termes("Mathématiques, Informatique, Mathématiques") == [
        "Mathématiques",
        "Informatique",
    ]
    assert extraire_termes("") == []
    assert extraire_termes(None) == []


def test_normaliser_metier_declare_filtre_les_non_reponses():
    assert normaliser_metier_declare("Anonyme ") is None
    assert normaliser_metier_declare("Aucun") is None
    assert normaliser_metier_declare("") is None
    assert normaliser_metier_declare(None) is None
    assert normaliser_metier_declare("Développeur web") == "Développeur web"


def test_resoud_un_code_parcours_au_mot_entier():
    assert resoudre_parcours_ou_mention("Igglia 2024", IDS_PARCOURS, NOMS_MENTIONS) == (
        "IGGLIA",
        None,
        "parcours",
    )


def test_ne_confond_pas_un_code_avec_une_sous_chaine():
    """`AEE`/`IAA` ne doivent pas ressortir d'un texte qui ne les contient
    qu'en tant que fragments d'un autre mot (ex. « MISA », « BACC »)."""
    assert resoudre_parcours_ou_mention("MISA+BACC+4", IDS_PARCOURS, NOMS_MENTIONS) == (
        None,
        None,
        "aucune",
    )


def test_resoud_une_mention_a_defaut_dun_code_de_parcours():
    parcours_id, mention_id, granularite = resoudre_parcours_ou_mention(
        "Biotechnologie et Agronomie 2022", IDS_PARCOURS, NOMS_MENTIONS
    )
    assert parcours_id is None
    assert mention_id == "MENTION-BIOTECH-AGRO"
    assert granularite == "mention"


def test_texte_vide_ne_resout_rien():
    assert resoudre_parcours_ou_mention("", IDS_PARCOURS, NOMS_MENTIONS) == (None, None, "aucune")
    assert resoudre_parcours_ou_mention(None, IDS_PARCOURS, NOMS_MENTIONS) == (
        None,
        None,
        "aucune",
    )


def _ligne_etudiant(**overrides) -> dict:
    base = {
        COL_TYPE: "Étudiant(e)",
        COL_PARCOURS_ACTUEL: "IGGLIA",
        COL_MATIERES_PREFEREES: "Mathématiques, Informatique",
        COL_NOTE_DECLAREE: "4",
        COL_SATISFACTION: "5",
    }
    base.update(overrides)
    return base


def _ligne_professionnel(**overrides) -> dict:
    base = {
        COL_TYPE: "Professionnel(le) (diplômé ISPM)",
        COL_PARCOURS_SUIVI: "ESIIA 2017",
        COL_METIER: "Développeur fullstack",
        COL_ADAPTATION: "4",
    }
    base.update(overrides)
    return base


def test_enregistrement_etudiant_reporte_la_meme_note_sur_maths_et_info():
    enregistrement = construire_enregistrement(
        _ligne_etudiant(), 1, IDS_PARCOURS, NOMS_MENTIONS, seuil_satisfaction=3
    )
    assert enregistrement["parcours_id"] == "IGGLIA"
    assert enregistrement["profil"]["resultats_scolaires"] == {
        "mathematiques": 16.0,
        "informatique": 16.0,
    }
    assert enregistrement["profil"]["matieres_preferees"] == ["Mathématiques", "Informatique"]
    assert enregistrement["usable_pour_eval"] is True
    assert enregistrement["label_fiable"] is True


def test_enregistrement_professionnel_remplit_preferences_professionnelles():
    enregistrement = construire_enregistrement(
        _ligne_professionnel(), 2, IDS_PARCOURS, NOMS_MENTIONS, seuil_satisfaction=3
    )
    assert enregistrement["parcours_id"] == "ESIIA"
    assert enregistrement["profil"]["preferences_professionnelles"] == ["Développeur fullstack"]
    assert enregistrement["profil"]["matieres_preferees"] == []
    assert enregistrement["profil"]["resultats_scolaires"] == {}


def test_champs_jamais_collectes_restent_vides_jamais_devines():
    enregistrement = construire_enregistrement(
        _ligne_etudiant(), 3, IDS_PARCOURS, NOMS_MENTIONS, seuil_satisfaction=3
    )
    profil = enregistrement["profil"]
    assert profil["serie_bac"] is None
    assert profil["activites_projets"] == []
    assert profil["competences_declarees"] == []
    assert profil["centres_interet"] == []
    assert profil["environnement_travail_recherche"] is None


def test_satisfaction_sous_le_seuil_rend_l_etiquette_non_fiable():
    enregistrement = construire_enregistrement(
        _ligne_etudiant(**{COL_SATISFACTION: "1"}),
        4,
        IDS_PARCOURS,
        NOMS_MENTIONS,
        seuil_satisfaction=3,
    )
    assert enregistrement["usable_pour_eval"] is True
    assert enregistrement["label_fiable"] is False


def test_type_inconnu_ne_produit_aucun_enregistrement():
    ligne = _ligne_etudiant(**{COL_TYPE: "autre chose"})
    assert construire_enregistrement(ligne, 5, IDS_PARCOURS, NOMS_MENTIONS, 3) is None


def test_metier_anonyme_ne_devient_pas_une_preference_professionnelle():
    enregistrement = construire_enregistrement(
        _ligne_professionnel(**{COL_METIER: "Anonyme"}),
        6,
        IDS_PARCOURS,
        NOMS_MENTIONS,
        seuil_satisfaction=3,
    )
    assert enregistrement["profil"]["preferences_professionnelles"] == []
