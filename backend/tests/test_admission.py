"""Tests de la règle d'admissibilité (`src/admission.py`).

Ce module était le seul de `src/` sans fichier de test dédié, alors que c'est
la **règle partagée** entre les deux chemins qui décident de l'admissibilité :
l'outil `verifier_prerequis` que l'agent appelle, et la rétrogradation des
parcours inaccessibles du volet hybride. Elle était exercée indirectement des
deux côtés ; ses cas limites — série absente, série blanche, prérequis inconnu
— ne l'étaient pas tous.
"""

from src.admission import SERIES_TOUTE, serie_satisfait_prerequis

SCIENTIFIQUE = ["Baccalauréat série C, D, S, ou série techniques industrielles"]
TOUTE_SERIE = ["Baccalauréat toute série"]


# --- Cas tranchés --------------------------------------------------------------


def test_une_serie_listee_satisfait_le_prerequis():
    assert serie_satisfait_prerequis("D", SCIENTIFIQUE) is True


def test_une_serie_absente_de_la_liste_ne_satisfait_pas():
    assert serie_satisfait_prerequis("A", SCIENTIFIQUE) is False


def test_toute_serie_accepte_n_importe_quelle_serie():
    assert serie_satisfait_prerequis("L", TOUTE_SERIE) is True
    assert SERIES_TOUTE in TOUTE_SERIE[0].lower()


def test_la_casse_n_est_pas_significative():
    assert serie_satisfait_prerequis("d", SCIENTIFIQUE) is True


def test_un_seul_prerequis_satisfait_suffit():
    assert serie_satisfait_prerequis("L", SCIENTIFIQUE + TOUTE_SERIE) is True


# --- Indécidable : `None` n'est pas un « non » ---------------------------------


def test_sans_serie_declaree_la_question_reste_ouverte():
    """`None` doit être remonté comme information manquante, pas transformé en
    refus : trancher à la place du candidat serait une décision administrative
    que l'assistant n'a pas à prendre (§16)."""
    assert serie_satisfait_prerequis(None, SCIENTIFIQUE) is None


def test_sans_prerequis_connu_la_question_reste_ouverte():
    assert serie_satisfait_prerequis("D", []) is None


def test_une_serie_faite_d_espaces_reste_indecidable():
    """Non-régression : `.strip()` n'était appliqué que dans la regex, pas au
    test de vérité. Une saisie blanche produisait un motif vide dont `\\b\\b`
    correspond à n'importe quel texte — et la règle confirmait l'admissibilité
    d'un candidat qui n'avait rien déclaré."""
    assert serie_satisfait_prerequis("   ", SCIENTIFIQUE) is None
    assert serie_satisfait_prerequis("", SCIENTIFIQUE) is None


# --- Frontières de mot ---------------------------------------------------------


def test_une_lettre_isolee_ne_matche_pas_au_milieu_d_un_mot():
    """« L » ne doit pas correspondre au « L » de « baccaLauréat ». Trouvé en
    testant « série L » contre « Baccalauréat série C, D, S »."""
    assert serie_satisfait_prerequis("L", SCIENTIFIQUE) is False


def test_une_serie_composee_est_reconnue_telle_quelle():
    assert serie_satisfait_prerequis("A2", ["Baccalauréat série A2 avec 12/20"]) is True
    assert serie_satisfait_prerequis("A2", SCIENTIFIQUE) is False


def test_un_caractere_special_ne_casse_pas_la_regex():
    """La série est échappée : une saisie contenant un métacaractère ne doit ni
    lever ni matcher n'importe quoi."""
    assert serie_satisfait_prerequis("C.*", SCIENTIFIQUE) is False
