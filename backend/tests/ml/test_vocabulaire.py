"""Tests de la résolution de vocabulaire ouvert.

Les couches déterministes (normalisation, alias) sont testées avec
`avec_semantique=False` : elles doivent suffire seules sur les cas courants, sans
charger le modèle d'embedding. Les tests du repli sémantique sont marqués `index`
(ils téléchargent/chargent le modèle ONNX, comme les tests RAG).
"""

import pytest

from src.ml.archetypes import VOCAB_COMPETENCES, VOCAB_MATIERES
from src.ml.vocabulaire import normaliser, resoudre

# --- Couche 1 : normalisation ------------------------------------------------


def test_normalisation_casse_accents_et_separateurs():
    assert normaliser("Mathématiques") == "mathematiques"
    assert normaliser("Physique-Chimie") == "physique_chimie"
    assert normaliser("  jeux vidéo  ") == "jeux_video"


def test_normalisation_conserve_le_plus_de_c_plus_plus():
    """`c++` est un terme réel qu'un candidat peut taper : le `+` ne doit pas
    être écrasé comme un séparateur ordinaire."""
    assert normaliser("C++") == "c++"


def test_normalisation_terme_vide():
    assert normaliser("") == ""
    assert normaliser("   ") == ""


# --- Couche 2 : alias (déterministe, sans embedding) -------------------------


def test_terme_exact_du_vocabulaire_est_reconnu():
    reconnus, non_reconnus = resoudre(["mathematiques"], VOCAB_MATIERES, avec_semantique=False)
    assert reconnus == ["mathematiques"]
    assert non_reconnus == []


def test_alias_courants_sont_resolus_sans_embedding():
    """Ces cas doivent passer par la couche alias : « info » ne ressort qu'à
    0,386 de similarité sémantique, sous le seuil, alors que c'est un synonyme
    évident — c'est précisément ce que la couche alias rattrape."""
    reconnus, non_reconnus = resoudre(
        ["maths", "info", "SVT"], VOCAB_MATIERES, avec_semantique=False
    )
    assert set(reconnus) == {"mathematiques", "informatique", "biologie"}
    assert non_reconnus == []


def test_alias_de_competence_est_resolu():
    reconnus, _ = resoudre(["Python", "algo"], VOCAB_COMPETENCES, avec_semantique=False)
    assert set(reconnus) == {"programmation", "algorithmique"}


def test_accents_et_casse_traverses_par_les_alias():
    reconnus, _ = resoudre(["Éco"], VOCAB_MATIERES, avec_semantique=False)
    assert reconnus == ["economie"]


def test_termes_inconnus_sont_remontes_tels_que_declares():
    """Un terme non reconnu ne doit jamais disparaître en silence : il est
    remonté dans sa forme d'origine pour pouvoir être cité à l'utilisateur."""
    _, non_reconnus = resoudre(["Philosophie"], VOCAB_MATIERES, avec_semantique=False)
    assert non_reconnus == ["Philosophie"]


def test_doublons_apres_resolution_sont_dedoublonnes():
    """« maths » et « mathematiques » retombent sur le même terme : le
    multi-hot ne doit pas le compter deux fois."""
    reconnus, _ = resoudre(
        ["maths", "mathematiques", "math"], VOCAB_MATIERES, avec_semantique=False
    )
    assert reconnus == ["mathematiques"]


def test_liste_vide_et_termes_vides():
    assert resoudre([], VOCAB_MATIERES, avec_semantique=False) == ([], [])
    assert resoudre(["", "  "], VOCAB_MATIERES, avec_semantique=False) == ([], [])


def test_sans_semantique_les_inconnus_restent_inconnus():
    reconnus, non_reconnus = resoudre(
        ["quelque chose de totalement inconnu"], VOCAB_MATIERES, avec_semantique=False
    )
    assert reconnus == []
    assert len(non_reconnus) == 1


# --- Couche 3 : repli sémantique ---------------------------------------------


@pytest.mark.index
def test_repli_semantique_rattrape_un_terme_proche():
    """« physique-chimie » n'est ni dans le vocabulaire ni dans les alias, mais
    reste sémantiquement proche de « physique » (0,72 mesuré)."""
    reconnus, non_reconnus = resoudre(["physique-chimie"], VOCAB_MATIERES)
    assert reconnus == ["physique"]
    assert non_reconnus == []


@pytest.mark.index
def test_repli_semantique_rejette_le_hors_domaine():
    """Le comportement le plus important : ne PAS fabriquer une correspondance.
    « philosophie » ressort à 0,36 vers « comptabilite » — mapper de force
    aurait produit un profil faux."""
    reconnus, non_reconnus = resoudre(["philosophie", "cuisine"], VOCAB_MATIERES)
    assert reconnus == []
    assert set(non_reconnus) == {"philosophie", "cuisine"}


@pytest.mark.index
def test_seuil_configurable_change_la_tolerance():
    """Un seuil très permissif accepterait ce que le seuil par défaut rejette :
    documente que le rejet vient bien du seuil, pas d'une absence de calcul."""
    reconnus_stricts, _ = resoudre(["philosophie"], VOCAB_MATIERES, seuil=0.9)
    reconnus_laxistes, _ = resoudre(["philosophie"], VOCAB_MATIERES, seuil=0.1)
    assert reconnus_stricts == []
    assert reconnus_laxistes != []
