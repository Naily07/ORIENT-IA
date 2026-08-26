"""Tests des fonctions pures du frontend Streamlit.

Une application Streamlit se teste mal de bout en bout (le rendu dépend d'un
runtime), mais ses helpers de transformation de données n'ont aucune raison
d'y échapper. L'interface elle-même a été vérifiée dans un navigateur réel
(les quatre pages, l'envoi d'une demande, l'affichage de la décision).
"""

import sys
from pathlib import Path

import pytest

# `frontend/` n'est pas un package installé (c'est un client, pas une
# bibliothèque) : on l'ajoute au chemin d'import pour ce test uniquement.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

streamlit = pytest.importorskip("streamlit", reason="frontend non installé")

from app import _liste_depuis_texte  # noqa: E402


def test_liste_simple():
    assert _liste_depuis_texte("maths, info") == ["maths", "info"]


def test_espaces_superflus_sont_retires():
    assert _liste_depuis_texte("  maths ,   info  ") == ["maths", "info"]


def test_entrees_vides_sont_ignorees():
    """Une virgule finale ou double ne doit pas produire de terme vide, qui
    serait ensuite envoyé au backend comme une déclaration réelle."""
    assert _liste_depuis_texte("maths,,info,") == ["maths", "info"]


def test_texte_vide_donne_une_liste_vide():
    assert _liste_depuis_texte("") == []
    assert _liste_depuis_texte("   ") == []


def test_la_casse_et_les_accents_sont_preserves():
    """Le frontend ne normalise volontairement pas : c'est
    `src.ml.vocabulaire` qui résout les termes et signale ce qu'il n'a pas
    reconnu. Nettoyer ici masquerait ce diagnostic."""
    assert _liste_depuis_texte("Mathématiques, Python") == ["Mathématiques", "Python"]
