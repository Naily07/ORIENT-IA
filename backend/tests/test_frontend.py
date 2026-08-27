"""Tests des fonctions pures du frontend Streamlit.

Une application Streamlit se teste mal de bout en bout (le rendu dépend d'un
runtime), mais ses helpers de transformation de données n'ont aucune raison
d'y échapper — et trois défauts trouvés en audit vivaient précisément là.

`app.py` n'est **pas** importé ici : il construit la barre latérale et la
navigation au chargement du module, ce qui n'a pas de sens hors runtime. Les
modules testés (`noyau`, `front_office`, `back_office`) n'exécutent rien à
l'import.
"""

import sys
from pathlib import Path

import pytest

# `frontend/` n'est pas un package installé (c'est un client, pas une
# bibliothèque) : on l'ajoute au chemin d'import pour ce test uniquement.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

pytest.importorskip("streamlit", reason="frontend non installé")

import noyau  # noqa: E402
from back_office import _echapper_dot  # noqa: E402
from front_office import _liste_depuis_texte  # noqa: E402

# --- Saisie du profil ---------------------------------------------------------


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


# --- Mention obligatoire (SEC-5, §16) — correctif d'audit ---------------------


def test_la_mention_obligatoire_est_disponible_sans_appel_api():
    """Non-régression : les deux emplacements d'affichage étaient conditionnés à
    une réponse de `GET /health`. API éteinte — l'état que la barre latérale
    affiche, donc un état que l'utilisateur voit réellement — et la mention
    réglementaire disparaissait de l'écran."""
    assert noyau.MENTION_OBLIGATOIRE
    assert "conseiller pédagogique" in noyau.MENTION_OBLIGATOIRE
    assert "décision officielle" in noyau.MENTION_OBLIGATOIRE


def test_la_mention_vient_de_la_source_unique_du_backend():
    """Le repli local ne doit servir qu'en dernier recours : quand `src` est
    importable, c'est `src.config` qui fait foi, sans recopie divergente."""
    from src.config import MENTION_OBLIGATOIRE

    assert noyau.BACKEND_IMPORTABLE is True
    assert noyau.MENTION_OBLIGATOIRE == MENTION_OBLIGATOIRE


# --- Client API — correctif d'audit -------------------------------------------


def test_une_erreur_http_leve_une_exception_typee(monkeypatch):
    """Non-régression : sans `raise_for_status`, un 500 renvoyait
    `{"detail": …}` que la page Observabilité traitait comme une liste de
    traces, produisant un `AttributeError` brut à l'écran."""
    import requests

    class _Reponse500:
        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error")

        def json(self):  # ne doit jamais être atteint
            return {"detail": "Internal Server Error"}

    monkeypatch.setattr(noyau.requests, "request", lambda *a, **k: _Reponse500())
    with pytest.raises(noyau.ApiIndisponible):
        noyau.api_get("/observabilite/traces")


def test_un_corps_non_json_leve_aussi_une_exception_typee(monkeypatch):
    class _ReponseHtml:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("Expecting value")

    monkeypatch.setattr(noyau.requests, "request", lambda *a, **k: _ReponseHtml())
    with pytest.raises(noyau.ApiIndisponible):
        noyau.api_get("/health")


def test_une_panne_reseau_leve_une_exception_typee(monkeypatch):
    import requests

    def _tombe(*_a, **_k):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(noyau.requests, "request", _tombe)
    with pytest.raises(noyau.ApiIndisponible):
        noyau.api_get("/health")


# --- Graphe DOT — correctif d'audit -------------------------------------------


def test_les_etiquettes_dot_sont_echappees():
    """Un guillemet dans un nom de parcours casserait le graphe entier. Le
    corpus actuel n'en contient pas, mais rien ne le garantit une fois les
    matières et compétences collectées (DATA-1)."""
    assert _echapper_dot('Génie "civil"') == 'Génie \\"civil\\"'
    assert _echapper_dot("chemin\\vers") == "chemin\\\\vers"


def test_un_nom_ordinaire_est_inchange():
    assert _echapper_dot("Informatique de Gestion") == "Informatique de Gestion"


# --- Artefacts de mesure ------------------------------------------------------


def test_un_artefact_absent_retourne_none_plutot_que_de_lever():
    assert noyau.charger_json_local("tests/fichier_qui_nexiste_pas.json") is None


def test_un_artefact_present_est_lu():
    resultats = noyau.charger_json_local("tests/eval_results_ml.json")
    assert resultats is not None
    assert "baseline_regression_logistique" in resultats
