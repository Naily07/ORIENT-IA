"""Socle commun aux deux espaces de l'interface (front-office et back-office).

Trois responsabilités, volontairement regroupées ici pour qu'aucune page n'ait
à les réimplémenter :

1. **Rendre `src` importable.** `frontend/` n'est pas un package et Streamlit
   est lancé depuis la racine du dépôt : sans cette insertion de chemin,
   `import src.config` échoue. C'était un défaut réel — l'onglet graphe
   (FE-4) tombait en « Graphe indisponible » sans le moindre diagnostic,
   et la mention obligatoire dépendait entièrement de l'API.
2. **Un client API qui échoue proprement.** `raise_for_status()` systématique :
   sans lui, un 500 renvoyait `{"detail": …}` que la page traitait comme une
   liste de traces, produisant un `AttributeError` brut à l'écran.
3. **La mention obligatoire (SEC-5, §16), toujours disponible.** Elle est lue
   dans `src.config`, la source unique, et non recopiée — donc affichable même
   quand l'API est injoignable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# --- 1. Chemin d'import -------------------------------------------------------
# `package-dir = backend` (pyproject.toml) : `src` n'est importable qu'avec
# backend/ sur le chemin. L'insertion est idempotente et sans effet si le
# projet a été installé en editable.
RACINE = Path(__file__).resolve().parents[1]
BACKEND = RACINE / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    from src.config import MENTION_OBLIGATOIRE

    BACKEND_IMPORTABLE = True
    RAISON_BACKEND = None
except Exception as erreur:  # noqa: BLE001 — l'interface doit démarrer quoi qu'il arrive
    # Repli explicite : le texte exact du §16, recopié uniquement si la source
    # unique est hors d'atteinte. Le bandeau signalera la dégradation, pour
    # qu'une désynchronisation ne passe jamais inaperçue.
    MENTION_OBLIGATOIRE = (
        "ORIENT'IA constitue un outil d'aide à l'orientation. Ses recommandations "
        "ne remplacent ni l'avis d'un conseiller pédagogique ni une décision "
        "officielle d'admission."
    )
    BACKEND_IMPORTABLE = False
    RAISON_BACKEND = str(erreur)


# --- 2. Client API ------------------------------------------------------------

API = os.getenv("API_URL", "http://localhost:8000")
DELAI_TRAITEMENT_S = 180  # le pipeline enchaîne plusieurs appels LLM lissés
DELAI_LECTURE_S = 15


class ApiIndisponible(RuntimeError):
    """L'API n'a pas répondu, ou a répondu autre chose que ce qui était attendu."""


def _appeler(methode: str, chemin: str, **kwargs):
    try:
        reponse = requests.request(methode, f"{API}{chemin}", **kwargs)
        reponse.raise_for_status()
        return reponse.json()
    except requests.RequestException as erreur:
        raise ApiIndisponible(str(erreur)) from erreur
    except ValueError as erreur:  # corps non-JSON malgré un 200
        raise ApiIndisponible(f"réponse illisible : {erreur}") from erreur


def api_get(chemin: str, **params):
    return _appeler("GET", chemin, params=params or None, timeout=DELAI_LECTURE_S)


def api_post(chemin: str, charge: dict, timeout: int = DELAI_TRAITEMENT_S):
    return _appeler("POST", chemin, json=charge, timeout=timeout)


@st.cache_data(ttl=30, show_spinner=False)
def etat_api() -> tuple[bool, dict | str]:
    """`(joignable, santé)` — mis en cache pour ne pas interroger l'API à
    chaque rerun de Streamlit (qui en déclenche un par interaction)."""
    try:
        return True, api_get("/health")
    except ApiIndisponible as erreur:
        return False, str(erreur)


def sante() -> dict | None:
    joignable, charge = etat_api()
    return charge if joignable and isinstance(charge, dict) else None


# --- 3. Éléments d'interface partagés -----------------------------------------


def afficher_mention_obligatoire(compacte: bool = False) -> None:
    """Mention exigée au §16, affichée **inconditionnellement**.

    Défaut corrigé : les deux emplacements précédents étaient conditionnés à
    une réponse de `/health`. API éteinte — l'état affiché par la barre
    latérale, donc un état que l'utilisateur voit réellement — et la mention
    disparaissait, alors que le sujet la rend non négociable.
    """
    if compacte:
        st.caption(f"ℹ️ {MENTION_OBLIGATOIRE}")
    else:
        st.info(f"ℹ️ **{MENTION_OBLIGATOIRE}**")


def bandeau_degradations() -> None:
    """Signale les dégradations qui changent la lecture de ce qui est affiché."""
    if not BACKEND_IMPORTABLE:
        st.warning(
            "Le paquet `src` n'est pas importable depuis cette interface "
            f"({RAISON_BACKEND}). Les vues qui lisent le corpus en local sont "
            "indisponibles, et la mention réglementaire affichée provient d'une "
            "copie de repli au lieu de `src.config`. Lancer depuis la racine du "
            "dépôt, ou installer le projet (`pip install -e .`)."
        )


def puce_etat_api() -> None:
    charge = sante()
    if charge:
        st.success("API joignable", icon="✅")
        st.caption(
            f"Modèle `{charge.get('modele', '?')}` · "
            f"clé LLM {'configurée' if charge.get('cle_llm_configuree') else '**absente**'}"
        )
    else:
        _, erreur = etat_api()
        st.error("API injoignable", icon="⛔")
        st.caption(f"`{API}` — démarrer le backend (`./run.sh`)")
        with st.expander("Détail"):
            st.code(str(erreur), language="text")


# --- Accès au back-office -----------------------------------------------------

CODE_ADMIN = os.getenv("ORIENTIA_ADMIN_CODE", "")


def exiger_acces_admin() -> None:
    """Porte d'entrée du back-office, appelée en tête de chaque page admin.

    Le contrôle vit ici plutôt que dans la construction de la navigation :
    conditionner la *liste des pages* rendait les liens profonds inopérants —
    ouvrir `/page_mesures` ou simplement recharger la page renvoyait au
    front-office, Streamlit ne connaissant pas la route dans une session neuve.

    Sans `ORIENTIA_ADMIN_CODE`, l'accès est ouvert et l'interface le dit :
    prétendre à une protection inexistante serait pire que son absence. Ce
    n'est de toute façon pas un contrôle de sécurité — le prototype ne
    manipule aucune donnée personnelle (§16).
    """
    if not CODE_ADMIN:
        st.sidebar.caption("🔓 Espace admin ouvert (aucun `ORIENTIA_ADMIN_CODE` défini).")
        return

    if st.session_state.get("admin_authentifie"):
        return

    st.title("🔒 Espace d'administration")
    st.info("Cet espace est réservé à l'équipe. Saisissez le code d'accès.")
    with st.form("acces_admin"):
        saisi = st.text_input("Code d'accès", type="password")
        if st.form_submit_button("Entrer", type="primary") and saisi:
            if saisi == CODE_ADMIN:
                st.session_state["admin_authentifie"] = True
                st.rerun()
            else:
                st.error("Code incorrect.")
    st.stop()


def charger_json_local(chemin_relatif: str) -> dict | list | None:
    """Lit un artefact d'évaluation produit par les scripts de mesure.

    Retourne `None` si le fichier n'a pas encore été généré : le back-office
    dit alors quelle commande le produit, plutôt que d'afficher un vide muet.
    """
    import json

    chemin = BACKEND / chemin_relatif
    if not chemin.exists():
        return None
    try:
        with open(chemin, encoding="utf-8") as fichier:
            return json.load(fichier)
    except (OSError, ValueError):
        return None


def artefact_absent(nom: str, commande: str) -> None:
    st.info(
        f"**{nom}** n'a pas encore été généré.\n\n"
        f"Le produire avec :\n```bash\n{commande}\n```"
    )
