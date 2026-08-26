"""Garde-fous en **sortie** : biais et profilage psychologique (§16 du sujet,
SEC-3, SEC-4).

Complète `guardrails.py`, qui ne couvre que l'entrée (injection, masquage).
Ici, deux détections tournées vers ce que l'assistant s'apprête à répondre :

- **Critères discriminatoires** (SEC-3) — une recommandation ne doit jamais
  se justifier par un facteur sensible (genre, origine, religion, âge,
  handicap, orientation sexuelle, situation de famille), même déclaré par
  l'utilisateur.
- **Profilage psychologique** (SEC-4) — l'assistant ne doit jamais prétendre
  déduire un trait de personnalité ou un état psychologique du style
  d'écriture de l'utilisateur.

**La défense principale est structurelle, pas ce module.** `ProfilCandidat`
et le vocabulaire de vectorisation (`src.ml.archetypes`) ne contiennent tout
simplement aucune dimension de genre, d'origine ou d'âge : le modèle ML ne
peut donc pas s'en servir, quoi que l'utilisateur déclare. Les fonctions
ci-dessous sont un **filet de sécurité heuristique** sur le texte produit par
le LLM (`resume`, `explication`, justifications) — imparfait par nature (une
détection par mots-clés sur du texte libre ne peut pas couvrir toutes les
formulations), documenté comme tel plutôt que présenté comme une garantie
absolue.
"""

import re

CRITERES_SENSIBLES: dict[str, str] = {
    "genre": r"\b(?:homme|femme|gar[çc]ons?|filles?|genre)\b",
    "origine": r"\borigine\s+(?:ethnique|nationale|sociale)\b|\bnationalit[ée]\b",
    "age": r"\bâge\b|\btrop\s+(?:jeune|âgé|vieux)\b",
    "handicap": r"\bhandicap\b|\bsitu[ée]\s+en\s+situation\s+de\s+handicap\b",
    "religion": r"\breligion\b|\bconfession\b",
    "orientation_sexuelle": r"\borientation\s+sexuelle\b",
    "situation_de_famille": r"\bsituation\s+(?:familiale|de\s+famille)\b|\bc[ée]libataire\b",
}

# Un critère sensible mentionné sans connecteur causal n'est pas en soi un
# problème (« le candidat, âgé de 19 ans, souhaite... » est neutre) : ce qui
# doit déclencher l'escalade, c'est le critère *utilisé comme justification*.
CONNECTEURS_CAUSAUX = re.compile(
    r"\bparce\s+qu[e']|\bcar\b|\b[ée]tant\s+donn[ée]\s+(?:que|qu')|"
    r"\ben\s+raison\s+d[e']|\bpuisqu[e']|\bdu\s+fait\s+d[e']",
    re.IGNORECASE,
)

_MOTIFS_CRITERES = {
    nom: re.compile(motif, re.IGNORECASE) for nom, motif in CRITERES_SENSIBLES.items()
}

_SEPARATEUR_PHRASES = re.compile(r"(?<=[.!?\n])\s+")


def detecter_criteres_discriminatoires(texte: str) -> dict:
    """Vrai si un critère sensible apparaît comme justification causale dans
    le texte (même phrase qu'un connecteur du type « parce que », « car »...).

    Retourne `{"detecte": bool, "criteres": list[str], "phrase": str | None}`.
    """
    if not texte:
        return {"detecte": False, "criteres": [], "phrase": None}

    for phrase in _SEPARATEUR_PHRASES.split(texte):
        if not CONNECTEURS_CAUSAUX.search(phrase):
            continue
        criteres = [nom for nom, motif in _MOTIFS_CRITERES.items() if motif.search(phrase)]
        if criteres:
            return {"detecte": True, "criteres": criteres, "phrase": phrase.strip()}

    return {"detecte": False, "criteres": [], "phrase": None}


_MOTIFS_PROFILAGE_PSYCHOLOGIQUE = re.compile(
    r"\bprofil\s+psychologique\b|"
    r"\btrait[s]?\s+de\s+(?:personnalit[ée]|caract[èe]re)\b|"
    r"\bvotre\s+personnalit[ée]\b|"
    r"\bvous\s+(?:semblez|paraissez)\s+(?:être\s+)?(?:quelqu['’]un|une?\s+personne)\b|"
    r"\bvotre\s+(?:style\s+d['’]écriture|fa[çc]on\s+d['’]écrire)\s+(?:sugg[èe]re|montre|r[ée]v[èe]le)\b|"
    r"\bon\s+peut\s+d[ée]duire\s+de\s+votre\b",
    re.IGNORECASE,
)


def detecter_profilage_psychologique(texte: str) -> dict:
    """Vrai si le texte semble déduire un trait psychologique de
    l'utilisateur plutôt que de s'appuyer sur ses préférences déclarées.

    Retourne `{"detecte": bool, "extrait": str | None}`.
    """
    if not texte:
        return {"detecte": False, "extrait": None}

    trouve = _MOTIFS_PROFILAGE_PSYCHOLOGIQUE.search(texte)
    if trouve is None:
        return {"detecte": False, "extrait": None}
    return {"detecte": True, "extrait": trouve.group(0)}


def verifier_sortie(*textes: str) -> dict:
    """Applique les deux détections à un ensemble de textes (résumé,
    explication, justifications...) et agrège le verdict.

    Retourne `{"danger": bool, "raison": str | None}`, dans le même format
    que `guardrails.check_injection()` pour rester facile à brancher dans
    l'orchestrateur."""
    for texte in textes:
        discrimination = detecter_criteres_discriminatoires(texte)
        if discrimination["detecte"]:
            return {
                "danger": True,
                "raison": (
                    f"Critère(s) sensible(s) utilisé(s) comme justification : "
                    f"{', '.join(discrimination['criteres'])} — phrase : "
                    f"« {discrimination['phrase']} »"
                ),
            }
        profilage = detecter_profilage_psychologique(texte)
        if profilage["detecte"]:
            return {
                "danger": True,
                "raison": (
                    f"Langage de profilage psychologique détecté : « {profilage['extrait']} »"
                ),
            }
    return {"danger": False, "raison": None}
