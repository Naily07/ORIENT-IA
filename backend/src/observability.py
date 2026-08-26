"""Observabilité — traces, appels d'outils et appels LLM.

Trois journaux JSONL indépendants, un événement par ligne (correspond à
l'exigence « traces à examiner » du sujet ORIENT'IA, §15) :

- `traces.jsonl`     : une entrée par requête traitée par le pipeline —
  décision finale, latence globale. Alimente `GET /observabilite/traces`.
- `tool_calls.jsonl` : une entrée par appel d'outil de l'agent, branché via
  un hook `set_log_appel(log_tool_call)` exposé par le futur module
  `tools.py` (pas de dépendance directe pour éviter un cycle d'imports).
- `llm_calls.jsonl`  : une entrée par appel au modèle génératif — prompts et
  réponses bruts. Branché directement dans `llm_call()` et
  `llm_call_with_tools()`, le point de passage unique du pipeline.

Format JSON Lines : un fichier qui grossit en continu, consultable sans base
de données. Toute valeur passe par `guardrails.masquer_objet()` avant
écriture — un profil ou un message libre peut contenir une donnée sensible en
clair, elle ne doit pas se retrouver telle quelle dans des logs relus pendant
la démo.

Repris d'un hackathon ISPM précédent : mécanisme entièrement
domaine-agnostique. `log_trace()` est volontairement générique (description +
contexte + décision + champs libres) plutôt que typé sur un schéma métier
particulier, puisque le schéma de décision d'ORIENT'IA n'est pas encore figé.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import config
from src.guardrails import masquer_objet

# --- Écriture ------------------------------------------------------------


def _horodatage() -> str:
    return datetime.now(UTC).isoformat()


def _ecrire_jsonl(fichier: Path, evenement: dict[str, Any]) -> None:
    """Ajoute une ligne JSON à un fichier JSONL, en créant le dossier si besoin.

    N'échoue jamais bruyamment : un souci d'écriture de log ne doit pas faire
    échouer le traitement d'une requête, l'observabilité est un effet de
    bord, pas une dépendance dure du pipeline.
    """
    try:
        fichier.parent.mkdir(parents=True, exist_ok=True)
        with open(fichier, "a", encoding="utf-8") as f:
            f.write(json.dumps(masquer_objet(evenement), ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def _lire_dernieres_lignes(fichier: Path, limite: int) -> list[dict[str, Any]]:
    if not fichier.exists():
        return []
    with open(fichier, encoding="utf-8") as f:
        lignes = [json.loads(ligne) for ligne in f if ligne.strip()]
    return list(reversed(lignes[-limite:]))


# --- Traces ------------------------------------------------------


def _valeur_serialisable(decision: Any) -> Any:
    if hasattr(decision, "model_dump"):
        return decision.model_dump()
    return decision


def log_trace(
    trace_id: str,
    description: str | None,
    contexte: list[dict[str, Any]] | None,
    decision: Any,
    latence_ms: float,
    **champs_supplementaires: Any,
) -> None:
    """Journalise le traitement complet d'une requête.

    `champs_supplementaires` accueille ce que l'orchestrateur voudra ajouter
    au fil de son développement (profil construit, résultat de
    classification/ML, scores de recherche...) sans que ce module ait besoin
    de connaître ces schémas à l'avance.
    """
    _ecrire_jsonl(
        config.fichier_traces,
        {
            "horodatage": _horodatage(),
            "trace_id": trace_id,
            "description": description,
            "nb_documents_contexte": len(contexte) if contexte else 0,
            "decision": _valeur_serialisable(decision),
            "latence_ms": round(latence_ms, 1),
            **{cle: _valeur_serialisable(valeur) for cle, valeur in champs_supplementaires.items()},
        },
    )


def lire_dernieres_traces(limite: int = 50) -> list[dict[str, Any]]:
    """Relit les `limite` dernières traces, les plus récentes en premier.
    Consommé par `GET /observabilite/traces`."""
    return _lire_dernieres_lignes(config.fichier_traces, limite)


# --- Appels d'outils -----------------------------------------------------
# Signature imposée par le futur `tools.set_log_appel()` : ne pas la changer
# sans mettre à jour l'appel correspondant dans `api.py`.


def log_tool_call(
    trace_id: str,
    nom: str,
    params: dict[str, Any],
    resultat: Any,
    statut: str,
    latence_ms: float,
) -> None:
    _ecrire_jsonl(
        config.fichier_tool_calls,
        {
            "horodatage": _horodatage(),
            "trace_id": trace_id,
            "outil": nom,
            "params": params,
            "resultat": resultat,
            "statut": statut,
            "latence_ms": latence_ms,
        },
    )


def lire_derniers_appels_outils(limite: int = 50) -> list[dict[str, Any]]:
    return _lire_dernieres_lignes(config.fichier_tool_calls, limite)


# --- Coût estimé ----------------------------------------------------------
# Tarifs approximatifs (USD / million de tokens), volontairement grossiers :
# un ordre de grandeur exploitable, pas une facture exacte.

COUT_PAR_MILLION_TOKENS_ENTREE_USD = 0.10
COUT_PAR_MILLION_TOKENS_SORTIE_USD = 0.40


def estimer_cout(tokens_entree: int | None, tokens_sortie: int | None) -> float | None:
    if tokens_entree is None or tokens_sortie is None:
        return None
    return round(
        tokens_entree / 1_000_000 * COUT_PAR_MILLION_TOKENS_ENTREE_USD
        + tokens_sortie / 1_000_000 * COUT_PAR_MILLION_TOKENS_SORTIE_USD,
        6,
    )


# --- Appels LLM ------------------------------------------------------------


def log_llm_call(
    prompt_systeme: str,
    contenu: Any,
    reponse_texte: str,
    modele: str,
    latence_ms: float,
    tokens_entree: int | None = None,
    tokens_sortie: int | None = None,
    etape: str = "non_precisee",
    trace_id: str | None = None,
) -> None:
    """Journalise un appel brut au modèle génératif — prompt et réponse tels
    quels, exigé par le sujet (§15 : « les entrées et sorties du modèle ML »).

    `trace_id` est optionnel : sa propagation jusqu'ici (pour relier un appel
    LLM à la requête qui l'a déclenché) est le travail de l'orchestrateur, pas
    encore construit. En attendant, `etape` (profil/rag/agent/...) identifie
    au moins la provenance de l'appel.
    """
    _ecrire_jsonl(
        config.fichier_llm_calls,
        {
            "horodatage": _horodatage(),
            "trace_id": trace_id,
            "etape": etape,
            "modele": modele,
            "prompt_systeme": prompt_systeme,
            "contenu": contenu,
            "reponse": reponse_texte,
            "tokens_entree": tokens_entree,
            "tokens_sortie": tokens_sortie,
            "cout_estime_usd": estimer_cout(tokens_entree, tokens_sortie),
            "latence_ms": round(latence_ms, 1),
        },
    )


def lire_derniers_appels_llm(limite: int = 50) -> list[dict[str, Any]]:
    return _lire_dernieres_lignes(config.fichier_llm_calls, limite)


class ChronoLatence:
    """Petit chronomètre pour mesurer une latence en millisecondes.

    `with ChronoLatence() as chrono: ...` puis `chrono.ms` — évite de
    dupliquer `time.time()` avant/après à chaque site d'appel.
    """

    def __enter__(self) -> "ChronoLatence":
        self._debut = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.ms = (time.perf_counter() - self._debut) * 1000
