"""Point d'appel unique vers le LLM (Google AI Studio / Gemini API).

Tout le pipeline (extraction de profil, RAG, agent, garde-fous) passe par
`llm_call` / `llm_call_with_tools`. Ce point de passage unique est ce qui
permet de brancher l'observabilité des prompts en un seul endroit plutôt que
sur chaque site d'appel.

Repris tel quel de l'infrastructure d'un hackathon ISPM précédent : ce module
est entièrement générique, sans logique métier d'orientation pédagogique.
"""

import logging
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.config import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Échec d'appel au LLM, déjà traduit en erreur métier.

    L'orchestrateur l'attrape pour dégrader proprement plutôt que de laisser
    remonter une 500 nue.
    """


class QuotaDepasseError(LLMError):
    """Quota Gemini épuisé malgré les tentatives de réémission."""


class SchemaNonConforme(LLMError):
    """Le modèle a répondu, mais sans respecter le `response_schema` demandé.

    Distinct de `QuotaDepasseError` et des pannes réseau : celles-ci ne se
    résolvent pas en réessayant le même prompt, alors qu'une non-conformité
    de schéma peut se corriger par une régénération (voir `sortie.py`, qui
    n'attrape que ce cas précis — pas les erreurs réseau, qui doivent
    remonter immédiatement).
    """


class BudgetTempsDepasse(LLMError):
    """Le budget global de la requête ne permet plus un appel LLM.

    Distinct d'un timeout isolé : réessayer ne ferait qu'aggraver le
    dépassement. L'orchestrateur le traite comme tout ``LLMError`` et produit
    sa réponse de repli contrôlée.
    """


_echeance_llm: ContextVar[float | None] = ContextVar("echeance_llm", default=None)


@contextmanager
def limiter_temps_llm(duree_s: float, *, depart: float | None = None):
    """Applique un budget cumulé à tous les appels LLM du contexte courant.

    ``ContextVar`` isole l'échéance de chaque requête FastAPI, y compris
    lorsque plusieurs requêtes synchrones sont servies dans des threads
    différents. Le point de départ peut être celui de l'orchestrateur afin
    que le garde-fou, le RAG et l'agent partagent exactement le même budget.
    """
    origine = time.monotonic() if depart is None else depart
    jeton = _echeance_llm.set(origine + max(0.0, duree_s))
    try:
        yield
    finally:
        _echeance_llm.reset(jeton)


def _temps_restant_s() -> float | None:
    echeance = _echeance_llm.get()
    if echeance is None:
        return None
    return max(0.0, echeance - time.monotonic())


def _exiger_temps_restant() -> float | None:
    restant = _temps_restant_s()
    if restant is not None and restant <= 0:
        raise BudgetTempsDepasse("Budget de temps global dépassé avant l'appel LLM")
    return restant


def _dormir_dans_budget(duree_s: float) -> None:
    restant = _exiger_temps_restant()
    if restant is not None and duree_s >= restant:
        raise BudgetTempsDepasse(
            f"Budget de temps global insuffisant pour attendre {duree_s:.1f} s"
        )
    time.sleep(duree_s)


def _delai_suggere(message: str) -> float | None:
    """Extrait le délai que l'API elle-même recommande d'attendre.

    Gemini renvoie « Please retry in 5.79s » dans le corps d'une 429 :
    respecter ce délai est plus efficace qu'un backoff exponentiel aveugle.
    """
    correspondance = re.search(r"retry in (\d+(?:\.\d+)?)s", message)
    return float(correspondance.group(1)) if correspondance else None


@lru_cache(maxsize=32)
def _get_client(timeout_ms: int | None = None) -> genai.Client:
    # Construction différée : genai.Client() exige une clé valide dès
    # l'instanciation, donc la construire au chargement du module
    # empêcherait d'importer src.llm_client tant que GEMINI_API_KEY n'est pas
    # configurée — y compris pour les tests qui n'appellent jamais le réseau.
    if not config.gemini_api_key:
        raise LLMError(
            "GEMINI_API_KEY absente. Copier .env.example vers .env et y coller "
            "la clé obtenue sur https://aistudio.google.com/apikey"
        )
    delai_ms = timeout_ms or int(config.llm_timeout_s * 1000)
    return genai.Client(
        api_key=config.gemini_api_key,
        # Borne haute sur chaque appel : sans elle, une API qui ne répond pas
        # fige la requête FastAPI et la réponse dégradée ne part jamais. Le
        # SDK attend des millisecondes.
        http_options=types.HttpOptions(timeout=delai_ms),
    )


_dernier_appel: float = 0.0
_verrou_debit = threading.Lock()


def _attendre_son_tour() -> None:
    """Espace les appels pour rester sous la limite de requêtes par minute.

    Subir une 429 coûte le délai de reprise imposé par l'API (parfois plus
    d'une minute) ; attendre quelques secondes en amont est nettement moins
    cher. Sérialisé par un verrou car FastAPI sert les requêtes en parallèle.
    """
    if config.llm_requetes_par_minute <= 0:
        return

    intervalle = 60.0 / config.llm_requetes_par_minute
    global _dernier_appel
    restant = _exiger_temps_restant()
    if restant is None:
        verrou_acquis = _verrou_debit.acquire()
    else:
        verrou_acquis = _verrou_debit.acquire(timeout=restant)
    if not verrou_acquis:
        raise BudgetTempsDepasse("Budget de temps global dépassé en attendant le débit LLM")
    try:
        attente = intervalle - (time.monotonic() - _dernier_appel)
        if attente > 0:
            _dormir_dans_budget(attente)
        _dernier_appel = time.monotonic()
    finally:
        _verrou_debit.release()


# Motifs d'erreur transitoires, qui se résolvent en général en réessayant le
# même appel : quota par minute dépassé, ou timeout/indisponibilité côté
# serveur Gemini. Trouvé en usage réel (AGT-1) : un appel dans une boucle
# d'agent avec un historique de conversation qui grossit (réponses d'outils
# incluses) peut dépasser le délai interne de l'API (504 DEADLINE_EXCEEDED)
# sans qu'aucun quota ne soit en cause — le distinguer permet de réessayer
# plutôt que d'abandonner sur ce qui n'est qu'un aléa réseau.
_MOTIFS_QUOTA = ("RESOURCE_EXHAUSTED", "429")
_MOTIFS_SERVEUR_TRANSITOIRE = ("DEADLINE_EXCEEDED", "UNAVAILABLE", "503", "504")


def _appeler_avec_reprise(contenu: str, parametres: types.GenerateContentConfig):
    """Appelle l'API en absorbant les erreurs transitoires (quota par minute
    ou indisponibilité passagère du serveur).

    Le Free Tier plafonne à 15 requêtes/minute (valeur constatée). Or le
    pipeline émet plusieurs appels par requête : sans reprise, une démo
    enchaînant quelques interactions déclenche une 429 en pleine
    présentation. On réémet en respectant le délai indiqué par l'API si elle
    en fournit un (cas du quota), sinon un délai par défaut (cas d'un
    timeout serveur, qui n'indique jamais de délai de reprise).
    """
    derniere_erreur: Exception | None = None

    for tentative in range(config.llm_max_tentatives):
        try:
            _attendre_son_tour()
            restant = _exiger_temps_restant()
            timeout_s = config.llm_timeout_s if restant is None else min(
                config.llm_timeout_s, restant
            )
            # Le SDK attend des millisecondes entières. Un minimum de 1 ms
            # conserve une borne valide quand l'échéance est imminente.
            timeout_ms = max(1, int(timeout_s * 1000))
            return _get_client(timeout_ms).models.generate_content(
                model=config.gemini_model,
                contents=contenu,
                config=parametres,
            )
        except BudgetTempsDepasse:
            raise
        except Exception as e:
            message = str(e)
            reessayable = any(
                motif in message for motif in (*_MOTIFS_QUOTA, *_MOTIFS_SERVEUR_TRANSITOIRE)
            )
            if not reessayable:
                raise LLMError(f"Appel LLM échoué ({type(e).__name__}) : {e}") from e

            derniere_erreur = e
            if tentative == config.llm_max_tentatives - 1:
                break

            attente = _delai_suggere(message) or config.llm_attente_quota
            attente += 0.5  # marge : la fenêtre de quota est glissante
            logger.warning(
                "Erreur transitoire (tentative %d/%d), reprise dans %.1fs : %s",
                tentative + 1,
                config.llm_max_tentatives,
                attente,
                message,
            )
            _dormir_dans_budget(attente)

    raise QuotaDepasseError(
        f"Appel LLM toujours en échec après {config.llm_max_tentatives} tentatives "
        f"(quota Free Tier ~15 requêtes/minute, ou indisponibilité serveur "
        f"persistante). Détail : {derniere_erreur}"
    ) from derniere_erreur


# --- Observabilité des appels LLM -------------------------------------------
# Hook plutôt qu'import direct : `src.observability` importe `src.guardrails`
# (masquage des secrets avant écriture), qui importerait `src.llm_client` en
# cas d'import direct — cycle. Branché au démarrage de l'API.

_log_llm_call: Any = None


def set_log_llm_call(fonction) -> None:
    """Branche le logger d'appels LLM. Signature attendue :
    `fonction(prompt_systeme, contenu, reponse_texte, modele, latence_ms,
    tokens_entree=None, tokens_sortie=None, etape=str, trace_id=None)`."""
    global _log_llm_call
    _log_llm_call = fonction


def _texte_ou_appels(reponse) -> str:
    """Texte de la réponse, ou un résumé des outils appelés si la réponse ne
    contient que des `function_call` (rien à journaliser dans `.text` dans ce
    cas)."""
    if reponse.text:
        return reponse.text
    if reponse.candidates:
        appels = [
            partie.function_call.name
            for partie in reponse.candidates[0].content.parts
            if partie.function_call is not None
        ]
        if appels:
            return f"[appels d'outils] {appels}"
    return ""


def _journaliser(
    reponse,
    prompt_systeme: str,
    contenu,
    latence_ms: float,
    etape: str,
    trace_id: str | None,
) -> None:
    if _log_llm_call is None:
        return
    usage = getattr(reponse, "usage_metadata", None)
    try:
        _log_llm_call(
            prompt_systeme=prompt_systeme,
            contenu=contenu,
            reponse_texte=_texte_ou_appels(reponse),
            modele=config.gemini_model,
            latence_ms=latence_ms,
            tokens_entree=getattr(usage, "prompt_token_count", None),
            tokens_sortie=getattr(usage, "candidates_token_count", None),
            etape=etape,
            trace_id=trace_id,
        )
    except Exception:
        logger.warning("Échec du log d'appel LLM, ignoré", exc_info=True)


def llm_call(
    prompt_systeme: str,
    prompt_utilisateur: str,
    response_schema: type[BaseModel] | None = None,
    *,
    etape: str = "non_precisee",
    trace_id: str | None = None,
) -> BaseModel | str:
    """Appelle le LLM et retourne soit du texte, soit un objet Pydantic validé.

    Quand `response_schema` est fourni, on s'appuie sur le mode JSON natif de
    l'API Gemini (`response_mime_type` + `response_schema`) plutôt que sur du
    parsing manuel : le modèle est contraint côté serveur, et `.parsed` rend
    directement une instance Pydantic déjà validée.

    `etape`/`trace_id` sont facultatifs et purement déclaratifs pour le
    logger d'appels LLM : aucun appelant existant n'a besoin d'être modifié.
    """
    parametres = types.GenerateContentConfig(
        system_instruction=prompt_systeme,
        temperature=config.llm_temperature,
        max_output_tokens=config.llm_max_output_tokens,
    )
    if response_schema is not None:
        parametres.response_mime_type = "application/json"
        parametres.response_schema = response_schema

    debut = time.perf_counter()
    reponse = _appeler_avec_reprise(prompt_utilisateur, parametres)
    latence_ms = (time.perf_counter() - debut) * 1000
    _journaliser(reponse, prompt_systeme, prompt_utilisateur, latence_ms, etape, trace_id)

    if response_schema is None:
        return reponse.text or ""

    resultat = reponse.parsed
    if resultat is None:
        raise SchemaNonConforme(
            f"Le modèle n'a pas produit de JSON conforme à {response_schema.__name__}. "
            f"Réponse brute : {(reponse.text or '')[:200]}"
        )
    return resultat


def llm_call_with_tools(
    messages: list,
    prompt_systeme: str,
    tools: list,
    response_schema: type[BaseModel] | None = None,
    *,
    etape: str = "agent",
    trace_id: str | None = None,
):
    """Appel LLM avec function calling.

    `messages` est l'historique au format Gemini (liste de `types.Content`,
    incluant les `FunctionCall`/`FunctionResponse` des itérations
    précédentes). La reprise sur quota et le lissage de débit sont les mêmes
    que `llm_call` : on passe par le même point d'appel.

    Quand `response_schema` est fourni, le modèle est contraint en JSON : si
    la réponse est une réponse finale (pas d'appel d'outil), `reponse.parsed`
    contient directement une instance Pydantic validée.
    """
    parametres = types.GenerateContentConfig(
        system_instruction=prompt_systeme,
        temperature=config.llm_temperature,
        max_output_tokens=config.llm_max_output_tokens,
        tools=tools,
    )
    if response_schema is not None:
        parametres.response_mime_type = "application/json"
        parametres.response_schema = response_schema

    debut = time.perf_counter()
    reponse = _appeler_avec_reprise(messages, parametres)
    latence_ms = (time.perf_counter() - debut) * 1000
    _journaliser(reponse, prompt_systeme, messages, latence_ms, etape, trace_id)
    return reponse
