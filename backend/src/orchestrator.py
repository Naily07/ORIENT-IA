"""Orchestrateur du pipeline ORIENT'IA (ORCH-1, ORCH-2, ORCH-3).

Enchaîne garde-fous d'entrée → recherche documentaire (RAG) → agent → sortie
structurée, sur les deux principes déjà appliqués dans EXAM-S2 :

1. **Une seule étape est bloquante : les garde-fous d'entrée.** Une tentative
   de manipulation détectée court-circuite tout le reste — aucun outil
   appelé, aucune recommandation générée, aucun contexte documentaire
   transmis à l'agent ni exposé dans la décision. La recherche documentaire
   est *optionnelle* : son échec dégrade la décision (moins de contexte,
   confiance plafonnée) plutôt que d'interrompre le traitement. **Correctif
   d'audit P1/T8** : garde-fous et recherche documentaire s'exécutent
   désormais en parallèle (`_executer_en_parallele`), la seconde ne
   dépendant pas du verdict de la première. Sur un message dangereux, le
   contexte RAG calculé en parallèle est un travail purement local (aucun
   appel réseau, aucune écriture) simplement jeté sans être utilisé — la
   garantie porte sur ce qui atteint l'agent et la décision, pas sur le fait
   qu'aucun calcul local n'ait jamais été tenté.
2. **Le dernier mot revient au code.** Une décision produite avec un
   pipeline amputé (RAG indisponible, budget de temps dépassé) ne peut pas
   afficher la même certitude qu'une décision complète : sa confiance est
   plafonnée et son incertitude déclarée, quoi que l'agent ait renvoyé. Le
   texte produit par l'agent est en plus scanné (SEC-3, SEC-4) : un critère
   discriminatoire utilisé comme justification, ou un langage de profilage
   psychologique, force une escalade vers un conseiller — cette
   responsabilité ne peut pas rester une simple consigne de prompt (§16 du
   sujet, non négociable).

`traiter_demande()` ne lève jamais : chaque échec possible a une réponse
dégradée mais valide (§2 du sujet EXAM-S2, gestion d'erreurs — même exigence
implicite pour ORIENT'IA : ne jamais renvoyer une erreur nue à l'utilisateur).
"""

import contextvars
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent import run_agent
from src.config import config
from src.guardrails import check_injection, masquer_donnees_sensibles
from src.llm_client import LLMError, limiter_temps_llm
from src.rag import retrieve_context
from src.schemas import OrientationInput, OrientationReponse, RecommandationDecision
from src.securite import verifier_sortie

logger = logging.getLogger(__name__)

# --- Hook d'observabilité -----------------------------------------------------
# Même mécanisme que `tools.set_log_appel` / `llm_client.set_log_llm_call` :
# évite une dépendance directe vers un module d'observabilité au chargement.

_log_trace: Any = None


def set_log_trace(fonction) -> None:
    """Branche l'écriture des traces. Signature attendue :
    `fonction(trace_id, message, contexte, decision, latence_ms, **extra)`."""
    global _log_trace
    _log_trace = fonction


# --- Budget de temps (ORCH-3) -------------------------------------------------


def _budget_epuise(depart: float) -> bool:
    return (time.monotonic() - depart) > config.orchestrateur_budget_s


def _etape_optionnelle(nom: str, fonction, defaut, depart: float) -> tuple[Any, str | None]:
    """Exécute une étape dont l'échec dégrade la décision sans l'interrompre.

    Retourne `(resultat, degradation)` : `degradation` est `None` si l'étape
    a réussi, sinon une phrase courte destinée à l'explication et aux logs.
    """
    if _budget_epuise(depart):
        return defaut, f"{nom} sautée (budget de {config.orchestrateur_budget_s:.0f} s dépassé)"
    try:
        return fonction(), None
    except Exception as e:  # noqa: BLE001 — aucune étape optionnelle ne doit faire échouer la demande
        logger.warning("Étape « %s » indisponible : %s", nom, e)
        return defaut, f"{nom} indisponible ({type(e).__name__})"


# --- Exécution en parallèle des étapes indépendantes (correctif d'audit P1/T8) -
#
# La vérification anti-injection (couche LLM, quand la couche mots-clés n'a
# rien trouvé) et la recherche documentaire n'ont aucune dépendance l'une sur
# l'autre : la seconde ne lit jamais le verdict de la première avant de
# s'exécuter. Jusqu'ici, elles s'enchaînaient malgré tout en série — un aller-
# retour LLM complet (≈1 à 5 s) avant même de lancer une recherche qui, elle,
# ne fait aucun appel réseau (embeddings ONNX locaux + Chroma). Les exécuter
# en parallèle ne change aucun résultat, seulement la latence perçue.
_executeur_parallele = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="orient-ia-orchestrateur"
)


def _executer_en_parallele(*fonctions):
    """Exécute des fonctions sans argument en parallèle et retourne leurs
    résultats dans l'ordre où elles ont été passées.

    Chaque fonction reçoit sa **propre** copie du contexte (`ContextVar`) de
    l'appelant, prise avant la soumission au pool. C'est nécessaire pour que
    `limiter_temps_llm` (le budget de temps global, ORCH-5) reste visible
    dans les threads du pool — un thread démarré sans cette copie explicite
    n'hérite d'aucune `ContextVar`, et un appel LLM lancé depuis ce thread ne
    verrait plus jamais son échéance. Une copie *par fonction*, pas une copie
    partagée : `Context.run()` refuse d'être appelé depuis deux threads à la
    fois sur le même objet contexte.

    Ne relève aucune exception : les fonctions passées ici gèrent déjà leurs
    propres échecs (voir `_etape_optionnelle` et `_verifier_injection_entree`)
    et ne sont censées jamais lever — comme c'était déjà le cas quand ces
    mêmes appels s'enchaînaient en série.
    """
    futures = [
        _executeur_parallele.submit(contextvars.copy_context().run, fonction)
        for fonction in fonctions
    ]
    return [future.result() for future in futures]


def _verifier_injection_entree(message: str, trace_id: str) -> tuple[dict, str | None]:
    """Garde-fou d'entrée : verdict anti-injection et dégradation éventuelle.

    Séparé de `_traiter_demande_dans_budget` pour pouvoir s'exécuter dans le
    pool de `_executer_en_parallele` sans muter d'état partagé (la liste
    `degradations` de l'appelant) depuis un autre thread — le résultat est
    retourné, jamais accumulé sur place.
    """
    try:
        return check_injection(message), None
    except Exception as e:  # noqa: BLE001 — check_injection absorbe déjà LLMError
        logger.exception("Garde-fous d'entrée en échec (trace_id=%s)", trace_id)
        risque = {"danger": False, "raison": None, "couche": None, "verification_llm": "erreur"}
        return risque, f"garde-fous d'entrée dégradés ({type(e).__name__})"


# --- Décisions de repli, toujours valides ------------------------------------


def _escalade_injection(message: str, risque: dict) -> RecommandationDecision:
    """Décision produite pour une demande détectée comme malveillante.

    Court-circuite tout le pipeline : aucun outil appelé, aucune
    recommandation générée. `confiance=1.0` ne porte pas sur un diagnostic
    d'orientation mais sur la décision elle-même — transmettre à un humain
    est l'issue sûre, sans ambiguïté."""
    raison = masquer_donnees_sensibles(
        risque.get("raison") or "contenu signalé comme tentative de manipulation"
    )
    extrait = masquer_donnees_sensibles(message or "").strip()[:200]
    return RecommandationDecision(
        resume=(
            "Demande non traitée automatiquement : tentative de manipulation "
            f"détectée. Extrait reçu : « {extrait} »"
        ),
        reponse=(
            "Je ne peux pas traiter cette demande : elle contient des consignes "
            "qui cherchent à modifier mon fonctionnement. Posez-moi votre "
            "question d'orientation directement et j'y répondrai."
        ),
        parcours_recommandes=[],
        confiance=1.0,
        informations_manquantes=[],
        explication=(
            f"Tentative de manipulation détectée : {raison}. Aucun outil n'a "
            "été appelé et aucune recommandation n'a été générée."
        ),
        sources=[],
        outils_utilises=[],
        action="escalade_conseiller",
        incertitude_declaree=True,
    )


def _decision_repli(motif: str) -> RecommandationDecision:
    """Décision de repli quand l'agent n'a pas pu conclure — jamais d'erreur
    nue vers l'utilisateur. `motif` est masqué (SEC-2) : il peut recopier un
    extrait brut d'une réponse LLM, potentiellement le message de
    l'utilisateur lui-même."""
    return RecommandationDecision(
        resume="Une erreur technique est survenue ; votre demande est transmise à un conseiller.",
        reponse=(
            "Désolé, un problème technique m'a empêché de traiter votre demande. "
            "Réessayez dans un instant, ou adressez-vous à un conseiller "
            "pédagogique de l'ISPM si cela persiste."
        ),
        parcours_recommandes=[],
        confiance=0.0,
        informations_manquantes=[],
        explication=masquer_donnees_sensibles(motif),
        sources=[],
        outils_utilises=[],
        action="escalade_conseiller",
        incertitude_declaree=True,
    )


def _appliquer_controle_de_sortie(decision: RecommandationDecision) -> RecommandationDecision:
    """Scanne le texte produit par l'agent (SEC-3, SEC-4) : un critère
    discriminatoire utilisé comme justification, ou un langage de profilage
    psychologique, force une escalade vers un conseiller humain.

    Vérifié en dernier, sur la décision qui sera effectivement montrée à
    l'utilisateur — après le plafonnement de confiance, pas avant, pour que
    la raison de l'escalade reste celle réellement en cause plutôt qu'être
    masquée par une dégradation de pipeline sans rapport."""
    textes = [decision.resume, decision.reponse, decision.explication] + [
        p.justification for p in decision.parcours_recommandes
    ]
    verdict = verifier_sortie(*textes)
    if not verdict["danger"]:
        return decision

    return decision.model_copy(
        update={
            "action": "escalade_conseiller",
            "incertitude_declaree": True,
            # Le texte produit contient un motif signalé (critère sensible,
            # profilage) : on ne le montre pas tel quel à l'utilisateur, on le
            # remplace par un renvoi neutre vers un conseiller.
            "reponse": (
                "Je préfère ne pas répondre en l'état : cette recommandation doit "
                "être revue par un conseiller pédagogique de l'ISPM avant d'être "
                "présentée. Adressez-vous à lui pour un accompagnement fiable."
            ),
            "explication": (
                f"{decision.explication}\n[Contrôle automatique] Réponse retenue pour "
                f"revue humaine : {verdict['raison']}."
            ),
        }
    )


def _appliquer_plafond_de_confiance(
    decision: RecommandationDecision, degradations: list[str]
) -> RecommandationDecision:
    """Contresigne la décision de l'agent : une dégradation plafonne la
    confiance et force la déclaration d'incertitude, quoi que l'agent ait
    produit de son côté."""
    if not degradations:
        return decision
    confiance = min(decision.confiance, config.orchestrateur_seuil_confiance)
    explication = (
        f"{decision.explication}\n[Contrôle automatique] Réponse produite avec un "
        f"pipeline dégradé : {' ; '.join(degradations)}."
    )
    return decision.model_copy(
        update={"confiance": confiance, "incertitude_declaree": True, "explication": explication}
    )


def _finaliser(
    trace_id: str,
    entree: OrientationInput,
    contexte: list[dict] | None,
    decision: RecommandationDecision,
    depart: float,
) -> OrientationReponse:
    latence_ms = round((time.monotonic() - depart) * 1000)
    if _log_trace is not None:
        try:
            _log_trace(
                trace_id,
                entree.message,
                contexte,
                decision,
                latence_ms,
                profil=entree.profil,
            )
        except Exception:  # noqa: BLE001 — une trace illisible ne doit pas faire perdre la décision
            logger.exception("Écriture de trace impossible (trace_id=%s)", trace_id)
    return OrientationReponse(trace_id=trace_id, decision=decision)


# --- Pipeline complet (ORCH-1) -----------------------------------------------


def _historique_sain(entree: OrientationInput) -> tuple[list, str | None]:
    """Écarte de l'historique rejoué les tours dont la question porte un motif
    d'injection, plutôt que de bloquer toute la requête.

    L'historique vient du client : rien n'empêche un appelant d'y glisser une
    consigne de manipulation qui n'apparaîtrait jamais dans `message` (§11). Le
    premier correctif contrôlait donc message **et** historique concaténés —
    mais une conversation réelle où l'utilisateur a tapé « ignore les
    documents… » au tour 3 (tentative détectée et refusée à ce moment-là) se
    retrouvait ensuite bloquée à *tous* les tours suivants, la question fautive
    restant dans l'historique rejoué. Observé en démonstration.

    On filtre plutôt : une question d'historique qui trip la couche mots-clés
    (déterministe, sans appel LLM) n'est simplement pas rejouée à l'agent. Le
    message courant, lui, reste contrôlé en entier par `check_injection`.

    Seules les **questions** sont examinées, pas les réponses : celles-ci sont
    notre propre prose, déjà filtrée par `verifier_sortie`.
    """
    sains = []
    retires = 0
    for tour in entree.historique:
        if tour.question.strip() and check_injection(tour.question, avec_llm=False)["danger"]:
            retires += 1
            continue
        sains.append(tour)
    note = (
        f"{retires} tour(s) d'historique écarté(s) (motif d'injection)" if retires else None
    )
    return sains, note


def traiter_demande(entree: OrientationInput) -> OrientationReponse:
    """Traite une demande de bout en bout et retourne la décision et sa trace.

    Ne lève jamais : chaque échec possible a une réponse dégradée (ORCH-3).
    Tous les appels LLM partagent la même échéance : leur timeout, leur
    lissage de débit et leurs reprises ne peuvent plus prolonger la requête
    au-delà du budget global (ORCH-5).
    """
    depart = time.monotonic()
    with limiter_temps_llm(config.orchestrateur_budget_s, depart=depart):
        return _traiter_demande_dans_budget(entree, depart)


def _traiter_demande_dans_budget(
    entree: OrientationInput, depart: float
) -> OrientationReponse:
    """Implémentation du pipeline sous l'échéance installée par l'entrée publique."""
    trace_id = str(uuid.uuid4())
    degradations: list[str] = []

    # 1+2. Garde-fous d'entrée et recherche documentaire, en parallèle.
    #
    # Les deux étapes sont indépendantes (voir `_executer_en_parallele`) :
    # seul l'**usage** du résultat de la recherche documentaire reste
    # conditionné au verdict des garde-fous, exactement comme avant. Sur un
    # message jugé dangereux, `contexte` est calculé (par un thread du pool,
    # en parallèle) mais **jamais utilisé** ci-dessous : ni transmis à
    # l'agent (celui-ci n'est même pas appelé), ni journalisé, ni renvoyé —
    # `_finaliser` reçoit `None`, inchangé. Le principe « aucun outil ni
    # recommandation sur une tentative de manipulation » (voir l'en-tête du
    # module) reste donc respecté dans son effet observable ; ce qui change
    # est qu'un calcul purement local (embeddings ONNX + Chroma, aucun appel
    # réseau ni écriture) est parfois mené puis jeté plutôt que jamais lancé.
    (risque, echec_injection), (contexte, echec_rag) = _executer_en_parallele(
        lambda: _verifier_injection_entree(entree.message, trace_id),
        lambda: _etape_optionnelle(
            "recherche documentaire", lambda: retrieve_context(entree.message), [], depart
        ),
    )
    if echec_injection:
        degradations.append(echec_injection)

    if risque["danger"]:
        decision = _escalade_injection(entree.message, risque)
        return _finaliser(trace_id, entree, None, decision, depart)

    historique_sain, note_historique = _historique_sain(entree)
    if note_historique:
        degradations.append(note_historique)

    if risque.get("verification_llm") == "indisponible":
        degradations.append("vérification anti-injection LLM indisponible")

    if echec_rag:
        degradations.append(echec_rag)

    # 3. Agent avec outils.
    if _budget_epuise(depart):
        degradations.append(
            f"agent sauté (budget de {config.orchestrateur_budget_s:.0f} s dépassé)"
        )
        decision = _decision_repli("budget de temps dépassé")
    else:
        try:
            decision = run_agent(
                entree.message, entree.profil, contexte, trace_id, historique_sain
            )
        except LLMError as e:
            logger.warning("Agent indisponible (trace_id=%s) : %s", trace_id, e)
            degradations.append(f"agent indisponible ({type(e).__name__})")
            decision = _decision_repli(str(e))
        except Exception as e:  # noqa: BLE001 — voir ci-dessous : la promesse est absolue
            # **Ne rattraper que `LLMError` ne suffisait pas.** Une réponse
            # finale au JSON valide mais non conforme au schéma fait lever une
            # `ValidationError` par `agent._valider_reponse_finale()`, qui
            # traversait tout l'orchestrateur — donc un HTTP 500 nu, alors que
            # ce module, `api.traiter()` et le ticket ORCH-3 promettent tous
            # trois l'inverse. Reproduit en simulant une sortie tronquée.
            #
            # La promesse « ne lève jamais » ne tolère pas une liste
            # d'exceptions connues : c'est la sortie d'un modèle de langage, on
            # ne peut pas énumérer ses façons d'échouer. Le type réel est
            # conservé dans la dégradation et la trace pour rester diagnosticable.
            logger.exception("Agent en échec inattendu (trace_id=%s)", trace_id)
            degradations.append(f"agent en échec ({type(e).__name__})")
            decision = _decision_repli(f"{type(e).__name__}: {e}")

    # 4. Contrôles déterministes finaux.
    decision = _appliquer_plafond_de_confiance(decision, degradations)
    decision = _appliquer_controle_de_sortie(decision)
    return _finaliser(trace_id, entree, contexte, decision, depart)
