"""API HTTP du projet (ORCH-4).

Module volontairement mince : il traduit du HTTP vers le pipeline
(`src.orchestrator`) et retour. Toute la logique de traitement vit dans
l'orchestrateur, ce qui permet de la tester sans passer par le réseau (voir
`backend/tests/test_orchestrator.py`).

Pas d'équivalent à `POST /tickets/valider` d'EXAM-S2 : aucun outil du
périmètre actuel n'est marqué sensible (`tools.OUTILS_SENSIBLES` est vide —
que des consultations), donc aucune action ne reste jamais en attente d'une
validation humaine. À ajouter si un outil sensible entre un jour dans le
périmètre (ex. une action d'inscription réelle).
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Query

from src.admin_api import router as admin_router
from src.config import MENTION_OBLIGATOIRE, config
from src.llm_client import set_log_llm_call
from src.ml.hybride import MARQUEUR_REGLE_ADMISSION
from src.ml.outils import AVERTISSEMENT_NON_EXPLOITABLE
from src.ml.outils import precharger as precharger_modeles_ml
from src.models import charger_corpus_rag
from src.observability import lire_dernieres_traces, log_llm_call, log_tool_call, log_trace
from src.orchestrator import set_log_trace, traiter_demande
from src.rag import index_a_jour, ingerer, retrieve_context
from src.schemas import OrientationInput, OrientationReponse
from src.tools import corpus_charge, initialiser_corpus, set_log_appel


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Les outils lisent le corpus structuré en mémoire (tools.py) : sans ce
    # chargement au démarrage, tout appel d'outil échouerait en
    # « corpus non initialisé ». `charger_corpus_formations()` (appelé par
    # `initialiser_corpus()`) tolère les fichiers absents.
    initialiser_corpus()

    # Hooks d'observabilité, branchés une fois au démarrage — pattern repris
    # d'EXAM-S2 pour éviter un import direct qui créerait un cycle entre
    # `observability.py` (importe `guardrails.py`) et `tools.py`/`llm_client.py`.
    set_log_appel(log_tool_call)
    set_log_llm_call(log_llm_call)
    set_log_trace(log_trace)

    # Ingestion automatique du corpus RAG si l'index est vide OU périmé.
    #
    # Constat d'audit C1 : la condition d'origine (`nombre_de_fragments() ==
    # 0`) ne se déclenchait qu'une base jamais peuplée. Un `chroma_db/` déjà
    # rempli par une démo précédente continuait alors de servir l'ancien
    # corpus, en ignorant silencieusement un `corpus_genere.json` enrichi —
    # rencontré en développement, il avait fallu forcer la ré-ingestion à la
    # main. `index_a_jour()` compare désormais une empreinte du corpus
    # (`rag.empreinte_corpus`) à celle réellement indexée.
    try:
        documents = charger_corpus_rag()
        if documents and not index_a_jour(documents):
            ingerer(documents)
    except Exception:
        import logging

        logging.getLogger("src.api").exception("Échec de l'ingestion automatique du corpus RAG")

    # Préchauffage (constat d'audit P3) : sans lui, `_modele()` (régression
    # logistique calibrée, `cv=5`) s'entraîne et le modèle d'embedding ONNX
    # se charge au tout premier appel — c'est le premier candidat d'une démo
    # qui payait cette latence, pas un choix voulu. Chacun des deux
    # préchauffages est indépendant et ne doit jamais empêcher l'autre ni le
    # démarrage du serveur.
    precharger_modeles_ml()
    try:
        # Interroger l'index force le calcul d'un embedding, ce que le seul
        # `count()` déjà effectué ci-dessus (via `index_a_jour`) ne fait pas
        # — `retrieve_context` sur un index vide renvoie `[]` sans jamais
        # appeler le modèle, donc ce préchauffage n'a d'effet réel qu'après
        # une ingestion réussie ; sans effet sinon, jamais bloquant.
        retrieve_context("préchauffage du modèle d'embedding au démarrage")
    except Exception:
        import logging

        logging.getLogger("src.api").warning(
            "Préchauffage de la recherche documentaire impossible", exc_info=True
        )

    yield


app = FastAPI(title="ORIENT'IA", lifespan=lifespan)
app.include_router(admin_router)


@app.post("/orientation/traiter", response_model=OrientationReponse)
def traiter(entree: OrientationInput) -> OrientationReponse:
    """Traite une demande d'orientation de bout en bout (ORCH-1).

    Retourne toujours 200 avec une décision valide : les échecs internes
    sont dégradés par l'orchestrateur (ORCH-3), jamais renvoyés en 500 nue.
    """
    return traiter_demande(entree)


@app.get("/observabilite/traces", tags=["observabilite"])
def observabilite_traces(
    limite: Annotated[
        int,
        Query(ge=1, le=500, description="Nombre maximal de traces à retourner"),
    ] = 50,
) -> list[dict]:
    """Retourne les dernières traces, les plus récentes en premier.

    Une liste vide est une réponse normale lorsque le pipeline n'a encore
    traité aucune demande.
    """
    return lire_dernieres_traces(limite)


@app.get("/health")
def health():
    """État du service.

    Volontairement sans appel LLM ni ouverture de l'index Chroma : un
    contrôle de santé doit répondre en quelques millisecondes. Expose ce qui
    explique le plus souvent une démo qui échoue — clé absente, corpus vide.
    """
    corpus = corpus_charge()
    return {
        "status": "ok",
        "modele": config.gemini_model,
        "cle_llm_configuree": bool(config.gemini_api_key),
        "corpus": {
            "mentions": len(corpus.mentions) if corpus else 0,
            "parcours": len(corpus.parcours) if corpus else 0,
        },
        # SEC-5 : source unique du texte exact, à afficher par le frontend
        # (FE-1, pas encore construit) plutôt que de le retaper à la main.
        "mention_obligatoire": MENTION_OBLIGATOIRE,
        # Marqueurs internes lus par le frontend (front_office._marqueurs) pour
        # masquer un score creux / signaler une admissibilité à vérifier sur
        # une carte de parcours — exposés ici pour la même raison que la
        # mention obligatoire : source unique plutôt que recopiée en dur côté
        # frontend, où une désynchronisation silencieuse passerait inaperçue.
        "marqueur_regle_admission": MARQUEUR_REGLE_ADMISSION,
        "avertissement_non_exploitable": AVERTISSEMENT_NON_EXPLOITABLE,
    }
