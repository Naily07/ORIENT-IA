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

from fastapi import FastAPI

from src.config import config
from src.llm_client import set_log_llm_call
from src.models import charger_corpus
from src.observability import lire_dernieres_traces, log_llm_call, log_tool_call, log_trace
from src.orchestrator import set_log_trace, traiter_demande
from src.rag import ingerer, nombre_de_fragments
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

    # Ingestion automatique du corpus RAG s'il est vide au démarrage.
    try:
        if nombre_de_fragments() == 0:
            documents = charger_corpus()
            if documents:
                ingerer(documents)
    except Exception:
        import logging

        logging.getLogger("src.api").exception("Échec de l'ingestion automatique du corpus RAG")

    yield


app = FastAPI(title="ORIENT'IA", lifespan=lifespan)


@app.post("/orientation/traiter", response_model=OrientationReponse)
def traiter(entree: OrientationInput) -> OrientationReponse:
    """Traite une demande d'orientation de bout en bout (ORCH-1).

    Retourne toujours 200 avec une décision valide : les échecs internes
    sont dégradés par l'orchestrateur (ORCH-3), jamais renvoyés en 500 nue.
    """
    return traiter_demande(entree)


@app.get("/observabilite/traces")
def observabilite_traces(limite: int = 50) -> list[dict]:
    """Les dernières traces, les plus récentes en premier."""
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
    }
