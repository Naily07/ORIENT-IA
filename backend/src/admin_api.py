"""Endpoints admin en lecture seule — traduction HTTP de ce que `back_office.py`
(Streamlit) calcule aujourd'hui par import direct des modules backend.

Existent pour que le nouveau frontend Next.js (`frontend-next/`) puisse afficher
les mêmes vues d'administration sans jamais importer de code Python — un
Node/TypeScript ne peut pas importer `src.graphe`/`src.sources`/`src.models`.
Aucune logique métier n'est dupliquée ici : chaque endpoint appelle les
fonctions déjà existantes et les sérialise, dans le même esprit que `api.py`
(« module volontairement mince »).

Pas de nouvelle authentification : ces endpoints restent ouverts, comme le
reste de l'API aujourd'hui — le contrôle d'accès vit dans le frontend
(cohérent avec `noyau.exiger_acces_admin`, qui n'est de toute façon pas un
contrôle de sécurité robuste — le prototype ne manipule aucune donnée
personnelle). Si l'API est un jour exposée au-delà de `localhost`, ce sera le
point d'ancrage naturel pour une vérification de code admin (voir BACKLOG.md).
"""

import json
from collections import Counter
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.config import RACINE, config
from src.graphe import construire_graphe, detecter_incoherences, type_et_id
from src.models import CorpusFormations, charger_corpus_formations
from src.observability import lire_dernieres_traces
from src.sources import EntreeRegistreSource, charger_registre_sources, verifier_provenance

router = APIRouter(prefix="/admin", tags=["admin"])


# --- Tableau de bord ---------------------------------------------------------


class ConfigurationCalibree(BaseModel):
    rag_seuil_pertinence: float
    rag_k: int
    agent_max_iterations: int
    orchestrateur_seuil_confiance: float


class EtatAvancementDonnees(BaseModel):
    matieres: int
    competences: int
    metiers: int
    prerequis: int


class TableauDeBordReponse(BaseModel):
    configuration: ConfigurationCalibree
    etat_avancement_donnees: EtatAvancementDonnees


@router.get("/tableau-de-bord")
def tableau_de_bord() -> TableauDeBordReponse:
    """Seuils calibrés + comptages du corpus structuré (miroir de
    `page_tableau_de_bord`). Les comptages mentions/parcours et l'état de la
    clé LLM restent sur `GET /health`, déjà exposés — pas dupliqués ici."""
    corpus = charger_corpus_formations()
    return TableauDeBordReponse(
        configuration=ConfigurationCalibree(
            rag_seuil_pertinence=config.rag_seuil_pertinence,
            rag_k=config.rag_k,
            agent_max_iterations=config.agent_max_iterations,
            orchestrateur_seuil_confiance=config.orchestrateur_seuil_confiance,
        ),
        etat_avancement_donnees=EtatAvancementDonnees(
            matieres=len(corpus.matieres),
            competences=len(corpus.competences),
            metiers=len(corpus.metiers),
            prerequis=len(corpus.prerequis),
        ),
    )


# --- Observabilité : tendances -----------------------------------------------
# Seule véritable logique nouvelle de ce module : aucune fonction existante
# n'agrège déjà les traces par période. Le reste de ce fichier est un
# passe-plat ; ceci mérite donc d'être testé isolément (voir test_admin_api.py).

Intervalle = Literal["heure", "jour"]


class SeauTendance(BaseModel):
    periode: str
    volume: int
    latence_moyenne_ms: float
    confiance_moyenne: float | None
    repartition_actions: dict[str, int]


class TendancesReponse(BaseModel):
    intervalle: Intervalle
    seaux: list[SeauTendance]


def _cle_periode(horodatage: str, intervalle: Intervalle) -> str | None:
    try:
        dt = datetime.fromisoformat(horodatage)
    except (TypeError, ValueError):
        return None
    return dt.strftime("%Y-%m-%d") if intervalle == "jour" else dt.strftime("%Y-%m-%dT%H:00")


def _agreger_par_periode(traces: list[dict], intervalle: Intervalle) -> list[SeauTendance]:
    seaux: dict[str, list[dict]] = {}
    for trace in traces:
        cle = _cle_periode(trace.get("horodatage", ""), intervalle)
        if cle is None:
            continue
        seaux.setdefault(cle, []).append(trace)

    resultat = []
    for periode in sorted(seaux):  # ordre chronologique croissant, pensé pour un graphique
        groupe = seaux[periode]
        latences = [t.get("latence_ms", 0) or 0 for t in groupe]
        confiances = [
            (t.get("decision") or {}).get("confiance")
            for t in groupe
            if (t.get("decision") or {}).get("confiance") is not None
        ]
        actions = Counter((t.get("decision") or {}).get("action", "?") for t in groupe)
        resultat.append(
            SeauTendance(
                periode=periode,
                volume=len(groupe),
                latence_moyenne_ms=round(sum(latences) / len(latences), 1) if latences else 0.0,
                confiance_moyenne=(
                    round(sum(confiances) / len(confiances), 3) if confiances else None
                ),
                repartition_actions=dict(actions),
            )
        )
    return resultat


@router.get("/observabilite/tendances")
def observabilite_tendances(
    intervalle: Intervalle = "jour",
    limite: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> TendancesReponse:
    traces = lire_dernieres_traces(limite)
    return TendancesReponse(intervalle=intervalle, seaux=_agreger_par_periode(traces, intervalle))


# --- Qualité des données ------------------------------------------------------


class QualiteDonneesReponse(BaseModel):
    incoherences: list[dict]
    donnees_manquantes: list[dict]
    contradictions: list[dict]
    registre_sources: list[EntreeRegistreSource]
    references_orphelines: list[str]


@router.get("/qualite-donnees")
def qualite_donnees() -> QualiteDonneesReponse:
    corpus = charger_corpus_formations()
    graphe_connaissances = construire_graphe(corpus)
    incoherences = detecter_incoherences(corpus, graphe_connaissances)
    manquantes = [i for i in incoherences if i.get("donnee_manquante")]
    contradictions = [i for i in incoherences if not i.get("donnee_manquante")]
    registre = charger_registre_sources()
    orphelines = verifier_provenance(
        [m.source_id for m in corpus.mentions] + [p.source_id for p in corpus.parcours],
        registre,
    )
    return QualiteDonneesReponse(
        incoherences=incoherences,
        donnees_manquantes=manquantes,
        contradictions=contradictions,
        registre_sources=registre,
        references_orphelines=orphelines,
    )


# --- Corpus structuré ----------------------------------------------------------


@router.get("/corpus")
def corpus() -> CorpusFormations:
    return charger_corpus_formations()


# --- Graphe de connaissances -----------------------------------------------------
# JSON structuré plutôt que DOT (`back_office.py::_graphe_dot`) : le frontend
# Next.js rend le graphe avec une librairie de diagramme nœuds-arêtes, pas
# graphviz — le format DOT n'aurait aucun consommateur côté Next.js.

_COULEURS_NOEUDS = {
    "Parcours": "#1f6f5c",
    "Mention": "#5b4b8a",
    "Prerequis": "#a85a25",
    "Competence": "#2c6fa8",
    "Metier": "#7a2c5c",
    "Matiere": "#3d6b1f",
}


class NoeudGraphe(BaseModel):
    id: str
    type: str
    nom: str
    couleur: str


class RelationGraphe(BaseModel):
    source: str
    cible: str
    relation: str


class GrapheReponse(BaseModel):
    noeuds: list[NoeudGraphe]
    relations: list[RelationGraphe]


@router.get("/graphe")
def graphe(types: Annotated[list[str] | None, Query()] = None) -> GrapheReponse:
    types_retenus = set(types) if types else set(_COULEURS_NOEUDS)
    g = construire_graphe(charger_corpus_formations())

    noeuds = []
    gardes = set()
    for noeud in g.nodes:
        type_entite, _ = type_et_id(noeud)
        if type_entite not in types_retenus:
            continue
        gardes.add(noeud)
        noeuds.append(
            NoeudGraphe(
                id=noeud,
                type=type_entite,
                nom=g.nodes[noeud].get("nom", noeud),
                couleur=_COULEURS_NOEUDS.get(type_entite, "#555555"),
            )
        )
    relations = [
        RelationGraphe(source=source, cible=cible, relation=str(donnees.get("relation", "")))
        for source, cible, donnees in g.edges(data=True)
        if source in gardes and cible in gardes
    ]
    return GrapheReponse(noeuds=noeuds, relations=relations)


# --- Mesures --------------------------------------------------------------------


class ArtefactMesure(BaseModel):
    """Équivalent HTTP de `charger_json_local`/`artefact_absent` : toujours 200,
    `disponible=False` + `commande` plutôt qu'un 404 — « pas encore généré »
    est un état normal et documenté du projet (BACKLOG.md, DATA-1), pas une
    erreur. `/admin/mesures` répond en une fois pour 3 artefacts indépendants ;
    un 404 global ne pourrait pas exprimer « ML absent mais RAG présent »."""

    disponible: bool
    donnees: dict | list | None = None
    commande: str | None = None


class MesuresReponse(BaseModel):
    ml: ArtefactMesure
    rag: ArtefactMesure
    systeme: ArtefactMesure


def _charger_artefact(nom_fichier: str, commande: str) -> ArtefactMesure:
    chemin = RACINE / "tests" / nom_fichier
    if not chemin.exists():
        return ArtefactMesure(disponible=False, commande=commande)
    try:
        with open(chemin, encoding="utf-8") as f:
            return ArtefactMesure(disponible=True, donnees=json.load(f))
    except (OSError, ValueError):
        return ArtefactMesure(disponible=False, commande=commande)


@router.get("/mesures")
def mesures() -> MesuresReponse:
    return MesuresReponse(
        ml=_charger_artefact("eval_results_ml.json", "cd backend && python -m tests.eval_ml"),
        rag=_charger_artefact(
            "eval_results_rag_calibration.json",
            "cd backend && python -m tests.calibrer_seuil_rag",
        ),
        systeme=_charger_artefact(
            "eval_results.json", "cd backend && python -m tests.eval_system"
        ),
    )
