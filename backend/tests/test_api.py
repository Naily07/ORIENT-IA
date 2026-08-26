"""Tests de l'API FastAPI (ORCH-4).

Le pipeline (`traiter_demande`) est simulé : ces tests vérifient le contrat
HTTP (schémas, codes de statut, branchement des hooks), pas la logique
métier — déjà couverte sans réseau par `test_orchestrator.py`.
"""

import pytest
from fastapi.testclient import TestClient

from src.schemas import OrientationReponse, RecommandationDecision


@pytest.fixture
def client(monkeypatch):
    # Évite tout accès à Chroma pendant le démarrage de l'app en tests :
    # l'ingestion automatique du corpus RAG ne se déclenche que si l'index
    # est vide (voir `src.api.lifespan`).
    monkeypatch.setattr("src.api.nombre_de_fragments", lambda: 1)

    from src.api import app

    with TestClient(app) as c:
        yield c


def _decision(**overrides) -> RecommandationDecision:
    valeurs = {
        "resume": "resume",
        "parcours_recommandes": [],
        "confiance": 0.8,
        "informations_manquantes": [],
        "explication": "explication",
        "sources": [],
        "outils_utilises": [],
        "action": "recommandation",
        "incertitude_declaree": False,
    }
    valeurs.update(overrides)
    return RecommandationDecision(**valeurs)


def test_health_repond_ok_avec_le_corpus_charge(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["status"] == "ok"
    assert corps["corpus"]["parcours"] == 16  # les 16 parcours réels de l'ISPM (DATA-1)


def test_traiter_appelle_l_orchestrateur_et_retourne_sa_reponse(client, monkeypatch):
    decision = _decision()
    monkeypatch.setattr(
        "src.api.traiter_demande",
        lambda entree: OrientationReponse(trace_id="trace-test", decision=decision),
    )

    reponse = client.post("/orientation/traiter", json={"message": "Quel parcours ?"})

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["trace_id"] == "trace-test"
    assert corps["decision"]["action"] == "recommandation"


def test_traiter_accepte_un_profil_optionnel(client, monkeypatch):
    entrees_recues = []
    monkeypatch.setattr(
        "src.api.traiter_demande",
        lambda entree: entrees_recues.append(entree)
        or OrientationReponse(trace_id="t", decision=_decision()),
    )

    client.post(
        "/orientation/traiter",
        json={"message": "Question", "profil": {"matieres_preferees": ["informatique"]}},
    )

    assert entrees_recues[0].profil.matieres_preferees == ["informatique"]


def test_traiter_rejette_un_corps_sans_message(client):
    reponse = client.post("/orientation/traiter", json={})
    assert reponse.status_code == 422


def test_observabilite_traces_retourne_une_liste(client, monkeypatch):
    monkeypatch.setattr(
        "src.api.lire_dernieres_traces", lambda limite=50: [{"trace_id": "x"}]
    )
    reponse = client.get("/observabilite/traces")
    assert reponse.status_code == 200
    assert reponse.json() == [{"trace_id": "x"}]
