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


def test_health_expose_la_mention_obligatoire(client):
    """SEC-5 : le frontend (FE-1, pas encore construit) doit pouvoir
    récupérer le texte exact plutôt que le retaper à la main."""
    from src.config import MENTION_OBLIGATOIRE

    reponse = client.get("/health")
    assert reponse.json()["mention_obligatoire"] == MENTION_OBLIGATOIRE


def test_health_expose_les_marqueurs_ml(client):
    """Les mêmes constantes que `front_office._marqueurs()` lit aujourd'hui
    par import direct, exposées ici pour un frontend qui ne peut pas
    importer de code Python (voir `src.admin_api`)."""
    from src.ml.hybride import MARQUEUR_REGLE_ADMISSION
    from src.ml.outils import AVERTISSEMENT_NON_EXPLOITABLE

    corps = client.get("/health").json()
    assert corps["marqueur_regle_admission"] == MARQUEUR_REGLE_ADMISSION
    assert corps["avertissement_non_exploitable"] == AVERTISSEMENT_NON_EXPLOITABLE


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


def test_observabilite_transmet_la_limite(client, monkeypatch):
    limites_recues = []

    def lire(limite: int):
        limites_recues.append(limite)
        return []

    monkeypatch.setattr("src.api.lire_dernieres_traces", lire)

    reponse = client.get("/observabilite/traces", params={"limite": 12})

    assert reponse.status_code == 200
    assert limites_recues == [12]


def test_observabilite_refuse_une_limite_invalide(client):
    assert client.get("/observabilite/traces", params={"limite": 0}).status_code == 422
    assert client.get("/observabilite/traces", params={"limite": 501}).status_code == 422
