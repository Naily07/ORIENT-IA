"""Tests des endpoints admin (`src.admin_api`).

Même style que `test_api.py` : le contrat HTTP est vérifié en monkeypatchant
les fonctions que `admin_api.py` réutilise, pas en rechargeant de vraies
données sur disque — la logique métier elle-même (construction du graphe,
détection d'incohérences, provenance) est déjà couverte sans réseau par
`test_graphe.py`/`test_sources.py`.
"""

import pytest
from fastapi.testclient import TestClient

from src.sources import EntreeRegistreSource
from tests.corpus_jouet import corpus_avec_incoherences, corpus_coherent


@pytest.fixture
def client(monkeypatch):
    # Évite tout accès à Chroma pendant le démarrage de l'app en tests :
    # l'ingestion automatique du corpus RAG ne se déclenche que si l'index
    # est absent ou périmé (voir `src.api.lifespan`, constat d'audit C1).
    monkeypatch.setattr("src.api.index_a_jour", lambda documents: True)
    # Évite d'entraîner le modèle ML et d'interroger Chroma au démarrage
    # (préchauffage, constat d'audit P3, voir `src.api.lifespan`).
    monkeypatch.setattr("src.api.precharger_modeles_ml", lambda: None)
    monkeypatch.setattr("src.api.retrieve_context", lambda *a, **k: [])

    from src.api import app

    with TestClient(app) as c:
        yield c


# --- Tableau de bord ----------------------------------------------------------


def test_tableau_de_bord_expose_la_configuration_calibree(client, monkeypatch):
    monkeypatch.setattr("src.admin_api.charger_corpus_formations", corpus_coherent)

    reponse = client.get("/admin/tableau-de-bord")

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["configuration"]["rag_k"] > 0
    assert corps["etat_avancement_donnees"]["matieres"] == 1
    assert corps["etat_avancement_donnees"]["prerequis"] == 2


# --- Observabilité : tendances -------------------------------------------------


def _trace(horodatage: str, action: str, latence_ms: float, confiance: float | None):
    decision = {"action": action}
    if confiance is not None:
        decision["confiance"] = confiance
    return {"horodatage": horodatage, "latence_ms": latence_ms, "decision": decision}


def test_tendances_agrege_par_jour(client, monkeypatch):
    traces = [
        _trace("2026-08-26T10:00:00+00:00", "recommandation", 100.0, 0.8),
        _trace("2026-08-26T14:00:00+00:00", "information", 200.0, 0.6),
        _trace("2026-08-27T09:00:00+00:00", "recommandation", 300.0, None),
    ]
    monkeypatch.setattr("src.admin_api.lire_dernieres_traces", lambda limite: traces)

    reponse = client.get("/admin/observabilite/tendances", params={"intervalle": "jour"})

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["intervalle"] == "jour"
    seaux = corps["seaux"]
    assert [s["periode"] for s in seaux] == ["2026-08-26", "2026-08-27"]  # ordre croissant

    premier = seaux[0]
    assert premier["volume"] == 2
    assert premier["latence_moyenne_ms"] == 150.0
    assert premier["confiance_moyenne"] == 0.7
    assert premier["repartition_actions"] == {"recommandation": 1, "information": 1}

    second = seaux[1]
    assert second["volume"] == 1
    assert second["confiance_moyenne"] is None  # aucune trace de ce seau n'en déclare


def test_tendances_agrege_par_heure(client, monkeypatch):
    traces = [
        _trace("2026-08-26T10:05:00+00:00", "recommandation", 100.0, 0.5),
        _trace("2026-08-26T10:55:00+00:00", "recommandation", 300.0, 0.5),
        _trace("2026-08-26T11:05:00+00:00", "information", 50.0, None),
    ]
    monkeypatch.setattr("src.admin_api.lire_dernieres_traces", lambda limite: traces)

    reponse = client.get("/admin/observabilite/tendances", params={"intervalle": "heure"})

    seaux = reponse.json()["seaux"]
    assert [s["periode"] for s in seaux] == ["2026-08-26T10:00", "2026-08-26T11:00"]
    assert seaux[0]["volume"] == 2
    assert seaux[1]["volume"] == 1


def test_tendances_ignore_les_horodatages_illisibles(client, monkeypatch):
    traces = [
        {"horodatage": "pas une date", "latence_ms": 1.0, "decision": {"action": "x"}},
        _trace("2026-08-26T10:00:00+00:00", "recommandation", 100.0, 0.5),
    ]
    monkeypatch.setattr("src.admin_api.lire_dernieres_traces", lambda limite: traces)

    reponse = client.get("/admin/observabilite/tendances")

    seaux = reponse.json()["seaux"]
    assert len(seaux) == 1
    assert seaux[0]["volume"] == 1


def test_tendances_transmet_la_limite(client, monkeypatch):
    limites_recues = []

    def lire(limite: int):
        limites_recues.append(limite)
        return []

    monkeypatch.setattr("src.admin_api.lire_dernieres_traces", lire)

    reponse = client.get("/admin/observabilite/tendances", params={"limite": 42})

    assert reponse.status_code == 200
    assert limites_recues == [42]


# --- Qualité des données --------------------------------------------------------


def test_qualite_donnees_distingue_manquantes_et_contradictions(client, monkeypatch):
    monkeypatch.setattr("src.admin_api.charger_corpus_formations", corpus_avec_incoherences)
    monkeypatch.setattr("src.admin_api.charger_registre_sources", list)

    reponse = client.get("/admin/qualite-donnees")

    assert reponse.status_code == 200
    corps = reponse.json()
    assert len(corps["incoherences"]) == len(corps["donnees_manquantes"]) + len(
        corps["contradictions"]
    )
    assert corps["registre_sources"] == []
    # `corpus_avec_incoherences` fait référence à des sources absentes du
    # registre vide fourni ci-dessus, via `Parcours.source_id`.
    assert "FORM-IGGLIA-JOUET" in corps["references_orphelines"]


def test_qualite_donnees_registre_present_ferme_les_orphelines(client, monkeypatch):
    monkeypatch.setattr("src.admin_api.charger_corpus_formations", corpus_coherent)
    registre = [
        EntreeRegistreSource(
            id="FORM-IGGLIA-JOUET",
            titre="Fiche IGGLIA",
            url="https://exemple.test",
            date_consultation="2026-01-01",
            statut="officiel",
        )
    ]
    monkeypatch.setattr("src.admin_api.charger_registre_sources", lambda: registre)

    reponse = client.get("/admin/qualite-donnees")

    assert reponse.json()["references_orphelines"] == []


# --- Corpus structuré ------------------------------------------------------------


def test_corpus_retourne_le_corpus_structure(client, monkeypatch):
    monkeypatch.setattr("src.admin_api.charger_corpus_formations", corpus_coherent)

    reponse = client.get("/admin/corpus")

    assert reponse.status_code == 200
    corps = reponse.json()
    assert len(corps["parcours"]) == 2
    assert {p["id"] for p in corps["parcours"]} == {"IGGLIA", "TEH"}


# --- Graphe de connaissances -------------------------------------------------------


def test_graphe_sans_filtre_retourne_tous_les_types(client, monkeypatch):
    monkeypatch.setattr("src.admin_api.charger_corpus_formations", corpus_coherent)

    reponse = client.get("/admin/graphe")

    corps = reponse.json()
    types_presents = {n["type"] for n in corps["noeuds"]}
    assert types_presents == {"Parcours", "Mention", "Matiere", "Competence", "Prerequis", "Metier"}
    assert any(r["relation"] == "enseigne" for r in corps["relations"])


def test_graphe_filtre_par_types(client, monkeypatch):
    monkeypatch.setattr("src.admin_api.charger_corpus_formations", corpus_coherent)

    reponse = client.get("/admin/graphe", params={"types": ["Parcours", "Mention"]})

    corps = reponse.json()
    assert {n["type"] for n in corps["noeuds"]} == {"Parcours", "Mention"}
    # `appartientA` relie Parcours -> Mention, les deux gardés par le filtre.
    assert any(r["relation"] == "appartientA" for r in corps["relations"])
    # `enseigne` relie Parcours -> Matiere : Matiere est filtrée, l'arête doit
    # disparaître plutôt que pointer vers un nœud absent de la réponse.
    assert not any(r["relation"] == "enseigne" for r in corps["relations"])


# --- Mesures ------------------------------------------------------------------


def test_mesures_signale_les_artefacts_absents(client, monkeypatch, tmp_path):
    monkeypatch.setattr("src.admin_api.RACINE", tmp_path)

    reponse = client.get("/admin/mesures")

    corps = reponse.json()
    for cle in ("ml", "rag", "systeme"):
        assert corps[cle]["disponible"] is False
        assert corps[cle]["donnees"] is None
        assert "python -m tests." in corps[cle]["commande"]


def test_mesures_retourne_les_donnees_presentes(client, monkeypatch, tmp_path):
    dossier_tests = tmp_path / "tests"
    dossier_tests.mkdir()
    (dossier_tests / "eval_results_ml.json").write_text('{"exactitude": 0.9}', encoding="utf-8")
    monkeypatch.setattr("src.admin_api.RACINE", tmp_path)

    reponse = client.get("/admin/mesures")

    corps = reponse.json()
    assert corps["ml"]["disponible"] is True
    assert corps["ml"]["donnees"] == {"exactitude": 0.9}
    assert corps["rag"]["disponible"] is False
