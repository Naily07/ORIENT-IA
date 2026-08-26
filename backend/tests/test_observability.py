"""Tests de l'écriture et de la lecture des traces JSONL, sur des fichiers
temporaires pour ne jamais toucher aux logs réels."""

from src.observability import (
    ChronoLatence,
    estimer_cout,
    lire_dernieres_traces,
    lire_derniers_appels_llm,
    lire_derniers_appels_outils,
    log_llm_call,
    log_tool_call,
    log_trace,
)


def test_log_trace_puis_lecture(tmp_path, monkeypatch):
    from src.config import config

    monkeypatch.setattr(config, "dossier_logs", tmp_path)
    log_trace("trace-1", "un profil très intéressé par les maths", [{"source_id": "X"}],
              {"parcours": "informatique"}, 42.0, profil={"matiere_preferee": "maths"})

    traces = lire_dernieres_traces()
    assert len(traces) == 1
    assert traces[0]["trace_id"] == "trace-1"
    assert traces[0]["nb_documents_contexte"] == 1
    assert traces[0]["decision"] == {"parcours": "informatique"}
    assert traces[0]["profil"] == {"matiere_preferee": "maths"}


def test_les_donnees_sensibles_sont_masquees_avant_ecriture(tmp_path, monkeypatch):
    from src.config import config

    monkeypatch.setattr(config, "dossier_logs", tmp_path)
    log_trace("trace-2", "mon mot de passe est Ete2024!", [], {}, 10.0)

    traces = lire_dernieres_traces()
    assert "Ete2024!" not in str(traces[0])


def test_log_tool_call_puis_lecture(tmp_path, monkeypatch):
    from src.config import config

    monkeypatch.setattr(config, "dossier_logs", tmp_path)
    log_tool_call("trace-3", "rechercher_formation", {"mot_cle": "info"}, {"resultats": []},
                  "succes", 5.0)

    appels = lire_derniers_appels_outils()
    assert len(appels) == 1
    assert appels[0]["outil"] == "rechercher_formation"


def test_log_llm_call_puis_lecture(tmp_path, monkeypatch):
    from src.config import config

    monkeypatch.setattr(config, "dossier_logs", tmp_path)
    log_llm_call("prompt système", "contenu", "réponse", "modele-test", 100.0,
                 tokens_entree=1000, tokens_sortie=200, etape="rag")

    appels = lire_derniers_appels_llm()
    assert len(appels) == 1
    assert appels[0]["etape"] == "rag"
    assert appels[0]["cout_estime_usd"] is not None


def test_estimer_cout_sans_tokens_retourne_none():
    assert estimer_cout(None, None) is None


def test_estimer_cout_calcule_un_montant_positif():
    assert estimer_cout(1_000_000, 1_000_000) > 0


def test_chrono_latence_mesure_une_duree_positive():
    with ChronoLatence() as chrono:
        pass
    assert chrono.ms >= 0
