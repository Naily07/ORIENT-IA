"""Vérifie que la configuration se charge avec des valeurs par défaut sûres,
sans dépendre d'un `.env` présent (le cas en CI ou sur une machine fraîche)."""

from src.config import Config, config


def test_config_expose_bien_les_valeurs_par_defaut():
    assert config.gemini_model
    assert config.rag_k > 0
    assert 0 <= config.rag_seuil_pertinence <= 2  # distance cosinus, borne large
    assert config.agent_max_iterations > 0


def test_chemins_derivent_du_dossier_racine():
    fresh = Config()
    assert fresh.fichier_traces.name == "traces.jsonl"
    assert fresh.fichier_tool_calls.parent == fresh.dossier_logs
    assert fresh.fichier_llm_calls.parent == fresh.dossier_logs


def test_seuil_confiance_est_une_probabilite():
    assert 0.0 <= config.orchestrateur_seuil_confiance <= 1.0
