"""Tests hors réseau du client LLM : extraction du délai de reprise, garde-fou
sur la clé API absente, et branchement du hook d'observabilité."""

import pytest

from src.llm_client import LLMError, _delai_suggere, _get_client, set_log_llm_call


def test_delai_suggere_extrait_la_valeur_de_l_api():
    assert _delai_suggere("Please retry in 5.79s") == 5.79
    assert _delai_suggere("Please retry in 12s") == 12.0


def test_delai_suggere_absent_retourne_none():
    assert _delai_suggere("erreur quelconque sans délai") is None


def test_get_client_leve_llmerror_sans_cle_api(monkeypatch):
    from src.config import config

    monkeypatch.setattr(config, "gemini_api_key", "")
    _get_client.cache_clear()
    with pytest.raises(LLMError):
        _get_client()
    _get_client.cache_clear()


def test_set_log_llm_call_branche_le_hook():
    appels = []
    set_log_llm_call(lambda **kw: appels.append(kw))

    import src.llm_client as llm_client_module

    assert llm_client_module._log_llm_call is not None
    llm_client_module._log_llm_call(prompt_systeme="x", contenu="y", reponse_texte="z",
                                     modele="m", latence_ms=1.0)
    assert len(appels) == 1

    set_log_llm_call(None)  # ne pas polluer les autres tests
