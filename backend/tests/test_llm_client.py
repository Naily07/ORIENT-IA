"""Tests hors réseau du client LLM : reprise, observabilité et budget cumulé."""

import pytest

from src.llm_client import (
    BudgetTempsDepasse,
    LLMError,
    _appeler_avec_reprise,
    _delai_suggere,
    _get_client,
    limiter_temps_llm,
    set_log_llm_call,
)


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


def test_budget_reduit_le_timeout_http_au_temps_restant(monkeypatch):
    import src.llm_client as llm_client_module

    timeouts_recus = []

    class _Models:
        @staticmethod
        def generate_content(**kwargs):
            return "ok"

    class _Client:
        models = _Models()

    monkeypatch.setattr(llm_client_module, "_attendre_son_tour", lambda: None)
    monkeypatch.setattr(
        llm_client_module,
        "_get_client",
        lambda timeout_ms=None: timeouts_recus.append(timeout_ms) or _Client(),
    )
    instants = iter([100.0, 103.0])
    monkeypatch.setattr(llm_client_module.time, "monotonic", lambda: next(instants))

    with limiter_temps_llm(5.0):
        assert _appeler_avec_reprise("contenu", object()) == "ok"

    assert timeouts_recus == [2000]


def test_reprise_est_refusee_si_son_attente_depasse_le_budget(monkeypatch):
    import src.llm_client as llm_client_module

    sommeils = []

    class _Models:
        @staticmethod
        def generate_content(**kwargs):
            raise RuntimeError("504 DEADLINE_EXCEEDED")

    class _Client:
        models = _Models()

    monkeypatch.setattr(llm_client_module, "_attendre_son_tour", lambda: None)
    monkeypatch.setattr(llm_client_module, "_get_client", lambda timeout_ms=None: _Client())
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda duree: sommeils.append(duree))
    monkeypatch.setattr(llm_client_module.time, "monotonic", lambda: 100.0)

    with limiter_temps_llm(1.0):
        with pytest.raises(BudgetTempsDepasse):
            _appeler_avec_reprise("contenu", object())

    assert sommeils == []
