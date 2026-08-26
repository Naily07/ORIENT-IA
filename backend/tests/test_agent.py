"""Tests de la boucle agent (AGT-1).

La majorité de ces tests simulent `llm_call_with_tools` et `executer_outil`
(aucun appel réseau). Un test marqué `reseau` en fin de fichier vérifie en
plus que la boucle fonctionne réellement contre l'API Gemini — à lancer avec
`pytest -m reseau` (nécessite `GEMINI_API_KEY` dans `.env`)."""

import pytest

from src.agent import run_agent
from src.schemas import ProfilCandidat, RecommandationDecision


class _FausseFunctionCall:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


class _FaussePart:
    def __init__(self, function_call=None):
        self.function_call = function_call


class _FauxContent:
    def __init__(self, parts):
        self.parts = parts


class _FauxCandidat:
    def __init__(self, content):
        self.content = content


class _FausseReponse:
    def __init__(self, parts, parsed=None, text=None):
        self.candidates = [_FauxCandidat(_FauxContent(parts))]
        self.parsed = parsed
        self.text = text


def _decision_type(**overrides) -> RecommandationDecision:
    valeurs = {
        "resume": "Profil orienté informatique",
        "parcours_recommandes": [],
        "confiance": 0.9,
        "informations_manquantes": [],
        "explication": "Le profil déclare un intérêt pour l'informatique.",
        "sources": [],
        "outils_utilises": [],
        "action": "recommandation",
        "incertitude_declaree": False,
    }
    valeurs.update(overrides)
    return RecommandationDecision(**valeurs)


def _reponse_finale(decision: RecommandationDecision) -> _FausseReponse:
    return _FausseReponse(parts=[_FaussePart(function_call=None)], parsed=decision)


def _reponse_appel_outil(nom: str, args: dict | None = None) -> _FausseReponse:
    return _FausseReponse(parts=[_FaussePart(function_call=_FausseFunctionCall(nom, args))])


def _executer_outil_factice(nom, params, trace_id):
    return {"statut": "succes", "resultat": {}}


@pytest.fixture
def profil():
    return ProfilCandidat(matieres_preferees=["informatique"])


def test_reponse_finale_immediate_sans_appel_d_outil(monkeypatch, profil):
    """Le modèle recommande directement, sans le moindre appel d'outil : le
    code force malgré tout la consultation du modèle ML avant de conclure
    (voir test_recommandation_sans_outil_ml_est_corrigee ci-dessous pour le
    détail de ce garde-fou). L'`action` finale n'est volontairement pas
    testée ici : elle dépend de la confiance du vrai modèle sur ce profil
    minimal, ce n'est pas ce que ce test vérifie (pas de boucle infinie, pas
    d'erreur sur une réponse finale immédiate)."""
    decision = _decision_type()
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Quel parcours pour moi ?", profil, None, "trace-1")

    assert resultat.outils_utilises == ["analyser_profil_ml"]


def test_boucle_avec_un_appel_d_outil_puis_reponse_finale(monkeypatch, profil):
    decision = _decision_type()
    reponses = iter(
        [
            _reponse_appel_outil("analyser_profil_ml"),
            _reponse_finale(decision),
        ]
    )
    monkeypatch.setattr("src.agent.llm_call_with_tools", lambda *a, **k: next(reponses))
    monkeypatch.setattr("src.agent.executer_outil", _executer_outil_factice)

    resultat = run_agent("Quel parcours pour moi ?", profil, None, "trace-2")

    assert resultat.action == "recommandation"
    assert resultat.outils_utilises == ["analyser_profil_ml"]


def test_limite_d_iterations_atteinte_escalade_proprement(monkeypatch, profil):
    """Le modèle qui n'appelle que des outils sans jamais conclure ne doit
    jamais bloquer indéfiniment ni renvoyer une erreur nue."""
    reponse_outil = _reponse_appel_outil("rechercher_formation")
    monkeypatch.setattr("src.agent.llm_call_with_tools", lambda *a, **k: reponse_outil)
    monkeypatch.setattr("src.agent.executer_outil", _executer_outil_factice)

    resultat = run_agent("Compare tout", profil, None, "trace-3")

    assert resultat.action == "escalade_conseiller"
    assert resultat.incertitude_declaree is True
    assert resultat.confiance == 0.0


def test_sources_absentes_du_contexte_sont_retirees(monkeypatch, profil):
    decision = _decision_type(sources=["FORM-REEL", "FORM-INVENTE"])
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )
    contexte = [{"source_id": "FORM-REEL", "contenu": "..."}]

    resultat = run_agent("Question", profil, contexte, "trace-4")

    assert resultat.sources == ["FORM-REEL"]


def test_confiance_faible_force_l_escalade(monkeypatch, profil):
    # `analyser_profil_ml` déjà dans outils_utilises : on isole ici le seul
    # comportement testé (seuil de confiance), sans déclencher en plus le
    # garde-fou qui recalculerait la confiance à partir du vrai modèle.
    decision = _decision_type(
        action="recommandation",
        confiance=0.2,
        incertitude_declaree=False,
        outils_utilises=["analyser_profil_ml"],
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-5")

    assert resultat.action == "escalade_conseiller"
    assert resultat.incertitude_declaree is True


def test_outils_utilises_reflete_les_appels_reels(monkeypatch, profil):
    """Le modèle peut prétendre avoir utilisé un outil dans sa sortie JSON :
    seul ce que le code a réellement exécuté doit apparaître. `verifier_prerequis`
    n'est pas `analyser_profil_ml` : le garde-fou de consultation du modèle ML
    (testé isolément plus bas) s'ajoute donc à la liste réelle."""
    decision = _decision_type(outils_utilises=["outil_invente_par_le_modele"])
    reponses = iter(
        [
            _reponse_appel_outil("verifier_prerequis", {"parcours": "IGGLIA"}),
            _reponse_finale(decision),
        ]
    )
    monkeypatch.setattr("src.agent.llm_call_with_tools", lambda *a, **k: next(reponses))
    monkeypatch.setattr("src.agent.executer_outil", _executer_outil_factice)

    resultat = run_agent("Question", profil, None, "trace-6")

    assert resultat.outils_utilises == ["verifier_prerequis", "analyser_profil_ml"]


def test_recommandation_sans_outil_ml_est_corrigee(monkeypatch, profil):
    """Trouvé en usage réel : un contexte RAG suffisamment riche permet
    parfois au modèle de recommander directement, sans jamais appeler
    `analyser_profil_ml`, malgré la consigne du prompt. Le code doit alors
    consulter le modèle lui-même plutôt que d'afficher un score inventé."""
    decision = _decision_type(
        parcours_recommandes=[
            {"parcours": "TEH", "score_adequation": 0.99, "justification": "invente"}
        ],
        confiance=0.99,
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-7")

    assert "analyser_profil_ml" in resultat.outils_utilises
    # Le score inventé par le modèle de langage a été remplacé par celui du
    # vrai modèle ML — jamais garanti d'être identique (et surtout pas de
    # rester à 0.99 sur un profil qui ne pointe pas franchement vers TEH).
    assert resultat.parcours_recommandes != decision.parcours_recommandes
    assert len(resultat.parcours_recommandes) == 16  # les 16 parcours réels, classés


def test_escalade_sans_outil_ml_est_aussi_corrigee(monkeypatch, profil):
    """Trouvé en évaluant le système (EVAL) : sur un profil pourtant
    renseigné, le modèle escaladait parfois directement à confiance nulle
    sans jamais avoir consulté le modèle ML — une escalade tout aussi peu
    fondée qu'une recommandation inventée. Le même garde-fou s'applique."""
    decision = _decision_type(
        action="escalade_conseiller",
        parcours_recommandes=[],
        confiance=0.0,
        incertitude_declaree=True,
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-8")

    assert "analyser_profil_ml" in resultat.outils_utilises
    assert len(resultat.parcours_recommandes) == 16


def test_demande_information_ne_declenche_pas_la_consultation_ml(monkeypatch, profil):
    """À l'inverse, une action `demande_information` (ou `information`,
    `renvoi_administration`) ne doit pas déclencher le garde-fou : consulter
    le modèle ML n'a pas de sens tant que le profil est jugé insuffisant."""
    decision = _decision_type(action="demande_information", confiance=0.6)
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-9")

    assert "analyser_profil_ml" not in resultat.outils_utilises


# --- Test réseau : vérifie la boucle contre l'API Gemini réelle -------------


@pytest.mark.reseau
def test_agent_reel_recommande_un_parcours_coherent():
    """Contrairement aux tests ci-dessus, celui-ci n'attrape aucune régression
    de logique déterministe (déjà couverte sans réseau) : il vérifie que le
    function calling natif de Gemini s'articule réellement avec `tools.py`
    (schémas de paramètres acceptés par l'API, `thought_signature` réattendu
    d'un tour à l'autre, sortie finale conforme à `RecommandationDecision`)."""
    from src.ml.archetypes import ARCHETYPES
    from src.tools import initialiser_corpus

    initialiser_corpus()
    profil = ProfilCandidat(
        matieres_preferees=["informatique", "mathematiques"],
        competences_declarees=["programmation"],
        centres_interet=["technologie"],
        serie_bac="D",
    )

    decision = run_agent(
        "Quel parcours me conseilles-tu ?", profil, None, "test-reseau-agent"
    )

    assert isinstance(decision, RecommandationDecision)
    # Exigence non négociable du prompt système : ne jamais recommander sans
    # être passé par le modèle ML.
    assert "analyser_profil_ml" in decision.outils_utilises
    assert decision.parcours_recommandes
    assert decision.parcours_recommandes[0].parcours in ARCHETYPES
