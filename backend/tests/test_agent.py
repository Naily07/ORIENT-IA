"""Tests de la boucle agent (AGT-1).

La majorité de ces tests simulent `llm_call_with_tools` et `executer_outil`
(aucun appel réseau). Un test marqué `reseau` en fin de fichier vérifie en
plus que la boucle fonctionne réellement contre l'API Gemini — à lancer avec
`pytest -m reseau` (nécessite `GEMINI_API_KEY` dans `.env`)."""

import pytest

from src.agent import run_agent
from src.ml.features import analyser_couverture
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


def test_source_retournee_par_un_outil_structure_est_conservee(monkeypatch, profil):
    """AGT-6 : une source qui provient d'un outil structuré (`Parcours.source_id`,
    remonté par ex. par `verifier_prerequis`) doit rester citable même sans aucun
    passage RAG fourni — correctif d'EVAL-17 (`eval_analyse.md`)."""
    decision = _decision_type(sources=["FORM-STRUCTURE", "FORM-INVENTE"])
    reponses = iter(
        [
            _reponse_appel_outil("verifier_prerequis", {"parcours": "IGGLIA"}),
            _reponse_finale(decision),
        ]
    )
    monkeypatch.setattr("src.agent.llm_call_with_tools", lambda *a, **k: next(reponses))
    monkeypatch.setattr(
        "src.agent.executer_outil",
        lambda nom, params, trace_id: {
            "statut": "succes",
            "resultat": {"parcours": "IGGLIA", "source_id": "FORM-STRUCTURE", "compatible": True},
        },
    )

    resultat = run_agent("Question", profil, None, "trace-4b")

    assert resultat.sources == ["FORM-STRUCTURE"]


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
    # Seuls les parcours significatifs, plus les 16 classes : au-delà du
    # leader les scores tombent sous 2 %, les remonter tous revenait à
    # présenter du bruit comme des recommandations.
    assert 1 <= len(resultat.parcours_recommandes) <= 3


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
    assert 1 <= len(resultat.parcours_recommandes) <= 3


def test_demande_information_sur_profil_trop_mince_ne_consulte_pas_le_ml(
    monkeypatch, profil
):
    """`demande_information` sur un profil que le compte déterministe de traits
    juge inexploitable (ici une seule matière déclarée, seuil à 2) : le
    garde-fou ne se déclenche pas. Classer un profil que le modèle ne peut pas
    exploiter reviendrait à présenter la distribution a priori comme un
    résultat (ML-9)."""
    assert not analyser_couverture(profil).exploitable  # prémisse du test
    decision = _decision_type(action="demande_information", confiance=0.6)
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-9")

    assert "analyser_profil_ml" not in resultat.outils_utilises
    assert resultat.action == "demande_information"


def test_demande_information_sur_profil_suffisant_consulte_le_ml_et_recommande(
    monkeypatch,
):
    """Défaut EVAL-11, reproduit ici sans réseau : sur un profil que le compte
    déterministe juge exploitable, le modèle de langage répondait « dites-m'en
    plus » sans consulter le modèle ML — qui rend pourtant un classement fondé
    sur ce même profil. Le code contresigne : il consulte, et la recommandation
    fondée n'est plus tue au profit d'une question."""
    profil_suffisant = ProfilCandidat(
        matieres_preferees=["biologie", "chimie"],
        centres_interet=["sante", "recherche"],
        serie_bac="D",
    )
    assert analyser_couverture(profil_suffisant).exploitable  # prémisse du test
    decision = _decision_type(
        action="demande_information",
        confiance=0.4,
        parcours_recommandes=[],
        informations_manquantes=["résultats scolaires"],
        reponse="Pouvez-vous me dire vos résultats scolaires ?",
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Quel parcours me conseilles-tu ?", profil_suffisant, None, "trace-11")

    assert "analyser_profil_ml" in resultat.outils_utilises
    assert resultat.action == "recommandation"
    assert resultat.parcours_recommandes
    # La demande de précisions n'est pas effacée : elle accompagne la
    # recommandation au lieu de la remplacer.
    assert resultat.informations_manquantes == ["résultats scolaires"]


def test_information_confiance_faible_ne_bascule_pas_en_escalade(monkeypatch, profil):
    """Observé en démonstration : une question factuelle à laquelle l'assistant
    ne répondait que partiellement (« liste des matières ? ») finissait « voyez
    un conseiller pédagogique ». Une confiance faible ne force l'escalade que
    sur une *recommandation*, pas sur une question factuelle."""
    decision = _decision_type(
        action="information", confiance=0.2, incertitude_declaree=False,
        reponse="Voici ce que je sais des matières de ce parcours.",
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Liste des matières ?", profil, None, "trace-info-faible")

    assert resultat.action == "information"
    assert resultat.incertitude_declaree is True


def test_reponse_conversationnelle_est_composee_si_absente(monkeypatch, profil):
    """`reponse` est le texte montré à l'utilisateur. Si le modèle ne l'a pas
    rempli (sortie tronquée), le code en compose une plutôt que d'afficher une
    bulle vide."""
    decision = _decision_type(
        action="information", confiance=0.8, reponse="",
        resume="Question sur la mention Informatique.",
        explication="La mention regroupe quatre parcours.",
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-reponse-vide")

    assert resultat.reponse.strip()
    assert "[Contrôle automatique]" not in resultat.reponse


def test_escalade_ajoute_une_phrase_conseiller_a_la_reponse(monkeypatch, profil):
    """Quand un contrôle déterministe passe l'action à `escalade_conseiller`,
    la version parlée (`reponse`) ne doit pas rester muette sur ce point."""
    decision = _decision_type(
        action="recommandation", confiance=0.1, incertitude_declaree=False,
        outils_utilises=["analyser_profil_ml"],
        reponse="Le parcours X vous conviendrait.",
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Quel parcours ?", profil, None, "trace-escalade-reponse")

    assert resultat.action == "escalade_conseiller"
    assert "conseiller" in resultat.reponse.lower()


# --- Politique de refus / incertitude / renvoi (AGT-4) -----------------------


@pytest.fixture
def corpus_jouet_pour_admission():
    """Bascule sur le corpus jouet (IGGLIA exige un bac C/D/S) le temps d'un
    test, puis revient au corpus réel — même pattern que la fixture `corpus`
    de `test_tools.py`, nécessaire ici car `_recouper_avec_regles_pedagogiques`
    appelle `tools.verifier_prerequis()` directement, hors function calling."""
    from src import tools
    from tests.corpus_jouet import corpus_coherent

    tools.initialiser_corpus(corpus_coherent())
    yield
    tools.initialiser_corpus()


def test_recommandation_inadmissible_est_escaladee(monkeypatch, corpus_jouet_pour_admission):
    """`ml.hybride.appliquer_regles_admission` rétrograde déjà les parcours
    inadmissibles sous les parcours accessibles à l'intérieur du modèle ML :
    mais si le modèle a lui-même appelé `analyser_profil_ml` plus tôt dans la
    boucle, rien ne garantit qu'il recopie mot pour mot cette annotation dans
    sa réponse finale plutôt que de la paraphraser. Le garde-fou revérifie
    donc indépendamment, via `tools.verifier_prerequis()`, que le parcours mis
    en avant reste compatible avec la série de baccalauréat déclarée."""
    profil_incompatible = ProfilCandidat(matieres_preferees=["informatique"], serie_bac="L")
    decision = _decision_type(
        action="recommandation",
        confiance=0.8,
        incertitude_declaree=False,
        parcours_recommandes=[
            {"parcours": "IGGLIA", "score_adequation": 0.54, "justification": "Score élevé."}
        ],
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil_incompatible, None, "trace-9b")

    assert resultat.action == "escalade_conseiller"
    assert resultat.incertitude_declaree is True
    assert "[Contrôle automatique]" in resultat.explication
    # Les scores restent ceux du modèle : seule l'action a changé.
    assert resultat.parcours_recommandes[0].parcours == "IGGLIA"


def test_recommandation_admissible_n_est_pas_affectee(monkeypatch, corpus_jouet_pour_admission):
    """Non-régression : un parcours de tête compatible avec la série de
    baccalauréat déclarée ne doit déclencher aucune escalade."""
    profil_compatible = ProfilCandidat(matieres_preferees=["informatique"], serie_bac="D")
    decision = _decision_type(
        action="recommandation",
        confiance=0.8,
        incertitude_declaree=False,
        parcours_recommandes=[
            {"parcours": "IGGLIA", "score_adequation": 0.54, "justification": "Score élevé."}
        ],
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil_compatible, None, "trace-9c")

    assert resultat.action == "recommandation"
    assert "[Contrôle automatique]" not in resultat.explication


def test_recommandation_sans_serie_bac_declaree_n_est_pas_affectee(monkeypatch, profil):
    """Non-régression : une série de baccalauréat non déclarée (`compatible`
    indéterminé) ne doit jamais être traitée comme un refus — même principe
    que `hybride.VerdictAdmission.inadmissible`. Ce test tourne volontairement
    sans le corpus jouet (`profil` n'a pas de `serie_bac`), pour couvrir aussi
    le cas où le corpus/graphe ne sont pas initialisés."""
    decision = _decision_type(
        action="recommandation",
        confiance=0.8,
        incertitude_declaree=False,
        parcours_recommandes=[
            {"parcours": "IGGLIA", "score_adequation": 0.54, "justification": "Score élevé."}
        ],
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil, None, "trace-9f")

    assert resultat.action == "recommandation"
    assert "[Contrôle automatique]" not in resultat.explication


def test_renvoi_administration_efface_les_parcours_recommandes(monkeypatch, profil):
    """§16 du sujet : une décision `renvoi_administration` n'est par définition
    pas un conseil pédagogique — elle ne doit jamais s'accompagner d'une
    recommandation de parcours, même si un appel ML antérieur en a produit une."""
    decision = _decision_type(
        action="renvoi_administration",
        confiance=0.9,
        incertitude_declaree=False,
        outils_utilises=["analyser_profil_ml"],
        parcours_recommandes=[
            {"parcours": "IGGLIA", "score_adequation": 0.54, "justification": "Score élevé."}
        ],
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-9d")

    assert resultat.action == "renvoi_administration"
    assert resultat.parcours_recommandes == []
    assert resultat.incertitude_declaree is True


def test_renvoi_administration_sans_parcours_n_est_pas_modifie(monkeypatch, profil):
    """Non-régression : une décision `renvoi_administration` déjà propre ne
    doit pas être altérée par ce garde-fou."""
    decision = _decision_type(
        action="renvoi_administration",
        confiance=0.9,
        incertitude_declaree=False,
        parcours_recommandes=[],
    )
    monkeypatch.setattr(
        "src.agent.llm_call_with_tools", lambda *a, **k: _reponse_finale(decision)
    )

    resultat = run_agent("Question", profil, None, "trace-9e")

    assert resultat.action == "renvoi_administration"
    assert resultat.incertitude_declaree is False


# --- Cohérence entre la prose et le classement du modèle (AGT-7) -------------


def _decision_avec_classement(explication: str, **overrides) -> RecommandationDecision:
    """Décision où le modèle place IGGLIA en tête, ESIIA loin derrière."""
    return _decision_type(
        parcours_recommandes=[
            {"parcours": "IGGLIA", "score_adequation": 0.54, "justification": "x"},
            {"parcours": "ESIIA", "score_adequation": 0.11, "justification": "x"},
        ],
        confiance=0.54,
        explication=explication,
        **overrides,
    )


def _simuler_agent_ayant_appele_le_modele(monkeypatch, decision: RecommandationDecision):
    """Fait réellement passer l'agent par `analyser_profil_ml` dans sa boucle.

    Indispensable ici : sans cet appel effectif, la consultation forcée
    (`_forcer_consultation_du_modele_ml`) remplacerait `parcours_recommandes`
    par le classement réel du modèle sur le profil de test, et le contrôle de
    cohérence porterait alors sur un classement que le test ne maîtrise pas.
    """
    reponses = iter(
        [_reponse_appel_outil("analyser_profil_ml"), _reponse_finale(decision)]
    )
    monkeypatch.setattr("src.agent.llm_call_with_tools", lambda *a, **k: next(reponses))
    monkeypatch.setattr("src.agent.executer_outil", _executer_outil_factice)


def test_prose_qui_omet_le_parcours_le_mieux_classe_est_signalee(monkeypatch, profil):
    """Le cas réel, reproduit 2 fois sur 2 : le classement place IGGLIA en
    tête mais la prose annonce ESIIA."""
    decision = _decision_avec_classement(
        "Le modèle recommande en priorité le parcours ESIIA, qui combine "
        "électronique et intelligence artificielle."
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil, None, "trace-10")

    assert "[Contrôle automatique]" in resultat.explication
    assert "IGGLIA en tête" in resultat.explication
    # Les scores eux-mêmes restent intacts : seule la narration avait dérivé.
    assert resultat.parcours_recommandes[0].parcours == "IGGLIA"
    assert resultat.action == "recommandation"


def test_prose_coherente_n_est_pas_annotee(monkeypatch, profil):
    decision = _decision_avec_classement(
        "Le modèle place IGGLIA en tête, devant ESIIA."
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil, None, "trace-11")

    assert "[Contrôle automatique]" not in resultat.explication


def test_comparaison_citant_un_autre_parcours_d_abord_n_est_pas_un_faux_positif(
    monkeypatch, profil
):
    """« contrairement à ESIIA, IGGLIA convient mieux » cite ESIIA en premier
    sans rien contredire : c'est l'omission du parcours recommandé qui trahit
    une dérive, pas son rang de citation."""
    decision = _decision_avec_classement(
        "Contrairement à ESIIA, plus orienté électronique, IGGLIA correspond "
        "mieux à ce profil."
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil, None, "trace-12")

    assert "[Contrôle automatique]" not in resultat.explication


def test_prose_sans_aucun_parcours_nomme_n_est_pas_annotee(monkeypatch, profil):
    """Une explication qui ne nomme aucun parcours ne contredit rien."""
    decision = _decision_avec_classement(
        "Ce profil scientifique ouvre plusieurs possibilités en informatique."
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil, None, "trace-13")

    assert "[Contrôle automatique]" not in resultat.explication


def test_mot_francais_contenant_un_sigle_ne_declenche_pas_le_controle(monkeypatch, profil):
    """« emploi » contient « EMP » : sans frontière de mot, toute prose
    parlant d'emploi serait signalée à tort."""
    decision = _decision_avec_classement(
        "IGGLIA offre de bonnes perspectives d'emploi pour un employeur du secteur."
    )
    _simuler_agent_ayant_appele_le_modele(monkeypatch, decision)

    resultat = run_agent("Question", profil, None, "trace-14")

    assert "[Contrôle automatique]" not in resultat.explication


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
