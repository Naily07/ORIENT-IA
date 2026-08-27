"""Spécification des outils de l'agent (§11 du sujet, AGT-2/AGT-3).

Une source de vérité unique (`OUTILS`), consommée par trois mécanismes —
même principe qu'EXAM-S2 :
- `declarer_outils()` : expose les outils au LLM (function calling natif
  Gemini) ;
- `valider_parametres()` / `executer_outil()` : validation et exécution ;
- chaque outil réalise une opération technique identifiable sur des données
  réelles (corpus structuré, modèle ML) — jamais une simple instruction
  ajoutée au prompt (règle explicite du sujet, §11).

Les 8 outils cités par le sujet sont couverts : `rechercher_formation`,
`comparer_parcours`, `analyser_profil_ml`, `calculer_score_adequation`,
`verifier_prerequis`, `rechercher_competences`, `identifier_debouches`,
`expliquer_recommandation`. Un neuvième, `detecter_incoherences`, vient de
l'extension symbolique (§12 du sujet, ONTO-4).

`verifier_prerequis` et `detecter_incoherences` interrogent le graphe de
connaissances (`src.graphe`, ONTO-2/ONTO-3/ONTO-4) plutôt que de filtrer le
corpus directement : une requête de graphe déterministe, pas un appel LLM.
`expliquer_recommandation` y ajoute un raisonnement multiétape (ONTO-5,
`src.graphe.chemin_competence_parcours_metier`), qui dégrade silencieusement
en liste vide si le graphe n'est pas initialisé — cet enrichissement est un
bonus par rapport au score du modèle ML, jamais une condition pour répondre.

**Traçabilité des réponses issues d'un `Parcours`/`Mention` (AGT-6).** Chaque
outil qui répond à partir d'un `Parcours` ou d'une `Mention` résolus (
`_fiche_parcours` et apparentés : `verifier_prerequis`, `identifier_debouches`,
`expliquer_recommandation`, `rechercher_formation`) remonte le `source_id`
porté par ce modèle (registre DATA-2, `src.sources`). Sans cela, une réponse
fondée sur ces outils structurés, plutôt que sur le RAG, ne pouvait plus citer
sa source alors qu'elle est bien disponible — défaut trouvé à l'évaluation
post-fusion (`backend/tests/eval_analyse.md`, EVAL-17). Voir
`agent._source_ids_des_outils`, qui les collecte pour élargir l'ensemble des
sources que l'agent est autorisé à citer.

**Le profil du candidat n'est jamais un paramètre du function calling.**
Demander au LLM de ressaisir un profil entier en argument JSON l'inviterait à
en inventer ou en oublier des champs — invérifiable. Le profil est fixé une
fois par appel à l'agent (`definir_profil_courant`, appelé par
`agent.run_agent`) et lu depuis cette fermeture par les outils qui en ont
besoin, sur le même principe que `initialiser_donnees()` dans EXAM-S2 pour la
base en mémoire.
"""

from contextvars import ContextVar
from typing import Any, Literal

import networkx as nx
from google.genai import types

from src.admission import serie_satisfait_prerequis
from src.graphe import chemin_competence_parcours_metier, prerequis_du_parcours
from src.graphe import construire_graphe as _construire_graphe
from src.graphe import detecter_incoherences as _detecter_incoherences_graphe
from src.ml.outils import (
    analyser_profil,
    definir_graphe_admission,
    identifier_points_forts,
)
from src.models import CorpusFormations, charger_corpus_formations
from src.schemas import ProfilCandidat

TypeOutil = Literal["consultation", "action"]


class OutilIndisponible(RuntimeError):
    """Le corpus ou le profil courant n'ont pas été initialisés.

    L'API doit appeler `initialiser_corpus()` au démarrage, et `agent.py`
    doit appeler `definir_profil_courant()` avant toute exécution d'outil.
    """


# --- Spécification métier ----------------------------------------------------

OUTILS: list[dict] = [
    {
        "name": "rechercher_formation",
        "type": "consultation",
        "sensible": False,
        "description": (
            "Recherche une mention ou un parcours par mot-clé (nom, sigle, ou "
            "domaine). Retourne les formations correspondantes avec leur mention, "
            "niveau de diplôme et source."
        ),
        "parameters": {
            "mot_cle": {
                "type": "STRING",
                "description": "Nom, sigle (ex. IGGLIA) ou domaine à rechercher",
            }
        },
    },
    {
        "name": "comparer_parcours",
        "type": "consultation",
        "sensible": False,
        "description": (
            "Compare deux parcours : mention, niveau de diplôme, prérequis "
            "d'admission, matières, compétences et débouchés connus."
        ),
        "parameters": {
            "parcours_a": {
                "type": "STRING",
                "description": "Identifiant ou nom du premier parcours",
            },
            "parcours_b": {
                "type": "STRING",
                "description": "Identifiant ou nom du second parcours",
            },
        },
    },
    {
        "name": "analyser_profil_ml",
        "type": "consultation",
        "sensible": False,
        "description": (
            "Fait analyser le profil courant par le modèle de Machine Learning "
            "entraîné et retourne tous les parcours candidats classés par score "
            "d'adéquation. N'a besoin d'aucun paramètre : le profil est celui "
            "déjà construit dans la conversation."
        ),
        "parameters": {},
    },
    {
        "name": "calculer_score_adequation",
        "type": "consultation",
        "sensible": False,
        "description": (
            "Calcule le score d'adéquation du modèle ML entre le profil "
            "courant et un parcours précis."
        ),
        "parameters": {
            "parcours": {"type": "STRING", "description": "Identifiant du parcours à évaluer"}
        },
    },
    {
        "name": "verifier_prerequis",
        "type": "consultation",
        "sensible": False,
        "description": (
            "Vérifie si le profil courant satisfait les prérequis d'admission "
            "connus d'un parcours (série de baccalauréat). Signale une "
            "information manquante plutôt que de deviner si la série de "
            "baccalauréat n'a pas été déclarée."
        ),
        "parameters": {
            "parcours": {"type": "STRING", "description": "Identifiant du parcours"}
        },
    },
    {
        "name": "rechercher_competences",
        "type": "consultation",
        "sensible": False,
        "description": "Recherche les parcours qui développent une compétence donnée.",
        "parameters": {
            "competence": {"type": "STRING", "description": "Nom ou identifiant de la compétence"}
        },
    },
    {
        "name": "identifier_debouches",
        "type": "consultation",
        "sensible": False,
        "description": "Liste les débouchés professionnels connus d'un parcours.",
        "parameters": {
            "parcours": {"type": "STRING", "description": "Identifiant du parcours"}
        },
    },
    {
        "name": "expliquer_recommandation",
        "type": "consultation",
        "sensible": False,
        "description": (
            "Explique pourquoi le modèle ML recommande un parcours précis pour "
            "le profil courant, à partir des traits déclarés qui pèsent le plus "
            "dans le score, et du chemin Compétence → Métier du graphe de "
            "connaissances lorsque ces données sont disponibles."
        ),
        "parameters": {
            "parcours": {"type": "STRING", "description": "Identifiant du parcours à expliquer"}
        },
    },
    {
        "name": "detecter_incoherences",
        "type": "consultation",
        "sensible": False,
        "description": (
            "Analyse le corpus structuré et le graphe de connaissances pour "
            "détecter des incohérences internes : référence orpheline, parcours "
            "sans débouché renseigné, compétence requise pour un métier mais "
            "inaccessible ou dont l'accès n'est pas vérifiable. Sert à répondre "
            "honnêtement sur la fiabilité des données disponibles, jamais à "
            "recommander un parcours."
        ),
        "parameters": {},
    },
]

_NOMS_OUTILS = {o["name"] for o in OUTILS}
OUTILS_SENSIBLES = {o["name"] for o in OUTILS if o["sensible"]}  # vide pour l'instant : aucune
# action destructive dans ce périmètre (que des consultations) — voir AGT-4
# pour la politique d'escalade, qui porte sur la confiance, pas la sensibilité
# d'un outil.


def spec_outil(nom: str) -> dict | None:
    return next((o for o in OUTILS if o["name"] == nom), None)



# --- Déclaration pour le function calling Gemini -----------------------------


def declarer_outils() -> list[types.Tool]:
    outils_declares = []
    for outil in OUTILS:
        proprietes = {
            nom: types.Schema(type=spec["type"], description=spec["description"])
            for nom, spec in outil["parameters"].items()
        }
        outils_declares.append(
            types.FunctionDeclaration(
                name=outil["name"],
                description=outil["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties=proprietes,
                    required=list(outil["parameters"]),
                ),
            )
        )
    return [types.Tool(function_declarations=outils_declares)]


# --- État en mémoire ---------------------------------------------------------

_corpus: CorpusFormations | None = None
_graphe: nx.DiGraph | None = None
# Le profil est **par requête** : FastAPI sert les endpoints synchrones dans un
# pool de threads, et une variable de module se faisait écraser d'une demande à
# l'autre. Voir `definir_profil_courant`.
_profil_courant: ContextVar[ProfilCandidat | None] = ContextVar(
    "profil_courant", default=None
)


def initialiser_corpus(corpus: CorpusFormations | None = None) -> None:
    """Charge le corpus structuré en mémoire et reconstruit le graphe de
    connaissances (ONTO-2) qui en dérive. Appelé au démarrage de l'API — les
    deux restent toujours synchronisés, jamais initialisés séparément, pour
    qu'un test qui bascule sur un corpus jouet (`tools.initialiser_corpus(c)`)
    voie aussi son graphe basculer avec lui.

    Le graphe est construit **avant** toute affectation : si la construction
    échoue, les deux globales gardent leur valeur précédente, cohérente entre
    elles. Affecter `_corpus` d'abord laisserait un corpus neuf face à un
    graphe périmé, et `verifier_prerequis` répondrait alors sur l'admissibilité
    en croisant les deux."""
    global _corpus, _graphe
    nouveau_corpus = corpus if corpus is not None else charger_corpus_formations()
    nouveau_graphe = _construire_graphe(nouveau_corpus)
    _corpus, _graphe = nouveau_corpus, nouveau_graphe
    # Une seule vérité pour le graphe : le volet hybride (`ml.outils`)
    # reconstruisait le sien depuis le disque et le mettait en cache, si bien
    # qu'il jugeait l'admissibilité d'après un corpus que plus personne ne
    # servait. Il reçoit désormais celui-ci.
    definir_graphe_admission(nouveau_graphe)


def definir_profil_courant(profil: ProfilCandidat) -> None:
    """Fixe le profil du candidat pour la durée d'un appel à l'agent.

    **Isolé par requête, pas partagé.** Le profil vivait dans une variable de
    module, alors que FastAPI exécute les endpoints synchrones dans un pool de
    threads : deux demandes simultanées s'écrasaient mutuellement. Mesuré — une
    requête déclarant la série « D » voyait ses outils lire « A », la série de
    l'autre requête. Un candidat se serait fait vérifier son admissibilité sur
    le profil de quelqu'un d'autre, silencieusement.

    `ContextVar` plutôt que `threading.local()` : le comportement est correct
    pour un pool de threads comme pour une éventuelle bascule en `async def`,
    où plusieurs requêtes partagent un même thread.

    Le profil reste hors du function calling, pour la raison expliquée en tête
    de module : demander au LLM de ressaisir un profil entier l'inviterait à en
    inventer ou à en oublier des champs.
    """
    _profil_courant.set(profil)


def corpus_charge() -> CorpusFormations | None:
    """Le corpus actuellement en mémoire, ou `None` s'il n'a pas encore été
    initialisé. Point d'accès public pour l'introspection (ex. `GET /health`),
    plutôt que de lire `_corpus` directement depuis un autre module."""
    return _corpus


def _base() -> CorpusFormations:
    if _corpus is None:
        raise OutilIndisponible(
            "Corpus non initialisé : appeler initialiser_corpus() au démarrage."
        )
    return _corpus


def _profil() -> ProfilCandidat:
    profil = _profil_courant.get()
    if profil is None:
        raise OutilIndisponible(
            "Profil non défini : appeler definir_profil_courant() avant l'agent."
        )
    return profil


def _graphe_actuel() -> nx.DiGraph:
    if _graphe is None:
        raise OutilIndisponible(
            "Graphe de connaissances non initialisé : appeler initialiser_corpus() au démarrage."
        )
    return _graphe


# --- Outils de consultation ---------------------------------------------------


def rechercher_formation(mot_cle: str) -> dict:
    texte = mot_cle.strip().lower()
    corpus = _base()
    mentions = [m for m in corpus.mentions if texte in m.nom.lower() or texte in m.id.lower()]
    parcours = [p for p in corpus.parcours if texte in p.nom.lower() or texte in p.id.lower()]
    if not mentions and not parcours:
        return {"statut": "aucun_resultat", "message": f"Aucune formation pour « {mot_cle} »."}
    return {
        "statut": "trouve",
        "mentions": [
            {"id": m.id, "nom": m.nom, "niveau": m.niveau, "source_id": m.source_id}
            for m in mentions
        ],
        "parcours": [
            {"id": p.id, "nom": p.nom, "mention_id": p.mention_id, "source_id": p.source_id}
            for p in parcours
        ],
    }


def _parcours_par_id_ou_nom(identifiant: str):
    texte = identifiant.strip().lower()
    for p in _base().parcours:
        if texte == p.id.lower() or texte in p.nom.lower():
            return p
    return None


def _mention_de(parcours) -> dict | None:
    if parcours is None:
        return None
    mention = next((m for m in _base().mentions if m.id == parcours.mention_id), None)
    return (
        {
            "id": mention.id,
            "nom": mention.nom,
            "niveau": mention.niveau,
            "source_id": mention.source_id,
        }
        if mention
        else None
    )


def _fiche_parcours(parcours) -> dict:
    return {
        "id": parcours.id,
        "nom": parcours.nom,
        "mention": _mention_de(parcours),
        # Identifiant du registre des sources (DATA-2) : remonté jusqu'à
        # l'agent pour qu'une réponse fondée sur cet outil structuré, et non
        # sur le RAG, reste citable dans `RecommandationDecision.sources`
        # (AGT-6, cf. `agent._source_ids_des_outils`).
        "source_id": parcours.source_id,
        # Même source que `verifier_prerequis` : le graphe (ONTO-3). Deux
        # chemins distincts pour répondre à « quels sont les prérequis de ce
        # parcours ? » finiraient par diverger, et l'agent peut appeler les
        # deux outils dans une même conversation.
        "prerequis": prerequis_du_parcours(_graphe_actuel(), parcours.id),
        "matieres": parcours.matieres,
        "competences": parcours.competences,
        "debouches": parcours.debouches,
        "note_completude": (
            "Matières, compétences et débouchés précis pas encore collectés pour ce "
            "parcours (voir BACKLOG.md, DATA-1)."
            if not (parcours.matieres or parcours.competences or parcours.debouches)
            else None
        ),
    }


def comparer_parcours(parcours_a: str, parcours_b: str) -> dict:
    a = _parcours_par_id_ou_nom(parcours_a)
    b = _parcours_par_id_ou_nom(parcours_b)
    manquants = [n for n, p in ((parcours_a, a), (parcours_b, b)) if p is None]
    if manquants:
        return {
            "statut": "aucun_resultat",
            "message": f"Parcours introuvable(s) : {', '.join(manquants)}.",
        }
    return {
        "statut": "trouve",
        "parcours_a": _fiche_parcours(a),
        "parcours_b": _fiche_parcours(b),
    }


def analyser_profil_ml() -> dict:
    analyse = analyser_profil(_profil())
    return analyse.model_dump()


def calculer_score_adequation(parcours: str) -> dict:
    """Score d'adéquation ML pour un parcours précis.

    Passe par `analyser_profil` plutôt que par `calculer_adequation` afin de
    récupérer le diagnostic d'exploitabilité : un score isolé, sorti d'un profil
    que le modèle n'a pas su exploiter, est indiscernable d'un score informatif
    pour l'agent qui le lit."""
    analyse = analyser_profil(_profil())
    score = next(
        (c.score_adequation for c in analyse.parcours_candidats if c.parcours == parcours), 0.0
    )
    resultat = {
        "parcours": parcours,
        "score_adequation": score,
        "profil_exploitable": analyse.profil_exploitable,
    }
    if not analyse.profil_exploitable:
        resultat["avertissement"] = analyse.justification
    return resultat


def verifier_prerequis(parcours: str) -> dict:
    """Vérifie la compatibilité du profil courant avec les prérequis d'un
    parcours — ONTO-3 : les prérequis sont lus depuis le graphe de
    connaissances (`src.graphe.prerequis_du_parcours`, relation `necessite`),
    une requête déterministe plutôt qu'un filtrage direct du corpus."""
    p = _parcours_par_id_ou_nom(parcours)
    if p is None:
        return {"statut": "aucun_resultat", "message": f"Parcours « {parcours} » introuvable."}

    descriptions = prerequis_du_parcours(_graphe_actuel(), p.id)
    profil = _profil()

    if not descriptions:
        return {
            "statut": "trouve",
            "parcours": p.id,
            "source_id": p.source_id,  # AGT-6 : citable même sans prérequis connu
            "prerequis": [],
            "compatible": None,
            "message": "Aucun prérequis d'admission connu pour ce parcours.",
        }

    # `.strip()` avant le test, pas seulement dans la regex plus bas : une
    # saisie composée uniquement d'espaces passerait le test de vérité, puis
    # produirait un motif vide dont la regex `\b\b` matche n'importe quel
    # texte de prérequis — et l'outil confirmerait l'admissibilité.
    serie_declaree = (profil.serie_bac or "").strip()
    if not serie_declaree:
        return {
            "statut": "information_manquante",
            "parcours": p.id,
            "source_id": p.source_id,
            "prerequis": descriptions,
            "compatible": None,
            "message": (
                "Série de baccalauréat non déclarée : impossible de vérifier "
                "la compatibilité."
            ),
        }

    # Règle partagée avec `ml.hybride` (voir `src/admission.py`) : une seule
    # implémentation, pour que l'outil appelé par l'agent et le filtrage
    # appliqué aux recommandations ML ne puissent pas diverger.
    compatible = serie_satisfait_prerequis(serie_declaree, descriptions)
    return {
        "statut": "trouve",
        "parcours": p.id,
        "source_id": p.source_id,
        "prerequis": descriptions,
        "serie_bac_declaree": serie_declaree,
        "compatible": compatible,
    }


def rechercher_competences(competence: str) -> dict:
    texte = competence.strip().lower()
    corpus = _base()
    ids_competences = {c.id for c in corpus.competences if texte in c.nom.lower()}
    mentionnee_dans_un_parcours = any(
        texte in c.lower() for p in corpus.parcours for c in p.competences
    )
    if not ids_competences and not mentionnee_dans_un_parcours:
        return {
            "statut": "aucun_resultat",
            "message": f"Compétence « {competence} » inconnue du corpus.",
        }

    parcours_correspondants = [
        p.id
        for p in corpus.parcours
        if ids_competences & set(p.competences) or any(texte in c.lower() for c in p.competences)
    ]
    return {"statut": "trouve", "parcours": parcours_correspondants}


def identifier_debouches(parcours: str) -> dict:
    p = _parcours_par_id_ou_nom(parcours)
    if p is None:
        return {"statut": "aucun_resultat", "message": f"Parcours « {parcours} » introuvable."}
    if not p.debouches:
        return {
            "statut": "information_manquante",
            "parcours": p.id,
            "source_id": p.source_id,
            "message": (
                "Débouchés précis non encore collectés pour ce parcours "
                "(voir BACKLOG.md, DATA-1)."
            ),
        }
    corpus = _base()
    noms = [m.nom for m in corpus.metiers if m.id in p.debouches]
    return {
        "statut": "trouve",
        "parcours": p.id,
        "source_id": p.source_id,
        "debouches": noms or p.debouches,
    }


def _raisonnement_graphe(parcours: str) -> list[dict]:
    """Chemin Compétence → Métier du graphe pour ce parcours (ONTO-5).

    Enrichissement, pas une condition : si le graphe n'a pas encore été
    initialisé, dégrade en liste vide plutôt que de faire échouer
    `expliquer_recommandation`, qui reste utilisable sur le seul score ML."""
    if _graphe is None:
        return []
    return chemin_competence_parcours_metier(_graphe, parcours)


def expliquer_recommandation(parcours: str) -> dict:
    """Explique une recommandation : score ML, traits déclarés qui pèsent, et
    chemin Compétence → Métier du graphe (ONTO-5).

    L'argument est résolu comme dans les autres outils qui prennent un
    parcours : le LLM passe indifféremment un sigle ou un nom, et un
    identifiant non résolu produirait un `raisonnement_graphe` vide —
    indiscernable, côté agent, d'une absence réelle de données."""
    p = _parcours_par_id_ou_nom(parcours)
    if p is None:
        return {"statut": "aucun_resultat", "message": f"Parcours « {parcours} » introuvable."}

    profil = _profil()
    # Une seule analyse ML pour le score et le diagnostic : appeler
    # `calculer_adequation` puis `identifier_points_forts` relançait la
    # résolution du profil trois fois au total.
    analyse = analyser_profil(profil)
    score = next(
        (c.score_adequation for c in analyse.parcours_candidats if c.parcours == p.id), 0.0
    )
    resultat = {
        "statut": "trouve",
        "parcours": p.id,
        "source_id": p.source_id,
        "score_adequation": score,
        "profil_exploitable": analyse.profil_exploitable,
        "points_forts": identifier_points_forts(profil),
        "raisonnement_graphe": _raisonnement_graphe(p.id),
    }
    if not analyse.profil_exploitable:
        resultat["avertissement"] = analyse.justification
    return resultat


def detecter_incoherences() -> dict:
    """Détecte les incohérences structurelles du corpus/graphe (ONTO-4).

    Toujours `statut: "trouve"` : l'analyse a abouti dans les deux cas, et un
    corpus sain est un résultat positif. `aucun_resultat` signifie partout
    ailleurs dans ce module « je n'ai pas pu répondre » — le renvoyer ici
    ferait dire à l'agent qu'il n'a aucune information sur la fiabilité des
    données, l'inverse de ce que l'outil vient d'établir."""
    incoherences = _detecter_incoherences_graphe(_base(), _graphe_actuel())
    if not incoherences:
        return {
            "statut": "trouve",
            "nombre": 0,
            "incoherences": [],
            "message": "Aucune incohérence structurelle détectée dans le corpus.",
        }
    return {
        "statut": "trouve",
        "nombre": len(incoherences),
        "incoherences": incoherences,
        "message": (
            f"{len(incoherences)} incohérence(s) structurelle(s) détectée(s). "
            "Les entrées marquées `donnee_manquante` signalent une information "
            "pas encore collectée (voir BACKLOG.md, DATA-1), pas une "
            "contradiction du corpus."
        ),
    }


# --- Registre des exécutions (AGT-3) -----------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "rechercher_formation": rechercher_formation,
    "comparer_parcours": comparer_parcours,
    "analyser_profil_ml": analyser_profil_ml,
    "calculer_score_adequation": calculer_score_adequation,
    "verifier_prerequis": verifier_prerequis,
    "rechercher_competences": rechercher_competences,
    "identifier_debouches": identifier_debouches,
    "expliquer_recommandation": expliquer_recommandation,
    "detecter_incoherences": detecter_incoherences,
}


def valider_parametres(nom: str, params: dict) -> bool:
    spec = spec_outil(nom)
    if spec is None:
        return False
    return all(cle in params for cle in spec["parameters"])


_log_appel: Any = None


def set_log_appel(fonction) -> None:
    """Branche le logger d'appels d'outils (`observability.log_tool_call`).
    Signature attendue : `fonction(trace_id, nom, params, resultat, statut, latence_ms)`."""
    global _log_appel
    _log_appel = fonction


def executer_outil(nom: str, params: dict, trace_id: str) -> dict:
    """Valide puis exécute un outil, en retournant un statut exploitable par
    l'agent — jamais une exception qui remonte jusqu'à l'utilisateur."""
    import time

    t0 = time.time()

    if nom not in _NOMS_OUTILS:
        return {"statut": "erreur", "message": f"Outil inconnu : {nom}"}

    if not valider_parametres(nom, params):
        manquants = [cle for cle in spec_outil(nom)["parameters"] if cle not in params]
        return {"statut": "erreur", "message": f"Paramètres manquants pour {nom} : {manquants}"}

    try:
        resultat = TOOL_REGISTRY[nom](**params)
        statut, contenu = "succes", resultat
    except OutilIndisponible as e:
        statut, contenu = "erreur", str(e)
    except Exception as e:  # noqa: BLE001 — un outil qui échoue ne doit pas planter l'agent
        statut, contenu = "erreur", str(e)

    if _log_appel is not None:
        _log_appel(trace_id, nom, params, contenu, statut, round((time.time() - t0) * 1000))

    return {"statut": statut, "resultat" if statut == "succes" else "message": contenu}
