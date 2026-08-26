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
`expliquer_recommandation`.

**Le profil du candidat n'est jamais un paramètre du function calling.**
Demander au LLM de ressaisir un profil entier en argument JSON l'inviterait à
en inventer ou en oublier des champs — invérifiable. Le profil est fixé une
fois par appel à l'agent (`definir_profil_courant`, appelé par
`agent.run_agent`) et lu depuis cette fermeture par les outils qui en ont
besoin, sur le même principe que `initialiser_donnees()` dans EXAM-S2 pour la
base en mémoire.
"""

import re
from typing import Any, Literal

from google.genai import types

from src.ml.outils import analyser_profil, calculer_adequation, identifier_points_forts
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
            "dans le score."
        ),
        "parameters": {
            "parcours": {"type": "STRING", "description": "Identifiant du parcours à expliquer"}
        },
    },
]

_NOMS_OUTILS = {o["name"] for o in OUTILS}
OUTILS_SENSIBLES = {o["name"] for o in OUTILS if o["sensible"]}  # vide pour l'instant : aucune
# action destructive dans ce périmètre (que des consultations) — voir AGT-4
# pour la politique d'escalade, qui porte sur la confiance, pas la sensibilité
# d'un outil.


def spec_outil(nom: str) -> dict | None:
    return next((o for o in OUTILS if o["name"] == nom), None)


def est_sensible(nom: str) -> bool:
    return nom in OUTILS_SENSIBLES


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
_profil_courant: ProfilCandidat | None = None


def initialiser_corpus(corpus: CorpusFormations | None = None) -> None:
    """Charge le corpus structuré en mémoire. Appelé au démarrage de l'API."""
    global _corpus
    _corpus = corpus if corpus is not None else charger_corpus_formations()


def definir_profil_courant(profil: ProfilCandidat) -> None:
    """Fixe le profil du candidat pour la durée d'un appel à l'agent."""
    global _profil_courant
    _profil_courant = profil


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
    if _profil_courant is None:
        raise OutilIndisponible(
            "Profil non défini : appeler definir_profil_courant() avant l'agent."
        )
    return _profil_courant


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
        "mentions": [{"id": m.id, "nom": m.nom, "niveau": m.niveau} for m in mentions],
        "parcours": [{"id": p.id, "nom": p.nom, "mention_id": p.mention_id} for p in parcours],
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
    return {"id": mention.id, "nom": mention.nom, "niveau": mention.niveau} if mention else None


def _descriptions_prerequis(ids_prerequis: list[str]) -> list[str]:
    corpus = _base()
    return [
        p.description for p in corpus.prerequis if p.id in ids_prerequis
    ]


def _fiche_parcours(parcours) -> dict:
    return {
        "id": parcours.id,
        "nom": parcours.nom,
        "mention": _mention_de(parcours),
        "prerequis": _descriptions_prerequis(parcours.prerequis),
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
    score = calculer_adequation(_profil(), parcours)
    return {"parcours": parcours, "score_adequation": score}


_SERIES_TOUTE = "toute série"


def verifier_prerequis(parcours: str) -> dict:
    p = _parcours_par_id_ou_nom(parcours)
    if p is None:
        return {"statut": "aucun_resultat", "message": f"Parcours « {parcours} » introuvable."}

    descriptions = _descriptions_prerequis(p.prerequis)
    profil = _profil()

    if not descriptions:
        return {
            "statut": "trouve",
            "parcours": p.id,
            "prerequis": [],
            "compatible": None,
            "message": "Aucun prérequis d'admission connu pour ce parcours.",
        }

    if not profil.serie_bac:
        return {
            "statut": "information_manquante",
            "parcours": p.id,
            "prerequis": descriptions,
            "compatible": None,
            "message": (
                "Série de baccalauréat non déclarée : impossible de vérifier "
                "la compatibilité."
            ),
        }

    # Frontière de mot obligatoire : une correspondance par simple sous-chaîne
    # ferait matcher n'importe quelle lettre isolée ("L") contre une lettre
    # présente au milieu d'un mot de la phrase ("baccaLauréat") — trouvé en
    # testant le cas "série L" contre "Baccalauréat série C, D, S".
    serie = re.escape(profil.serie_bac.strip())
    motif_serie = re.compile(rf"\b{serie}\b", re.IGNORECASE)
    compatible = any(
        _SERIES_TOUTE in d.lower() or motif_serie.search(d) for d in descriptions
    )
    return {
        "statut": "trouve",
        "parcours": p.id,
        "prerequis": descriptions,
        "serie_bac_declaree": profil.serie_bac,
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
            "message": (
                "Débouchés précis non encore collectés pour ce parcours "
                "(voir BACKLOG.md, DATA-1)."
            ),
        }
    corpus = _base()
    noms = [m.nom for m in corpus.metiers if m.id in p.debouches]
    return {"statut": "trouve", "parcours": p.id, "debouches": noms or p.debouches}


def expliquer_recommandation(parcours: str) -> dict:
    profil = _profil()
    score = calculer_adequation(profil, parcours)
    points_forts = identifier_points_forts(profil)
    return {
        "parcours": parcours,
        "score_adequation": score,
        "points_forts": points_forts,
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
