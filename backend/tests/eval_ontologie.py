"""Preuve mesurée de l'apport de l'ontologie/graphe de connaissances (ONTO-6).

Le sujet est explicite (§12) : « l'usage d'une ontologie n'est pas
obligatoire, mais son apport devra être démontré pour être valorisé ». Ce
script compare, sur des cas concrets, ce que le système établit *avec* le
graphe de connaissances contre ce qu'aucun outil de `tools.py` n'établissait
*sans* lui.

Les deux mesures passent par les **outils réellement exposés à l'agent**
(`src.tools`), pas par les fonctions de `src.graphe` appelées directement :
une preuve qui court-circuite le chemin de production ne prouve rien sur ce
que l'agent obtiendra à l'exécution.

1. Sur le **corpus réel ISPM** (`charger_corpus_formations()`, 16 parcours,
   DATA-1) : `tools.detecter_incoherences` (ONTO-4) n'a pas d'équivalent sans
   le graphe — capacité entièrement nouvelle, mesurée directement sur les
   données réelles du projet.
2. Sur un **corpus jouet complet** (`tests.corpus_jouet.corpus_coherent`, ce
   que DATA-1 vise à terme) : `tools.expliquer_recommandation` (ONTO-5) est
   appelé une fois, et le résultat est présenté deux fois — restreint aux
   champs qui existaient avant ONTO-5, puis complet. L'écart entre les deux
   est exactement ce que le graphe ajoute.

Le corpus jouet est nécessaire au point 2 parce que le corpus réel ne
contient encore aucune compétence ni aucun métier structuré (DATA-1
incomplet, voir BACKLOG.md) : y mesurer le raisonnement multiétape donnerait
un résultat vide qui ne prouverait rien sur le mécanisme. Point 1, lui, se
mesure honnêtement dès aujourd'hui sur le vrai corpus.

    cd backend && python -m tests.eval_ontologie
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from src import tools
from src.models import charger_corpus_formations
from src.schemas import ProfilCandidat
from tests.corpus_jouet import corpus_coherent

# Champs que `expliquer_recommandation` retournait avant ONTO-5 : le score du
# modèle ML et les traits déclarés qui pèsent le plus. Tout le reste est
# l'apport du graphe.
CHAMPS_AVANT_ONTO5 = ("statut", "parcours", "score_adequation", "points_forts")


def _compter_par_type(incoherences: list[dict]) -> dict[str, int]:
    compte: dict[str, int] = {}
    for incoherence in incoherences:
        compte[incoherence["type"]] = compte.get(incoherence["type"], 0) + 1
    return compte


def _mesurer_corpus_reel() -> dict:
    tools.initialiser_corpus(charger_corpus_formations())
    corpus = charger_corpus_formations()
    resultat = tools.detecter_incoherences()
    incoherences = resultat["incoherences"]
    donnees_manquantes = [i for i in incoherences if i.get("donnee_manquante")]

    return {
        "nombre_parcours": len(corpus.parcours),
        "detecter_incoherences": {
            "sans_graphe": (
                "mécanisme inexistant : aucun outil de tools.py n'examinait la "
                "cohérence structurelle du corpus avant ONTO-4"
            ),
            "avec_graphe": {
                "statut": resultat["statut"],
                "nombre": resultat["nombre"],
                "par_type": _compter_par_type(incoherences),
                "dont_donnees_non_collectees": len(donnees_manquantes),
                "dont_contradictions_reelles": len(incoherences) - len(donnees_manquantes),
                "message": resultat["message"],
            },
        },
        "lecture": (
            "Les constats remontés sur le corpus réel sont aujourd'hui tous des "
            "données non encore collectées (débouchés, DATA-1), pas des "
            "contradictions : le champ `donnee_manquante` et le message de "
            "l'outil les distinguent explicitement pour que l'agent ne présente "
            "pas un chantier de collecte comme un défaut de fiabilité. Zéro "
            "contradiction réelle est le résultat attendu sur un corpus dont "
            "chaque entrée a été saisie avec sa source (DATA-2)."
        ),
    }


def _mesurer_corpus_jouet() -> dict:
    tools.initialiser_corpus(corpus_coherent())
    tools.definir_profil_courant(ProfilCandidat(competences_declarees=["programmation"]))

    # Un seul appel au vrai outil de l'agent : les deux vues en sont dérivées.
    explication = tools.expliquer_recommandation("IGGLIA")
    avant_onto5 = {c: explication[c] for c in CHAMPS_AVANT_ONTO5}

    incoherences = tools.detecter_incoherences()

    return {
        "objectif": (
            "Démontrer expliquer_recommandation/ONTO-5 sur un corpus aussi "
            "complet que celui visé par DATA-1 à terme, puisque le corpus réel "
            "ne l'est pas encore sur les compétences/métiers."
        ),
        "expliquer_recommandation_IGGLIA": {
            "sans_graphe_avant_onto5": avant_onto5,
            "avec_graphe_onto5": explication,
            "apport_mesure": {
                "champs_ajoutes": sorted(set(explication) - set(avant_onto5)),
                "nombre_de_chemins_competence_metier": len(explication["raisonnement_graphe"]),
            },
        },
        "detecter_incoherences": {
            "nombre": incoherences["nombre"],
            "par_type": _compter_par_type(incoherences["incoherences"]),
            "detail": incoherences["incoherences"],
        },
    }


def evaluer_ontologie() -> dict:
    resultats = {
        "date": datetime.now(UTC).isoformat(),
        "corpus_reel_ispm": _mesurer_corpus_reel(),
        "corpus_jouet_complet": _mesurer_corpus_jouet(),
        "conclusion": (
            "Le graphe ajoute deux capacités qu'aucun outil sans graphe ne "
            "couvrait : (1) une détection systématique d'incohérences "
            "structurelles (ONTO-4), exercée dès aujourd'hui sur le corpus "
            "réel ISPM via l'outil que l'agent appelle ; (2) un chemin explicite "
            "Compétence -> Parcours -> Métier dans l'explication d'une "
            "recommandation (ONTO-5), mesuré comme l'écart entre la sortie de "
            "expliquer_recommandation avant et après le graphe, sur un corpus "
            "complet en attendant que DATA-1 fournisse compétences et métiers "
            "pour le corpus réel."
        ),
    }
    # L'évaluation a piloté l'état global de tools : le remettre sur le corpus
    # réel pour ne pas laisser le processus dans un état surprenant.
    tools.initialiser_corpus()
    return resultats


def sauvegarder(resultats: dict, chemin: Path | None = None) -> Path:
    chemin = chemin or (Path(__file__).parent / "eval_results_ontologie.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    return chemin


if __name__ == "__main__":
    resultats = evaluer_ontologie()
    chemin = sauvegarder(resultats)
    print(json.dumps(resultats, indent=2, ensure_ascii=False))
    print(f"\nRésultats écrits dans {chemin}")
