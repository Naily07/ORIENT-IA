"""Calibration mesurée du seuil de pertinence et de k (RAG-5, §14 du sujet).

**Pourquoi ce script existe.** `config.rag_seuil_pertinence = 0.75` et
`config.rag_k = 8` sont hérités d'un autre projet, sur un corpus de support
informatique. `config.py` le dit lui-même : « valeurs de départ héritées d'un
autre corpus, à recalibrer sur le corpus pédagogique ISPM, pas à prendre pour
acquises ». Tant que la mesure n'est pas faite, deux des dimensions exigées au
§14 — « pertinence des documents récupérés » et « rappel des sources utiles » —
restent invérifiées.

**Le piège à ne pas redécouvrir.** ChromaDB indexe en L2 au carré par défaut,
pas en cosinus. `rag._collection()` force explicitement `hnsw:space=cosine` ;
sans cela, un seuil « raisonnable » donne 0 % de rappel. Les distances
mesurées ici sont donc des distances cosinus, dans [0, 2].

**Méthode.** Un jeu de questions dont la source attendue est connue, plus des
questions volontairement hors corpus. Pour chaque couple (seuil, k) on mesure :

- le **rappel** : la source attendue est-elle dans les passages retournés ?
- la **précision** : quelle part des passages retournés vient de la bonne
  source ?
- le **taux de silence correct** : sur une question hors corpus, ne rien
  retourner est le comportement voulu, pas un échec (§9 : « reconnaître les
  situations dans lesquelles les informations disponibles ne permettent pas de
  conclure »).

Un seuil trop haut ramène du bruit sur les questions hors corpus ; un seuil
trop bas fait chuter le rappel. Le compromis retenu doit être **lu dans le
tableau**, pas supposé.

    cd backend && python -m tests.calibrer_seuil_rag
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from src import rag
from src.config import config
from src.models import charger_corpus

# Questions dont la source attendue est connue, écrites à partir du corpus réel
# (`backend/data/corpus.json`) sans recopier ses formulations : une question qui
# reprend mot pour mot le document mesurerait la recherche exacte, pas la
# recherche sémantique.
QUESTIONS_AVEC_SOURCE: list[dict] = [
    {"question": "Quelle filière mêle informatique de gestion et intelligence artificielle ?",
     "source_attendue": "DOC-IGGLIA"},
    {"question": "Quel parcours combine électronique et structure des ordinateurs ?",
     "source_attendue": "DOC-ESIIA"},
    {"question": "Où étudier les statistiques appliquées à la banque et au commerce ?",
     "source_attendue": "DOC-ISAIA"},
    {"question": "Quelle formation porte sur le multimédia et les technologies de communication ?",
     "source_attendue": "DOC-IMTICIA"},
    {"question": "Quel parcours prépare aux industries chimiques et minières ?",
     "source_attendue": "DOC-ICMP"},
    {"question": "Quelle filière traite des plantes médicinales de Madagascar ?",
     "source_attendue": "DOC-PIP"},
    {"question": "Quel parcours forme à l'aménagement urbain et aux infrastructures ?",
     "source_attendue": "DOC-GCA"},
    {"question": "Quelle formation vise l'hôtellerie et le patrimoine culturel ?",
     "source_attendue": "DOC-TEH"},
    {"question": "Quels sont les niveaux de diplôme délivrés par l'établissement ?",
     "source_attendue": "DOC-NIVEAUX-DIPLOMES"},
    {"question": "Quelle série de baccalauréat faut-il pour s'inscrire ?",
     "source_attendue": "DOC-CONDITIONS-ADMISSION"},
    {"question": "Combien coûtent les droits d'inscription ?",
     "source_attendue": "DOC-CONDITIONS-ADMISSION"},
    {"question": "Quel parcours mêle droit et outils informatiques ?",
     "source_attendue": "DOC-DTJA"},
]

# Questions volontairement hors corpus : le comportement attendu est de ne
# **rien** retourner. C'est ce qui protège de l'invention d'une formation (§16).
QUESTIONS_HORS_CORPUS: list[str] = [
    "Quel est le programme de la filière de médecine vétérinaire ?",
    "Comment obtenir une bourse Erasmus pour partir au Canada ?",
    "Quelles sont les horaires d'ouverture de la piscine municipale ?",
    "Quel est le classement mondial de l'établissement en robotique ?",
]

# Grille resserrée entre 0,50 et 0,65 : le premier balayage large a montré que
# tout se joue là. Au-dessus de 0,65 le silence sur les questions hors corpus
# tombe à zéro (le RAG répond toujours quelque chose, ce qui invite le modèle à
# broder), en dessous de 0,50 le rappel s'effondre.
SEUILS = (0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.75)
VALEURS_K = (3, 5, 8)


def _mesurer(seuil: float, k: int) -> dict:
    rappels: list[float] = []
    precisions: list[float] = []
    distances_bonnes: list[float] = []

    for cas in QUESTIONS_AVEC_SOURCE:
        fragments = rag.retrieve_context(
            cas["question"], k=k, seuil=seuil, mode="vectoriel"
        )
        sources = [f["source_id"] for f in fragments]
        trouve = cas["source_attendue"] in sources
        rappels.append(float(trouve))
        precisions.append(
            sources.count(cas["source_attendue"]) / len(sources) if sources else 0.0
        )
        if trouve:
            distances_bonnes.extend(
                f["distance"] for f in fragments if f["source_id"] == cas["source_attendue"]
            )

    silences = [
        1.0 if not rag.retrieve_context(q, k=k, seuil=seuil, mode="vectoriel") else 0.0
        for q in QUESTIONS_HORS_CORPUS
    ]

    rappel = sum(rappels) / len(rappels)
    silence = sum(silences) / len(silences)
    return {
        "seuil": seuil,
        "k": k,
        "rappel": rappel,
        "precision_moyenne": sum(precisions) / len(precisions),
        "silence_correct_hors_corpus": silence,
        # Moyenne harmonique rappel / silence : un seuil qui ramène tout obtient
        # un rappel parfait et un silence nul, et doit être pénalisé comme tel.
        "compromis": (
            0.0 if (rappel + silence) == 0 else 2 * rappel * silence / (rappel + silence)
        ),
        "distance_moyenne_bonne_source": (
            sum(distances_bonnes) / len(distances_bonnes) if distances_bonnes else None
        ),
    }


def calibrer() -> dict:
    documents = charger_corpus()
    if not documents:
        raise RuntimeError("Corpus documentaire vide : rien à calibrer.")
    rag.ingerer(documents)

    mesures = [_mesurer(seuil, k) for k in VALEURS_K for seuil in SEUILS]
    meilleur = max(mesures, key=lambda m: (m["compromis"], m["rappel"]))

    return {
        "date": datetime.now(UTC).isoformat(),
        "corpus": {"documents": len(documents), "fragments": rag.nombre_de_fragments()},
        "espace_de_distance": "cosinus (forcé explicitement, voir rag._collection)",
        "questions": {
            "avec_source_attendue": len(QUESTIONS_AVEC_SOURCE),
            "hors_corpus": len(QUESTIONS_HORS_CORPUS),
        },
        "configuration_actuelle": {"seuil": config.rag_seuil_pertinence, "k": config.rag_k},
        "mesures": mesures,
        "meilleur_compromis": meilleur,
        "limite": (
            f"{len(QUESTIONS_AVEC_SOURCE)} questions seulement, écrites par l'équipe à "
            "partir du corpus : l'intervalle de confiance est large et le jeu ne couvre "
            "pas les formulations d'un vrai candidat. À élargir avec les questions "
            "réellement posées pendant la démonstration."
        ),
    }


def sauvegarder(resultats: dict, chemin: Path | None = None) -> Path:
    chemin = chemin or (Path(__file__).parent / "eval_results_rag_calibration.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    return chemin


if __name__ == "__main__":
    resultats = calibrer()
    chemin = sauvegarder(resultats)

    print(f"{'seuil':>7} {'k':>3} {'rappel':>8} {'precis.':>8} {'silence':>8} {'compromis':>10}")
    for m in resultats["mesures"]:
        print(
            f"{m['seuil']:>7} {m['k']:>3} {m['rappel']:>8.2f} "
            f"{m['precision_moyenne']:>8.2f} {m['silence_correct_hors_corpus']:>8.2f} "
            f"{m['compromis']:>10.3f}"
        )
    meilleur = resultats["meilleur_compromis"]
    print(f"\nMeilleur compromis : seuil={meilleur['seuil']} k={meilleur['k']}")
    print(f"Configuration actuelle : {resultats['configuration_actuelle']}")
    print(f"\nRésultats écrits dans {chemin}")
