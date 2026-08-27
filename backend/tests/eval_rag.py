"""Évaluation reproductible du retrieval RAG (RAG-6 / EVAL-3).

Le jeu est stocké dans ``eval_rag.json`` et reste inchangé entre la baseline
vectorielle et RAG-4. Le script n'appelle aucun LLM : il mesure le composant de
recherche lui-même, de façon déterministe et sans quota externe.

    cd backend
    python -m tests.eval_rag --mode vectoriel --sortie tests/eval_results_rag_vectoriel.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src import rag
from src.config import config
from src.models import charger_corpus

DOSSIER_TESTS = Path(__file__).parent
JEU_PAR_DEFAUT = DOSSIER_TESTS / "eval_rag.json"


def charger_jeu(chemin: Path = JEU_PAR_DEFAUT) -> list[dict]:
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)["cas"]


def evaluer(mode: str = "vectoriel") -> dict:
    documents = charger_corpus()
    if not documents:
        raise RuntimeError("Corpus documentaire vide : rien à évaluer.")
    rag.ingerer(documents)

    details = []
    rappels = []
    precisions = []
    silences = []
    for cas in charger_jeu():
        fragments = rag.retrieve_context(cas["question"], mode=mode)
        obtenues = [fragment["source_id"] for fragment in fragments]
        attendues = set(cas["sources_attendues"])
        obtenues_uniques = set(obtenues)

        if attendues:
            rappel = len(attendues & obtenues_uniques) / len(attendues)
            precision = (
                len(attendues & obtenues_uniques) / len(obtenues_uniques)
                if obtenues_uniques
                else 0.0
            )
            rappels.append(rappel)
            precisions.append(precision)
            succes = rappel == 1.0
        else:
            rappel = None
            precision = None
            silence = not obtenues
            silences.append(float(silence))
            succes = silence

        details.append(
            {
                "id": cas["id"],
                "question": cas["question"],
                "sources_attendues": cas["sources_attendues"],
                "sources_obtenues": obtenues,
                "rappel": rappel,
                "precision": precision,
                "succes": succes,
            }
        )

    rappel = sum(rappels) / len(rappels)
    silence = sum(silences) / len(silences)
    return {
        "date": datetime.now(UTC).isoformat(),
        "mode": mode,
        "corpus": {"documents": len(documents), "fragments": rag.nombre_de_fragments()},
        "configuration": {"seuil_vectoriel": config.rag_seuil_pertinence, "k": config.rag_k},
        "metriques": {
            "rappel_sources": rappel,
            "precision_sources": sum(precisions) / len(precisions),
            "silence_correct_hors_corpus": silence,
            "compromis_rappel_silence": (
                0.0
                if rappel + silence == 0
                else 2 * rappel * silence / (rappel + silence)
            ),
            "cas_reussis": sum(detail["succes"] for detail in details),
            "cas_total": len(details),
        },
        "details": details,
        "limite": (
            "Jeu rédigé par l'équipe à partir du corpus, sans questions réelles de candidats. "
            "La précision mesure les sources récupérées, tandis que le garde-fou des citations "
            "générées est couvert séparément par test_rag.py."
        ),
    }


def sauvegarder(resultats: dict, chemin: Path) -> None:
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(resultats, fichier, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("vectoriel", "hybride"), default="vectoriel")
    parser.add_argument("--sortie", type=Path, required=True)
    args = parser.parse_args()
    resultats = evaluer(args.mode)
    sauvegarder(resultats, args.sortie)
    print(json.dumps(resultats["metriques"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
