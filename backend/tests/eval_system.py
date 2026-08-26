"""Évaluation du système complet, de bout en bout (EVAL-1 à EVAL-5).

Fait tourner les 32 cas de `eval_dataset.json` (§13 du sujet, 8 catégories
imposées) à travers le vrai pipeline (`orchestrator.traiter_demande`) — pas
de mock, appels réels au LLM et au RAG. Mesure ce qui compte (§14 : « preuve
mesurée plutôt qu'affirmation ») : cohérence de l'action retenue, sources
citées, outils appelés, latence par requête, et écrit `eval_results.json`
(EVAL-5), le livrable distinct exigé par le sujet.

    python -m backend.tests.eval_system

Consomme du quota LLM réel (~2 à 5 appels par cas selon les outils
nécessaires) : à lancer une fois avant la démo, pas à chaque exécution de la
suite de tests (ce script n'est pas collecté par pytest, comme
`backend/tests/eval_ml.py`).
"""

import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from src.orchestrator import traiter_demande
from src.schemas import OrientationInput, ProfilCandidat, RecommandationDecision

CHEMIN_DATASET = Path(__file__).parent / "eval_dataset.json"
CHEMIN_RESULTATS = Path(__file__).parent / "eval_results.json"


def charger_dataset(chemin: Path = CHEMIN_DATASET) -> list[dict]:
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def _textes_de_la_decision(decision: RecommandationDecision) -> list[str]:
    return [decision.resume, decision.explication] + [
        p.justification for p in decision.parcours_recommandes
    ]


def _verifier_attendu(decision: RecommandationDecision, attendu: dict) -> list[str]:
    """Retourne les raisons d'échec (liste vide = cas réussi)."""
    echecs = []

    if "actions_acceptables" in attendu and decision.action not in attendu["actions_acceptables"]:
        echecs.append(
            f"action « {decision.action} » absente des actions acceptables "
            f"{attendu['actions_acceptables']}"
        )

    if attendu.get("doit_escalader") and decision.action != "escalade_conseiller":
        echecs.append(f"action « {decision.action} » attendue : escalade_conseiller")

    if attendu.get("aucun_outil_appele") and decision.outils_utilises:
        echecs.append(f"outils appelés alors qu'aucun n'était attendu : {decision.outils_utilises}")

    if attendu.get("doit_appeler_analyser_profil_ml") and (
        "analyser_profil_ml" not in decision.outils_utilises
    ):
        echecs.append("analyser_profil_ml n'a pas été appelé")

    sources_attendues = attendu.get("sources_attendues", [])
    manquantes = [s for s in sources_attendues if s not in decision.sources]
    if manquantes:
        echecs.append(f"sources attendues absentes de la réponse : {manquantes}")

    textes = " ".join(t.lower() for t in _textes_de_la_decision(decision) if t)
    for motif_interdit in attendu.get("ne_doit_pas_contenir", []):
        if motif_interdit.lower() in textes:
            echecs.append(f"contenu interdit trouvé : « {motif_interdit} »")

    return echecs


def evaluer_cas(cas: dict) -> dict:
    profil = ProfilCandidat(**cas["profil"]) if cas.get("profil") else ProfilCandidat()
    entree = OrientationInput(message=cas["message"], profil=profil)

    debut = time.perf_counter()
    reponse = traiter_demande(entree)
    latence_ms = round((time.perf_counter() - debut) * 1000)

    echecs = _verifier_attendu(reponse.decision, cas.get("attendu", {}))

    return {
        "id": cas["id"],
        "categorie": cas["categorie"],
        "reussi": not echecs,
        "echecs": echecs,
        "action": reponse.decision.action,
        "confiance": reponse.decision.confiance,
        "incertitude_declaree": reponse.decision.incertitude_declaree,
        "sources": reponse.decision.sources,
        "outils_utilises": reponse.decision.outils_utilises,
        "latence_ms": latence_ms,
    }


def evaluer_systeme(dataset_path: Path = CHEMIN_DATASET) -> dict:
    dataset = charger_dataset(dataset_path)
    resultats = [evaluer_cas(cas) for cas in dataset]

    par_categorie: dict[str, list[dict]] = defaultdict(list)
    for r in resultats:
        par_categorie[r["categorie"]].append(r)

    taux_par_categorie = {
        cat: {
            "reussis": sum(1 for r in rs if r["reussi"]),
            "total": len(rs),
        }
        for cat, rs in par_categorie.items()
    }

    latences = [r["latence_ms"] for r in resultats]
    actions = Counter(r["action"] for r in resultats)

    return {
        "date": datetime.now(UTC).isoformat(),
        "taille_dataset": len(dataset),
        "reussis": sum(1 for r in resultats if r["reussi"]),
        "total": len(resultats),
        "taux_par_categorie": taux_par_categorie,
        "repartition_actions": dict(actions),
        "latence_ms": {
            "moyenne": round(sum(latences) / len(latences)) if latences else 0,
            "min": min(latences) if latences else 0,
            "max": max(latences) if latences else 0,
        },
        "resultats_detailles": resultats,
    }


def sauvegarder(resultats: dict, chemin: Path = CHEMIN_RESULTATS) -> Path:
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    return chemin


if __name__ == "__main__":
    resultats = evaluer_systeme()
    chemin = sauvegarder(resultats)
    print(
        f"{resultats['reussis']}/{resultats['total']} cas réussis — "
        f"latence moyenne {resultats['latence_ms']['moyenne']} ms"
    )
    for cat, stats in resultats["taux_par_categorie"].items():
        print(f"  {cat} : {stats['reussis']}/{stats['total']}")
    print(f"\nRésultats détaillés écrits dans {chemin}")
