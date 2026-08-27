"""Prépare des candidats de `Metier` (débouchés) à partir d'un guide de
correspondance filières ISPM généré automatiquement, hors corpus officiel.

**Ce que c'est, et ce que ce n'est pas.** Le corpus structuré (DATA-1/DATA-3)
n'avait jamais collecté les débouchés métiers par parcours : `metiers.json`
n'existait pas, `Parcours.debouches` était vide partout, et `ONTO-6` recensait
ce manque (16 constats `donnee_manquante`, 0 contradiction). Un fichier
externe généré automatiquement (grand modèle de langage, sans confirmation
ISPM) couvre les 16 parcours avec des intitulés de métiers plausibles — même
statut de provenance que les matières issues des calendriers OCR
(`SRC-CALENDRIERS-FACEBOOK`, déjà en production) : **externe**, jamais
présenté comme officiel, mais utilisable par l'agent avec cette traçabilité
(règle §4 du sujet).

Par défaut, ce script produit des **candidats** dans
`backend/data/a_valider/metiers_candidats.json` — que
`src.models.charger_corpus_formations()` ne charge jamais (nom de fichier
différent de `metiers.json`). Avec `--ecrire-corpus`, il écrit en plus
`backend/data/metiers.json` et met à jour `Parcours.debouches` dans
`backend/data/parcours.json`, sur le même modèle que
`scripts/extraire_matieres.py --ecrire-corpus` — décision produit du
2026-08-27 : fusion telle quelle, sans relecture individuelle des intitulés,
la source restant marquée `externe` au registre pour toute réponse qui les cite.

Usage :
    cd backend && python -m scripts.preparer_candidats_metiers \
        --source <fichier.json> --ecrire-corpus
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

RACINE_DATA = Path(__file__).resolve().parent.parent / "data"

SOURCE_ID_CANDIDATS = "SRC-GUIDE-FILIERES-GENERE"


def _slug(nom: str, prefixe: str) -> str:
    """`"Développeur Web/Mobile"` -> `"MET-DEVELOPPEUR-WEBMOBILE"`."""
    decompose = unicodedata.normalize("NFKD", nom.strip())
    sans_accent = "".join(c for c in decompose if not unicodedata.combining(c))
    caracteres = [c if c.isalnum() else "-" for c in sans_accent.upper()]
    slug = re.sub(r"-+", "-", "".join(caracteres)).strip("-")
    return f"{prefixe}-{slug}"[:60]


def charger_json(chemin: Path) -> list[dict]:
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def charger_mention_par_parcours(dossier_data: Path = RACINE_DATA) -> dict[str, str]:
    """`{"IGGLIA": "Informatique et Télécommunications", ...}`, dérivé du
    corpus réel (`parcours.json` + `mentions.json`) plutôt que ressaisi, pour
    ne pas avoir deux vérités qui peuvent diverger."""
    parcours_path = dossier_data / "parcours.json"
    mentions_path = dossier_data / "mentions.json"
    if not parcours_path.exists() or not mentions_path.exists():
        return {}
    parcours = charger_json(parcours_path)
    mentions = {m["id"]: m["nom"] for m in charger_json(mentions_path)}
    return {p["id"]: mentions.get(p["mention_id"], "") for p in parcours}


def extraire_debouches_par_parcours(
    entrees: list[dict], ids_parcours_connus: set[str]
) -> dict[str, list[str]]:
    """Ne garde que les entrées dont `code_filiere` correspond à un parcours
    réellement présent dans notre corpus (`parcours.json`) — une source
    externe peut nommer des filières qui n'existent pas chez nous, ou sous un
    autre sigle, et on ne veut pas leur inventer une place."""
    resultat: dict[str, list[str]] = {}
    for entree in entrees:
        code = entree.get("code_filiere")
        debouches = entree.get("debouches") or []
        if code not in ids_parcours_connus or not debouches:
            continue
        vus: list[str] = []
        for nom in debouches:
            if nom not in vus:
                vus.append(nom)
        resultat[code] = vus
    return resultat


def construire_candidats_metiers(
    debouches_par_parcours: dict[str, list[str]],
    mention_par_parcours: dict[str, str],
    source_id: str = SOURCE_ID_CANDIDATS,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Dédoublonne les intitulés de métiers **au caractère près** (pas de
    rapprochement flou) : fusionner à tort deux intitulés distincts serait
    plus difficile à détecter, à la relecture humaine, qu'un doublon laissé
    de côté."""
    id_par_nom: dict[str, str] = {}
    metiers: list[dict] = []
    par_parcours: dict[str, list[str]] = {}

    for parcours_id, noms in sorted(debouches_par_parcours.items()):
        ids_pour_ce_parcours: list[str] = []
        for nom in noms:
            if nom not in id_par_nom:
                identifiant = _slug(nom, "MET")
                id_par_nom[nom] = identifiant
                metiers.append(
                    {
                        "id": identifiant,
                        "nom": nom,
                        "secteur": mention_par_parcours.get(parcours_id) or None,
                        "source_id": source_id,
                    }
                )
            ids_pour_ce_parcours.append(id_par_nom[nom])
        par_parcours[parcours_id] = ids_pour_ce_parcours

    return metiers, par_parcours


def ecrire_corpus(
    metiers: list[dict], par_parcours: dict[str, list[str]], dossier_data: Path = RACINE_DATA
) -> dict:
    """Écrit `metiers.json` et met à jour `Parcours.debouches` dans
    `parcours.json` — même geste que `extraire_matieres.ecrire_corpus` pour
    les matières : réécriture intégrale et idempotente des deux fichiers,
    rejouable à chaque nouvelle version de la source."""
    (dossier_data / "metiers.json").write_text(
        json.dumps(metiers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    chemin_parcours = dossier_data / "parcours.json"
    liste_parcours = json.loads(chemin_parcours.read_text(encoding="utf-8"))
    rattaches = 0
    for parcours in liste_parcours:
        trouves = par_parcours.get(parcours["id"])
        if trouves:
            parcours["debouches"] = trouves
            rattaches += 1
    chemin_parcours.write_text(
        json.dumps(liste_parcours, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "metiers": len(metiers),
        "parcours_rattaches": rattaches,
        "parcours_sans_debouche": [p["id"] for p in liste_parcours if not p.get("debouches")],
    }


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument(
        "--source", type=Path, required=True, help="fichier JSON du guide filières généré"
    )
    parseur.add_argument(
        "--sortie",
        type=Path,
        default=RACINE_DATA / "a_valider" / "metiers_candidats.json",
        help="fichier JSON de candidats produit",
    )
    parseur.add_argument(
        "--ecrire-corpus",
        action="store_true",
        help="écrit aussi metiers.json et met à jour parcours.json[].debouches",
    )
    arguments = parseur.parse_args()

    entrees = charger_json(arguments.source)
    mention_par_parcours = charger_mention_par_parcours()
    ids_parcours_connus = set(mention_par_parcours) or {e["id"] for e in charger_json(
        RACINE_DATA / "parcours.json"
    )}

    debouches_par_parcours = extraire_debouches_par_parcours(entrees, ids_parcours_connus)
    metiers, par_parcours = construire_candidats_metiers(
        debouches_par_parcours, mention_par_parcours
    )

    ignores = sorted({e.get("code_filiere") for e in entrees} - set(debouches_par_parcours))

    sortie = {
        "avertissement": (
            "Métiers (débouchés) extraits d'une source externe générée automatiquement, "
            "non confirmée par l'ISPM (voir registre_sources.json, "
            f"{SOURCE_ID_CANDIDATS}, statut externe). Ce fichier est un instantané des "
            "candidats produits par ce script — la donnée qui fait foi, une fois "
            "`--ecrire-corpus` exécuté, vit dans metiers.json/parcours.json[].debouches."
        ),
        "metiers": metiers,
        "debouches_par_parcours": par_parcours,
    }
    if ignores:
        sortie["parcours_source_ignores"] = ignores

    arguments.sortie.parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.sortie, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"{len(metiers)} métiers candidats sur {len(debouches_par_parcours)} parcours "
        f"-> {arguments.sortie}"
    )
    if ignores:
        print(f"Parcours de la source ignorés (absents de notre corpus) : {ignores}")

    if arguments.ecrire_corpus:
        bilan = ecrire_corpus(metiers, par_parcours)
        print(
            f"  metiers.json  : {bilan['metiers']} métiers\n"
            f"  parcours.json : {bilan['parcours_rattaches']} parcours rattachés"
        )
        if bilan["parcours_sans_debouche"]:
            print(f"  sans débouché : {', '.join(bilan['parcours_sans_debouche'])}")


if __name__ == "__main__":
    main()
