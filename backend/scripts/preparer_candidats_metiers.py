"""Prépare des candidats de `Metier` (débouchés) à partir d'un guide de
correspondance filières ISPM généré automatiquement, hors corpus officiel.

**Ce que c'est, et ce que ce n'est pas.** Le corpus structuré (DATA-1/DATA-3)
n'a jamais collecté les débouchés métiers par parcours : `metiers.json`
n'existe pas, `Parcours.debouches` est vide partout, et `ONTO-6` recense ce
manque (16 constats `donnee_manquante`, 0 contradiction). Un fichier externe
généré automatiquement (grand modèle de langage, sans confirmation ISPM)
couvre les 16 parcours avec des intitulés de métiers plausibles. La règle non
négociable du §4 (« une information non vérifiée ne doit pas être présentée
comme officielle ») interdit de le fusionner tel quel dans `metiers.json`.

Ce script produit donc des **candidats à valider par un humain**, dans
`backend/data/a_valider/metiers_candidats.json` — un fichier que
`src.models.charger_corpus_formations()` ne charge jamais (nom de fichier
différent de `metiers.json`), donc sans effet sur les réponses de l'agent
tant que personne ne l'a relu et fusionné à la main.

Usage :
    cd backend && python -m scripts.preparer_candidats_metiers --source <fichier.json>
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
        "a_valider": True,
        "avertissement": (
            "Candidats de métiers (débouchés) extraits d'une source externe générée "
            "automatiquement, non confirmée par l'ISPM (voir registre_sources.json, "
            f"{SOURCE_ID_CANDIDATS}). Ne pas fusionner dans metiers.json ou "
            "parcours.json[].debouches sans relecture humaine (règle §4)."
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


if __name__ == "__main__":
    main()
