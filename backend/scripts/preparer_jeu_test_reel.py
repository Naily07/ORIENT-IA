"""Construit le jeu de test ML **réel et gelé** (DATA-7/ML-7) à partir d'une
enquête de terrain exportée en CSV, et son registre de collecte (DATA-5).

**Pourquoi un script plutôt qu'un fichier livré tel quel.** L'export brut
contient des réponses individuelles (texte libre, horodatage précis) que le
consentement recueilli limite à un usage anonymisé. Ce script ne lit donc le
CSV brut qu'en entrée (`--source`, jamais commité) et n'écrit que des dérivés
anonymisés : `data/ml/jeu_test_reel.json` (profils au format `ProfilCandidat`,
prêts pour ML-7) et `data/enquete/registre_collecte.md` (DATA-5).

**Ce que ce jeu de test est, et ce qu'il n'est pas.** C'est un jeu de
*validation/test*, jamais d'entraînement : les 800 profils synthétiques
(`donnees_synthetiques.py`, DATA-6) restent le jeu d'entraînement, cette
enquête reste la seule mesure de généralisation à de vrais candidats
(ML-7). Champs jamais collectés par ce questionnaire (`serie_bac`,
`activites_projets`, `competences_declarees`, `centres_interet`,
`environnement_travail_recherche`) restent explicitement vides plutôt que
d'être inventés — même principe que `analyser_couverture` (ML-2) : un champ
manquant doit rester manquant, pas silencieusement complété.

**La question notes n'interroge que « maths/info » en une seule échelle
1-5.** Contrairement au corpus synthétique (une note par matière), il n'y a
qu'un seul niveau auto-déclaré ici. On le reporte donc à l'identique sur
`mathematiques` et `informatique` dans `resultats_scolaires` : c'est fidèle à
ce qui a été demandé (un jugement combiné sur les deux matières), pas une
invention d'une précision par matière qui n'a jamais été recueillie.

Usage :
    cd backend && python -m scripts.preparer_jeu_test_reel --source <export.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.guardrails import masquer_objet

RACINE_DATA = Path(__file__).resolve().parent.parent / "data"

# --- Colonnes de l'export (texte exact des questions) --------------------------

COL_TIMESTAMP = "Timestamp"
COL_TYPE = "Êtes-vous actuellement étudiant(e) ou déjà en poste professionnellement ?"
COL_PARCOURS_ACTUEL = "Parcours/mention actuel"
COL_MATIERES_PREFEREES = "Matières préférées avant l'ISPM"
COL_NOTE_DECLAREE = "Résultats scolaires approximatifs (maths/info)"
COL_SATISFACTION = "Satisfaction actuelle de ce choix"
COL_PARCOURS_SUIVI = "Parcours suivi + année de diplôme"
COL_METIER = "Métier exercé aujourd'hui"
COL_ADAPTATION = "Ce parcours était-il adapté à votre métier actuel ?"

TYPE_MAP = {
    "Étudiant(e)": "etudiant",
    "Professionnel(le) (diplômé ISPM)": "professionnel",
}

_MENTIONS_MATIERES = ("mathematiques", "informatique")


def parser_type(valeur: str | None) -> str | None:
    return TYPE_MAP.get((valeur or "").strip())


def convertir_echelle_vers_note20(valeur: str | None) -> float | None:
    """Échelle auto-déclarée 1-5 -> note /20, mise à l'échelle linéaire.

    Un simple `valeur * 4` : pas de palier caché à justifier, une note de 5/5
    devient 20/20 et une note de 1/5 devient 4/20."""
    if valeur is None or not str(valeur).strip():
        return None
    try:
        entier = int(float(valeur))
    except ValueError:
        return None
    if not 1 <= entier <= 5:
        return None
    return float(entier * 4)


def extraire_termes(valeur: str | None) -> list[str]:
    """`"Mathématiques, Informatique"` -> `["Mathématiques", "Informatique"]`,
    dédoublonné et sans terme vide."""
    if not valeur or not valeur.strip():
        return []
    vus: list[str] = []
    for terme in valeur.split(","):
        terme = terme.strip()
        if terme and terme not in vus:
            vus.append(terme)
    return vus


_NON_REPONSE = {"", "anonyme", "aucun", "n/a", "na", "-"}


def normaliser_metier_declare(valeur: str | None) -> str | None:
    """`"Anonyme"` ou une non-réponse équivalente n'est pas un métier —
    plutôt une absence de réponse que le formulaire ne distinguait pas d'une
    vraie déclaration. La confondre avec un intitulé de poste fabriquerait un
    trait qui n'a jamais été déclaré."""
    if valeur is None:
        return None
    nettoye = valeur.strip()
    if not nettoye or nettoye.lower() in _NON_REPONSE:
        return None
    return nettoye


def resoudre_parcours_ou_mention(
    texte: str | None, ids_parcours: set[str], noms_mentions: dict[str, str]
) -> tuple[str | None, str | None, str]:
    """Cherche un code de parcours connu (mot entier, insensible à la casse)
    dans `texte` ; à défaut, un nom de mention connu en sous-chaîne.

    Retourne `(parcours_id, mention_id, granularite)` avec `granularite` ∈
    `{"parcours", "mention", "aucune"}`. Ne jamais deviner un parcours précis
    à partir d'un seul nom de mention : plusieurs parcours y cohabitent
    (ex. Biotechnologie et Agronomie -> AEE, IAA **ou** PIP), un choix
    arbitraire fabriquerait une étiquette fausse plutôt qu'absente.
    """
    if not texte or not texte.strip():
        return None, None, "aucune"

    for code in ids_parcours:
        if re.search(rf"\b{re.escape(code)}\b", texte, re.IGNORECASE):
            return code, None, "parcours"

    texte_normalise = texte.lower()
    for mention_id, nom in noms_mentions.items():
        if nom.lower() in texte_normalise:
            return None, mention_id, "mention"

    return None, None, "aucune"


def charger_referentiel(dossier_data: Path = RACINE_DATA) -> tuple[set[str], dict[str, str]]:
    """`(ids_parcours, {mention_id: nom})`, lus depuis le corpus réel plutôt
    que ressaisis, pour rester alignés si le corpus évolue."""
    with open(dossier_data / "parcours.json", encoding="utf-8") as f:
        ids_parcours = {p["id"] for p in json.load(f)}
    with open(dossier_data / "mentions.json", encoding="utf-8") as f:
        noms_mentions = {m["id"]: m["nom"] for m in json.load(f)}
    return ids_parcours, noms_mentions


def construire_enregistrement(
    ligne: dict[str, str],
    numero: int,
    ids_parcours: set[str],
    noms_mentions: dict[str, str],
    seuil_satisfaction: int,
) -> dict | None:
    type_repondant = parser_type(ligne.get(COL_TYPE))
    if type_repondant is None:
        return None

    if type_repondant == "etudiant":
        label_brut = ligne.get(COL_PARCOURS_ACTUEL)
        matieres_preferees = extraire_termes(ligne.get(COL_MATIERES_PREFEREES))
        note = convertir_echelle_vers_note20(ligne.get(COL_NOTE_DECLAREE))
        resultats_scolaires = {m: note for m in _MENTIONS_MATIERES} if note is not None else {}
        preferences_professionnelles: list[str] = []
        satisfaction_brute = ligne.get(COL_SATISFACTION)
    else:
        label_brut = ligne.get(COL_PARCOURS_SUIVI)
        matieres_preferees = []
        resultats_scolaires = {}
        metier = normaliser_metier_declare(ligne.get(COL_METIER))
        preferences_professionnelles = [metier] if metier else []
        satisfaction_brute = ligne.get(COL_ADAPTATION)

    try:
        satisfaction = int(float(satisfaction_brute)) if satisfaction_brute else None
    except ValueError:
        satisfaction = None

    parcours_id, mention_id, granularite = resoudre_parcours_ou_mention(
        label_brut, ids_parcours, noms_mentions
    )

    profil = {
        "matieres_preferees": matieres_preferees,
        "resultats_scolaires": resultats_scolaires,
        "competences_declarees": [],
        "centres_interet": [],
        "activites_projets": [],
        "preferences_professionnelles": preferences_professionnelles,
        "environnement_travail_recherche": None,
        "serie_bac": None,
        "informations_manquantes": [],
    }

    return {
        "id": f"reel_{numero:04d}",
        "type": type_repondant,
        "profil": profil,
        "parcours_id": parcours_id,
        "mention_id": mention_id,
        "granularite_label": granularite,
        "label_brut": label_brut,
        "satisfaction": satisfaction,
        "label_fiable": satisfaction is not None and satisfaction >= seuil_satisfaction,
        "usable_pour_eval": parcours_id is not None,
    }


def anonymiser(enregistrement: dict) -> dict:
    """Masque défensivement e-mails/secrets résiduels dans le texte libre
    conservé (métier déclaré) — au cas où une réponse en contiendrait, jamais
    prévu par le formulaire."""
    return masquer_objet(enregistrement)


def date_seule(horodatage: str | None) -> str | None:
    if not horodatage:
        return None
    try:
        return datetime.strptime(horodatage, "%m/%d/%Y %H:%M:%S").date().isoformat()
    except ValueError:
        return None


def ecrire_registre_collecte(chemin: Path, enregistrements: list[dict], dates: list[str]) -> None:
    """DATA-5 : registre de collecte de l'enquête (populations, période,
    volumes, procédure d'anonymisation, biais constatés) — calculé sur les
    données réellement traitées, pas déclaré à part."""
    total = len(enregistrements)
    etudiants = sum(1 for e in enregistrements if e["type"] == "etudiant")
    professionnels = total - etudiants
    usables = sum(1 for e in enregistrements if e["usable_pour_eval"])
    fiables = sum(1 for e in enregistrements if e["usable_pour_eval"] and e["label_fiable"])
    granularites = Counter(e["granularite_label"] for e in enregistrements)

    distribution_mentions: Counter[str] = Counter()
    for e in enregistrements:
        if e["mention_id"]:
            distribution_mentions[e["mention_id"]] += 1
        elif e["parcours_id"]:
            distribution_mentions[e["parcours_id"]] += 1

    periode = f"{min(dates)} → {max(dates)}" if dates else "non déterminée"

    lignes = [
        "# Registre de collecte de l'enquête (DATA-5)",
        "",
        "Généré automatiquement par `scripts/preparer_jeu_test_reel.py` à partir "
        "de l'export anonymisé — ne pas éditer à la main, relancer le script.",
        "",
        "## Population et période",
        "",
        f"- **Réponses reçues** : {total}",
        f"- **Étudiants actuels** : {etudiants}",
        f"- **Professionnels diplômés** : {professionnels}",
        f"- **Période de collecte** : {periode}",
        "",
        "## Volumes retenus pour l'évaluation ML (ML-7)",
        "",
        f"- **Parcours ou mention reconnu** : {usables}/{total}",
        f"  - dont étiquette au niveau **parcours** (précise) : {granularites.get('parcours', 0)}",
        f"  - dont étiquette au niveau **mention** seulement (ambiguë entre plusieurs "
        f"parcours, exclue des métriques par parcours) : {granularites.get('mention', 0)}",
        f"  - **non reconnu** (réponse libre non rattachable, écartée) : "
        f"{granularites.get('aucune', 0)}",
        f"- **Étiquette jugée fiable** (satisfaction/adéquation déclarée ≥ 3/5) : {fiables}",
        "",
        "## Texte de consentement recueilli",
        "",
        "> J'accepte que mes réponses anonymisées soient utilisées dans le cadre "
        "d'un projet académique de l'ISPM (hackathon ORIENT'IA). Aucune information "
        "permettant de m'identifier ne sera collectée.",
        "",
        "## Procédure d'anonymisation appliquée",
        "",
        "- Colonne de consentement retirée avant tout traitement.",
        "- Horodatage réduit à la date (heure/minute/seconde supprimées).",
        "- `guardrails.masquer_objet` appliqué à chaque enregistrement produit "
        "(masque e-mails et motifs de secret résiduels dans le texte libre conservé).",
        "- Réponse « Anonyme »/« Aucun »/équivalent sur le métier déclaré traitée "
        "comme une non-réponse, jamais comme un intitulé de poste.",
        "- Aucun champ nominatif n'a été collecté par le formulaire source "
        "(consentement ci-dessus) ; le texte libre conservé dans le jeu de test "
        "final se limite aux matières préférées et à l'intitulé de poste déclaré.",
        "",
        "## Biais et limites constatés",
        "",
        "- Échantillon fortement concentré sur les mentions déjà identifiées : "
        + ", ".join(f"{k} ({v})" for k, v in distribution_mentions.most_common())
        + ".",
        "- La question de niveau scolaire ne porte que sur un jugement combiné "
        "« maths/info » (échelle 1-5), jamais une note par matière : reporté à "
        "l'identique sur `mathematiques` et `informatique`, ce qui sous-estime la "
        "variance réelle entre ces deux matières.",
        "- Aucune réponse ne renseigne `serie_bac`, `activites_projets`, "
        "`competences_declarees`, `centres_interet` ni "
        "`environnement_travail_recherche` : ces champs restent vides dans le jeu "
        "de test, jamais complétés par supposition.",
        "- Échantillon de petite taille (moins de 100 réponses) : un recoupement "
        "entre `label_brut` (parcours + année) et le métier déclaré pourrait, en "
        "théorie, permettre à une personne connaissant la promotion concernée "
        "d'identifier un répondant. Ce risque de ré-identification par petits "
        "effectifs n'est pas éliminé par le masquage de motifs (e-mails/secrets) "
        "appliqué ici — à garder à l'esprit avant toute diffusion au-delà de "
        "l'équipe projet.",
    ]
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--source", type=Path, required=True, help="export CSV brut de l'enquête")
    parseur.add_argument(
        "--sortie", type=Path, default=RACINE_DATA / "ml" / "jeu_test_reel.json"
    )
    parseur.add_argument(
        "--registre-collecte", type=Path, default=RACINE_DATA / "enquete" / "registre_collecte.md"
    )
    parseur.add_argument(
        "--seuil-satisfaction",
        type=int,
        default=3,
        help="satisfaction/adéquation minimale (/5) pour juger l'étiquette fiable",
    )
    arguments = parseur.parse_args()

    ids_parcours, noms_mentions = charger_referentiel()

    enregistrements: list[dict] = []
    dates: list[str] = []
    with open(arguments.source, encoding="utf-8-sig", newline="") as f:
        for numero, ligne in enumerate(csv.DictReader(f), start=1):
            enregistrement = construire_enregistrement(
                ligne, numero, ids_parcours, noms_mentions, arguments.seuil_satisfaction
            )
            if enregistrement is None:
                continue
            enregistrements.append(anonymiser(enregistrement))
            date = date_seule(ligne.get(COL_TIMESTAMP))
            if date:
                dates.append(date)

    arguments.sortie.parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.sortie, "w", encoding="utf-8") as f:
        json.dump(enregistrements, f, ensure_ascii=False, indent=2)
        f.write("\n")

    ecrire_registre_collecte(arguments.registre_collecte, enregistrements, dates)

    usables = sum(1 for e in enregistrements if e["usable_pour_eval"])
    print(f"{len(enregistrements)} réponses traitées -> {arguments.sortie}")
    print(f"{usables} utilisables pour ML-7 (parcours reconnu)")
    print(f"Registre de collecte -> {arguments.registre_collecte}")


if __name__ == "__main__":
    main()
