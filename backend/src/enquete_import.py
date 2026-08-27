"""Import des réponses d'enquête réelles vers notre schéma (DATA-7).

    python -m src.enquete_import <csv>              # enquête tierce
    python -m src.enquete_import <csv> --orientia   # notre formulaire

Deux collectes, deux importeurs, et surtout **deux fichiers de sortie qui ne
sont jamais fusionnés** : les moyenner produirait un chiffre unique
impossible à interpréter, puisque les questionnaires ne demandaient pas les
mêmes choses.

| | Notre enquête | Enquête tierce |
|---|---|---|
| Réponses exploitables | 14 | 68 |
| Traits par profil (médiane) | **7** | 1 |
| Profils réellement exploitables par le modèle | **14/14** | 23/68 |
| Série de bac déclarée | **14/14** | 0/68 |
| Parcours représentés | 5/16 | **14/16** |

Les deux sont complémentaires et le resteront : la nôtre a la profondeur, la
leur la couverture. Aucune ne remplace l'autre.

Transforme un export brut en `list[ReponseEnquete]`, en traçant la provenance
de chaque champ. Reproductible : relancé sur le même CSV, il produit le même
fichier (le générateur pseudo-aléatoire est amorcé sur l'identifiant de la
réponse, pas sur l'horloge).

Origine de l'enquête tierce
---------------------------
Enquête menée par une équipe tierce sur le même sujet ISPM, publiée sous
licence MIT dans `source externe`. 86 réponses, 71 étudiants et
15 professionnels — exactement les deux populations que le §5 du sujet exige,
et la sur-représentation étudiante que ce même paragraphe annonçait.

**Ce n'est pas notre collecte.** Le questionnaire diffusé n'est pas le nôtre
(`backend/data/enquete/questionnaire.md`), et il ne demandait ni série de
baccalauréat, ni compétences, ni centres d'intérêt, ni environnement de
travail. Ce fichier importe ce qui existe et **fabrique explicitement** le
reste — voir `src.enquete` pour ce que cette distinction interdit.

Ce que le questionnaire tiers a réellement collecté
---------------------------------------------------
| Champ                 | Étudiants | Professionnels |
|-----------------------|-----------|----------------|
| parcours (étiquette)  | oui       | oui, en texte libre à parser |
| matières préférées    | oui       | non            |
| résultats (échelle 1–5)| oui      | non            |
| satisfaction          | oui       | non            |
| métier exercé         | non       | oui            |

Les professionnels n'ont donc **aucun trait de profil** exploitable : leur
seule contribution mesurable est l'étiquette et le métier. C'est une limite
du questionnaire tiers, pas de l'import.
"""

import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

from src.config import config
from src.enquete import (
    CHAMPS_EXPLOITES_PAR_LE_MODELE,
    ReponseEnquete,
    sauvegarder_reponses,
)
from src.ml.archetypes import ARCHETYPES, PARCOURS_CONNUS
from src.models import charger_corpus_formations
from src.schemas import ProfilCandidat

# Conversion de l'échelle 1–5 du questionnaire vers une note /20. Reprise de
# la table de l'équipe tierce (`map_survey_to_features.py`) pour rester
# comparable à leurs propres résultats. C'est une **dérivation**, pas une
# mesure : le répondant n'a jamais donné de note sur 20.
ECHELLE_VERS_NOTE20 = {1: 6, 2: 10, 3: 13, 4: 16, 5: 19}

# Séries de baccalauréat, avec un poids reflétant leur fréquence relative
# parmi les bacheliers malgaches candidats au supérieur scientifique. Poids
# grossiers et assumés comme tels : ils évitent seulement de tirer une série
# rare aussi souvent qu'une série courante.
POIDS_SERIES = {"C": 3, "D": 5, "S": 2, "A": 2, "A2": 1}


def _normaliser(texte: str) -> str:
    decompose = unicodedata.normalize("NFKD", (texte or "").casefold())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def _resoudre_parcours(brut: str) -> str | None:
    """Extrait un sigle de parcours d'une réponse libre.

    Les professionnels écrivent « Igglia 2014-2019 », « ESIIA - 2017 »,
    « IGGLIA + 2023 ». On cherche un sigle connu en frontière de mot ; sans
    correspondance certaine, on retourne `None` plutôt que de deviner — une
    étiquette fausse contaminerait le jeu d'évaluation.
    """
    if not brut:
        return None
    texte = _normaliser(brut)
    for sigle in sorted(PARCOURS_CONNUS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(sigle.lower())}\b", texte):
            return sigle
    return None


def _series_admissibles(
    parcours_id: str, prerequis_par_parcours: dict[str, list[str]]
) -> list[str]:
    """Séries de bac compatibles avec les prérequis officiels du parcours.

    La génération de `serie_bac` s'appuie sur `backend/data/prerequis.json`
    (collecté sur le site officiel de l'ISPM), pas sur les archétypes ML :
    c'est ce qui la rend utilisable sans contaminer ML-7.
    """
    descriptions = prerequis_par_parcours.get(parcours_id) or []
    texte = " ".join(descriptions).lower()
    if not texte:
        return list(POIDS_SERIES)
    if "toute serie" in _normaliser(texte):
        return list(POIDS_SERIES)

    admissibles = [
        serie for serie in POIDS_SERIES
        if re.search(rf"\b{serie.lower()}\b", _normaliser(texte))
    ]
    return admissibles or list(POIDS_SERIES)


def _tirer_serie_bac(parcours_id: str, prerequis: dict[str, list[str]], rng: random.Random) -> str:
    candidates = _series_admissibles(parcours_id, prerequis)
    poids = [POIDS_SERIES.get(s, 1) for s in candidates]
    return rng.choices(candidates, weights=poids, k=1)[0]


def _tirer_sous_ensemble(valeurs: list[str], rng: random.Random) -> list[str]:
    """Une minorité des traits de l'archétype — jamais la totalité.

    Même principe que `donnees_synthetiques` : un profil qui reprendrait tout
    l'archétype serait trivialement identifiable.
    """
    if not valeurs:
        return []
    k = max(1, round(len(valeurs) * rng.uniform(0.3, 0.6)))
    return rng.sample(valeurs, k)


def _completer_profil(
    profil: ProfilCandidat,
    parcours_id: str | None,
    prerequis: dict[str, list[str]],
    provenance: dict[str, str],
    rng: random.Random,
) -> ProfilCandidat:
    """Fabrique les champs que le questionnaire tiers n'a jamais demandés.

    Chaque champ écrit ici est marqué `generee`. `serie_bac` vient des
    prérequis officiels ; les autres viennent des archétypes ML, ce qui les
    rend impropres à l'évaluation (voir `src.enquete`).
    """
    if parcours_id is None:
        return profil

    donnees = profil.model_dump()

    if not donnees.get("serie_bac"):
        donnees["serie_bac"] = _tirer_serie_bac(parcours_id, prerequis, rng)
        provenance["serie_bac"] = "generee"

    archetype = ARCHETYPES.get(parcours_id)
    if archetype is None:
        return ProfilCandidat(**donnees)

    correspondances = (
        ("competences_declarees", "competences"),
        ("centres_interet", "centres_interet"),
        ("preferences_professionnelles", "preferences_professionnelles"),
    )
    for champ_profil, champ_archetype in correspondances:
        if not donnees.get(champ_profil):
            donnees[champ_profil] = _tirer_sous_ensemble(archetype[champ_archetype], rng)
            provenance[champ_profil] = "generee"

    if not donnees.get("environnement_travail_recherche"):
        donnees["environnement_travail_recherche"] = archetype["environnement"]
        provenance["environnement_travail_recherche"] = "generee"

    return ProfilCandidat(**donnees)


def _entier_ou_none(valeur: str | None) -> int | None:
    try:
        return int((valeur or "").strip())
    except (TypeError, ValueError):
        return None


def _liste_depuis_texte(texte: str | None) -> list[str]:
    return [t.strip() for t in (texte or "").split(",") if t.strip()]


def importer_csv(chemin: Path, completer: bool = True) -> list[ReponseEnquete]:
    """Convertit l'export brut du formulaire en réponses typées.

    `completer=False` produit uniquement les champs réellement déclarés —
    utile pour vérifier ce que l'enquête a effectivement collecté, sans
    aucune fabrication.
    """
    corpus = charger_corpus_formations()
    descriptions = {p.id: p.description for p in corpus.prerequis}
    prerequis = {
        p.id: [descriptions[i] for i in p.prerequis if i in descriptions]
        for p in corpus.parcours
    }

    with open(chemin, encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))

    colonnes = list(lignes[0].keys()) if lignes else []
    # Le formulaire tiers n'a pas d'identifiants de colonnes stables : on se
    # repère sur leur position, documentée dans l'en-tête du CSV.
    # La colonne 6 (« qu'est-ce qui vous a motivé ») est volontairement
    # ignorée : c'est du texte libre narratif, dont extraire des centres
    # d'intérêt exigerait une interprétation — donc une invention déguisée en
    # donnée déclarée.
    (
        COL_POPULATION, COL_PARCOURS_ETU, COL_MATIERES, COL_RESULTATS,
        COL_SATISFACTION, COL_PARCOURS_PRO, COL_METIER, COL_ADEQUATION,
    ) = (2, 3, 4, 5, 7, 8, 9, 10)

    reponses: list[ReponseEnquete] = []
    for index, ligne in enumerate(lignes, start=1):
        identifiant = f"reponse_{index:04d}"
        # Amorce déterministe : relancer l'import ne change pas les valeurs
        # fabriquées, ce qui rend le fichier produit diffable.
        rng = random.Random(identifiant)

        est_etudiant = "tudiant" in (ligne.get(colonnes[COL_POPULATION]) or "")
        population = "etudiant" if est_etudiant else "professionnel"

        brut_parcours = (
            ligne.get(colonnes[COL_PARCOURS_ETU]) if est_etudiant
            else ligne.get(colonnes[COL_PARCOURS_PRO])
        )
        parcours = _resoudre_parcours(brut_parcours or "")

        provenance: dict[str, str] = {}
        donnees_profil: dict = {}

        matieres = _liste_depuis_texte(ligne.get(colonnes[COL_MATIERES]))
        if matieres:
            donnees_profil["matieres_preferees"] = matieres
            provenance["matieres_preferees"] = "declaree"

        echelle = _entier_ou_none(ligne.get(colonnes[COL_RESULTATS]))
        if echelle is not None and matieres:
            note = ECHELLE_VERS_NOTE20.get(echelle)
            if note is not None:
                # Le répondant a noté « maths/info » globalement, pas matière
                # par matière : la note est reportée sur ses matières déclarées.
                donnees_profil["resultats_scolaires"] = {m: float(note) for m in matieres}
                provenance["resultats_scolaires"] = "derivee"

        profil = ProfilCandidat(**donnees_profil)
        if completer:
            profil = _completer_profil(profil, parcours, prerequis, provenance, rng)

        genere_dans_les_features = any(
            provenance.get(champ) == "generee" for champ in CHAMPS_EXPLOITES_PAR_LE_MODELE
        )
        if parcours is None:
            motif = "parcours non résolu depuis la réponse libre"
        elif not matieres:
            motif = "aucun trait de profil réellement déclaré"
        elif genere_dans_les_features:
            motif = "traits fabriqués à partir des archétypes ML — évaluation contaminée"
        else:
            motif = None

        reponses.append(
            ReponseEnquete(
                id=identifiant,
                population=population,
                parcours_declare=parcours,
                parcours_brut=(brut_parcours or "").strip() or None,
                profil=profil,
                provenance=provenance,
                satisfaction=_entier_ou_none(ligne.get(colonnes[COL_SATISFACTION])),
                metier_exerce=(ligne.get(colonnes[COL_METIER]) or "").strip() or None,
                adequation_formation_metier=_entier_ou_none(
                    ligne.get(colonnes[COL_ADEQUATION])
                ),
                utilisable_pour_evaluation=motif is None,
                motif_exclusion=motif,
            )
        )

    return reponses


# --- Import de NOTRE formulaire (DATA-4) --------------------------------------
# Colonnes de l'export Google Forms, par position. **Pas par nom** : l'export
# duplique les intitulés entre les deux sections (« Série de votre
# baccalauréat » apparaît deux fois), et `csv.DictReader` écrase alors
# silencieusement la première occurrence.

COLONNES_ORIENTIA = {
    "population": 2,
    "etudiant": {
        "serie_bac": 3, "parcours": 4, "matieres": 5, "matieres_libres": 6,
        "competences": 7, "interets": 8, "environnement": 9,
        "satisfaction": 10, "referait": 11, "alternative": 12,
    },
    "professionnel": {
        "serie_bac": 13, "parcours": 14, "metier": 15, "matieres": 16,
        "matieres_libres": 17, "competences": 18, "interets": 19,
        "environnement": 20, "adequation": 21, "alternative": 22,
    },
}

HORS_ISPM = "hors ispm"
AUCUNE_COMPETENCE = "aucune en particulier"


def _sigle_depuis_choix(valeur: str) -> str | None:
    """Extrait le sigle d'un choix « IGGLIA — Informatique de Gestion… ».

    Retourne `None` pour « Une formation hors ISPM » : la réponse est réelle
    mais sort du périmètre des 16 parcours que le modèle connaît.
    """
    texte = (valeur or "").strip()
    if not texte or HORS_ISPM in _normaliser(texte):
        return None
    sigle = texte.split("—")[0].strip().split()[0] if texte else ""
    return sigle if sigle in PARCOURS_CONNUS else _resoudre_parcours(texte)


def _cellule(ligne: list[str], bloc: dict, nom: str) -> str:
    """Valeur d'une colonne repérée par position, ou chaîne vide.

    Une fonction plutôt qu'une fermeture dans la boucle : capturer la ligne
    courante dans une closure est un piège classique (la fermeture voit la
    dernière valeur si son appel est différé) que `ruff` signale à raison.
    """
    position = bloc.get(nom)
    if position is None or position >= len(ligne):
        return ""
    return ligne[position].strip()


def _valeurs_multiples(brut: str) -> list[str]:
    """Cases à cocher Google Forms : valeurs séparées par des virgules.

    « Aucune en particulier » est une absence déclarée, pas une compétence :
    la conserver ferait compter un trait qui n'en est pas un dans le calcul
    d'exploitabilité (`features.CouvertureProfil`).
    """
    return [
        v.strip() for v in (brut or "").split(",")
        if v.strip() and _normaliser(v).strip() != AUCUNE_COMPETENCE
    ]


def importer_csv_orientia(chemin: Path) -> list[ReponseEnquete]:
    """Importe l'export de **notre** formulaire (`questionnaire.md`).

    Différence essentielle avec l'enquête tierce : notre questionnaire demande
    tous les champs du profil. **Rien n'est fabriqué ici** — chaque valeur est
    `declaree`, et l'ensemble des réponses étiquetées est directement
    exploitable pour ML-7.

    Anomalie de routage traitée : les 15 premières réponses ont été collectées
    avec un saut de page mal posé (voir `generer_google_form.gs`), qui faisait
    enchaîner les étudiants sur la section professionnelle. Pour un répondant
    déclaré étudiant, seule la section étudiante est lue — c'est celle vers
    laquelle il a été légitimement routé, et la seule dont les questions
    correspondent à sa situation. Une réponse contredisait d'ailleurs sa
    propre étiquette d'une section à l'autre.
    """
    with open(chemin, encoding="utf-8", newline="") as f:
        lignes = list(csv.reader(f))

    reponses: list[ReponseEnquete] = []
    for index, ligne in enumerate(lignes[1:], start=1):
        identifiant = f"orientia_{index:04d}"

        est_etudiant = "tudiant" in (
            ligne[COLONNES_ORIENTIA["population"]]
            if COLONNES_ORIENTIA["population"] < len(ligne) else ""
        )
        population = "etudiant" if est_etudiant else "professionnel"
        bloc = COLONNES_ORIENTIA["etudiant" if est_etudiant else "professionnel"]

        brut_parcours = _cellule(ligne, bloc, "parcours")
        parcours = _sigle_depuis_choix(brut_parcours)

        matieres = _valeurs_multiples(_cellule(ligne, bloc, "matieres"))
        matieres += _valeurs_multiples(_cellule(ligne, bloc, "matieres_libres"))
        competences = _valeurs_multiples(_cellule(ligne, bloc, "competences"))
        interets = _valeurs_multiples(_cellule(ligne, bloc, "interets"))
        environnement = _cellule(ligne, bloc, "environnement") or None

        provenance = {
            nom: "declaree"
            for nom, valeur in (
                ("matieres_preferees", matieres),
                ("competences_declarees", competences),
                ("centres_interet", interets),
                ("environnement_travail_recherche", environnement),
                ("serie_bac", _cellule(ligne, bloc, "serie_bac")),
            )
            if valeur
        }

        profil = ProfilCandidat(
            matieres_preferees=matieres,
            competences_declarees=competences,
            centres_interet=interets,
            environnement_travail_recherche=environnement,
            serie_bac=_cellule(ligne, bloc, "serie_bac") or None,
        )

        traits = len(matieres) + len(competences) + len(interets) + (1 if environnement else 0)
        if parcours is None:
            motif = (
                "formation hors ISPM" if HORS_ISPM in _normaliser(brut_parcours)
                else "parcours non résolu"
            )
        elif traits == 0:
            motif = "aucun trait de profil déclaré"
        else:
            motif = None

        reponses.append(
            ReponseEnquete(
                id=identifiant,
                population=population,
                parcours_declare=parcours,
                parcours_brut=brut_parcours or None,
                profil=profil,
                provenance=provenance,
                satisfaction=_entier_ou_none(_cellule(ligne, bloc, "satisfaction")),
                metier_exerce=_cellule(ligne, bloc, "metier") or None,
                adequation_formation_metier=_entier_ou_none(_cellule(ligne, bloc, "adequation")),
                utilisable_pour_evaluation=motif is None,
                motif_exclusion=motif,
            )
        )

    return reponses


def _resume(reponses: list[ReponseEnquete]) -> dict:
    utilisables = [r for r in reponses if r.utilisable_pour_evaluation]
    motifs: dict[str, int] = {}
    for r in reponses:
        if r.motif_exclusion:
            motifs[r.motif_exclusion] = motifs.get(r.motif_exclusion, 0) + 1
    populations: dict[str, int] = {}
    for r in reponses:
        populations[r.population] = populations.get(r.population, 0) + 1
    return {
        "total": len(reponses),
        "utilisables_pour_evaluation": len(utilisables),
        "populations": populations,
        "etiquettes_resolues": sum(1 for r in reponses if r.parcours_declare),
        "motifs_exclusion": motifs,
    }


def _importer_notre_formulaire(source: Path) -> None:
    reponses = importer_csv_orientia(source)
    chemin = sauvegarder_reponses(
        reponses, config.dossier_data / "enquete" / "reponses_orientia.json"
    )
    print(json.dumps({"reponses_orientia.json": _resume(reponses)}, ensure_ascii=False, indent=2))
    print(f"\nÉcrit dans {chemin}")
    print(
        "\nTous les champs sont DÉCLARÉS : notre questionnaire les demande tous.\n"
        "Aucune fabrication, donc tout enregistrement étiqueté est évaluable."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.enquete_import <csv> [--orientia]")
        raise SystemExit(1)

    source = Path(sys.argv[1])

    # Notre propre export a une structure différente de l'enquête tierce :
    # deux sections, intitulés dupliqués, et tous les champs collectés.
    if "--orientia" in sys.argv:
        _importer_notre_formulaire(source)
        raise SystemExit(0)

    # DEUX fichiers, et la séparation est le cœur du sujet : compléter un
    # enregistrement le rend inévaluable, puisque les traits fabriqués
    # viennent du générateur qui a produit le jeu d'entraînement. Livrer un
    # seul fichier « enrichi » ferait disparaître le jeu de test.
    reelles = importer_csv(source, completer=False)
    completees = importer_csv(source, completer=True)

    dossier = config.dossier_data / "enquete"
    chemin_eval = sauvegarder_reponses(reelles, dossier / "reponses_anonymisees.json")
    chemin_demo = sauvegarder_reponses(completees, dossier / "profils_completes.json")

    print(json.dumps(
        {
            "reponses_anonymisees.json (jeu d'évaluation, ML-7)": _resume(reelles),
            "profils_completes.json (démonstration uniquement)": _resume(completees),
        },
        ensure_ascii=False, indent=2,
    ))
    print(f"\nJeu d'évaluation   : {chemin_eval}")
    print(f"Profils complétés  : {chemin_demo}")
    print(
        "\nRappel : `profils_completes.json` contient des champs FABRIQUÉS "
        "(voir `provenance`).\nIl ne doit jamais servir à mesurer quoi que ce soit."
    )
