"""Import des réponses de notre enquête vers notre schéma (DATA-7).

    python -m src.enquete_import <chemin_du_csv>

Transforme l'export brut de notre Google Form (`questionnaire.md`,
`generer_google_form.gs`) en `list[ReponseEnquete]`, en traçant la provenance
de chaque champ.

Notre questionnaire demande **tous** les champs du profil : rien n'est
fabriqué ici, chaque valeur est `declaree`, et toute réponse dont l'étiquette
est résolue est directement exploitable pour ML-7.

Deux pièges de l'export Google Forms, tous deux rencontrés en réel
--------------------------------------------------------------------
1. **Intitulés de colonnes dupliqués** entre les deux sections (« Série de
   votre baccalauréat » y figure deux fois). `csv.DictReader` écrase alors
   silencieusement la première occurrence — d'où une lecture strictement par
   position.
2. **Routage** : les premières réponses ont été collectées avec un saut de
   page mal posé, qui faisait enchaîner les étudiants sur la section
   professionnelle. Pour un répondant déclaré étudiant, seule la section
   étudiante est lue — celle vers laquelle il a été légitimement routé. Une
   réponse contredisait d'ailleurs sa propre étiquette d'une section à
   l'autre.
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

from src.config import config
from src.enquete import ReponseEnquete, sauvegarder_reponses
from src.ml.archetypes import PARCOURS_CONNUS
from src.schemas import ProfilCandidat

# Colonnes de l'export, par position. **Pas par nom** : voir le piège 1
# ci-dessus.
COLONNES = {
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


def _normaliser(texte: str) -> str:
    decompose = unicodedata.normalize("NFKD", (texte or "").casefold())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def _resoudre_parcours(brut: str) -> str | None:
    """Extrait un sigle de parcours d'une réponse libre.

    Sans correspondance certaine, retourne `None` plutôt que de deviner : une
    étiquette fausse contaminerait le jeu d'évaluation.
    """
    if not brut:
        return None
    texte = _normaliser(brut)
    for sigle in sorted(PARCOURS_CONNUS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(sigle.lower())}\b", texte):
            return sigle
    return None


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


def _cellule(ligne: list[str], bloc: dict, nom: str) -> str:
    """Valeur d'une colonne repérée par position, ou chaîne vide.

    Une fonction plutôt qu'une fermeture dans la boucle : capturer la ligne
    courante dans une closure est un piège classique que `ruff` signale à
    raison.
    """
    position = bloc.get(nom)
    if position is None or position >= len(ligne):
        return ""
    return ligne[position].strip()


def _entier_ou_none(valeur: str | None) -> int | None:
    try:
        return int((valeur or "").strip())
    except (TypeError, ValueError):
        return None


def importer_csv(chemin: Path) -> list[ReponseEnquete]:
    """Convertit l'export de notre formulaire en réponses typées."""
    with open(chemin, encoding="utf-8", newline="") as f:
        lignes = list(csv.reader(f))

    reponses: list[ReponseEnquete] = []
    for index, ligne in enumerate(lignes[1:], start=1):
        identifiant = f"orientia_{index:04d}"

        colonne_population = COLONNES["population"]
        est_etudiant = "tudiant" in (
            ligne[colonne_population] if colonne_population < len(ligne) else ""
        )
        population = "etudiant" if est_etudiant else "professionnel"
        bloc = COLONNES["etudiant" if est_etudiant else "professionnel"]

        brut_parcours = _cellule(ligne, bloc, "parcours")
        parcours = _sigle_depuis_choix(brut_parcours)

        matieres = _valeurs_multiples(_cellule(ligne, bloc, "matieres"))
        matieres += _valeurs_multiples(_cellule(ligne, bloc, "matieres_libres"))
        competences = _valeurs_multiples(_cellule(ligne, bloc, "competences"))
        interets = _valeurs_multiples(_cellule(ligne, bloc, "interets"))
        environnement = _cellule(ligne, bloc, "environnement") or None
        serie_bac = _cellule(ligne, bloc, "serie_bac") or None

        provenance = {
            nom: "declaree"
            for nom, valeur in (
                ("matieres_preferees", matieres),
                ("competences_declarees", competences),
                ("centres_interet", interets),
                ("environnement_travail_recherche", environnement),
                ("serie_bac", serie_bac),
            )
            if valeur
        }

        profil = ProfilCandidat(
            matieres_preferees=matieres,
            competences_declarees=competences,
            centres_interet=interets,
            environnement_travail_recherche=environnement,
            serie_bac=serie_bac,
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
                adequation_formation_metier=_entier_ou_none(
                    _cellule(ligne, bloc, "adequation")
                ),
                utilisable_pour_evaluation=motif is None,
                motif_exclusion=motif,
            )
        )

    return reponses


def _resume(reponses: list[ReponseEnquete]) -> dict:
    utilisables = [r for r in reponses if r.utilisable_pour_evaluation]
    motifs: dict[str, int] = {}
    populations: dict[str, int] = {}
    for r in reponses:
        if r.motif_exclusion:
            motifs[r.motif_exclusion] = motifs.get(r.motif_exclusion, 0) + 1
        populations[r.population] = populations.get(r.population, 0) + 1
    return {
        "total": len(reponses),
        "utilisables_pour_evaluation": len(utilisables),
        "populations": populations,
        "etiquettes_resolues": sum(1 for r in reponses if r.parcours_declare),
        "motifs_exclusion": motifs,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.enquete_import <csv>")
        raise SystemExit(1)

    reponses = importer_csv(Path(sys.argv[1]))
    chemin = sauvegarder_reponses(
        reponses, config.dossier_data / "enquete" / "reponses_orientia.json"
    )
    print(json.dumps({"reponses_orientia.json": _resume(reponses)}, ensure_ascii=False, indent=2))
    print(f"\nÉcrit dans {chemin}")
    print(
        "\nTous les champs sont DÉCLARÉS : notre questionnaire les demande tous.\n"
        "Aucune fabrication, donc tout enregistrement étiqueté est évaluable."
    )
