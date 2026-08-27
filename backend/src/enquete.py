"""Réponses d'enquête réelles et leur provenance (DATA-5, DATA-7).

Ce module porte le contrat de données de l'enquête terrain. Sa raison d'être
tient en une phrase : **distinguer, champ par champ, ce qui a été réellement
déclaré de ce qui a été dérivé ou fabriqué.**

Trois provenances, et la distinction n'est pas cosmétique :

- `declaree` — le répondant l'a écrit. Seule provenance utilisable pour
  mesurer quoi que ce soit.
- `derivee` — calculé sans rien inventer à partir d'une réponse réelle
  (échelle 1–5 convertie en note /20, sigle de parcours extrait d'un texte
  libre du type « Igglia 2014-2019 »).
- `generee` — **fabriqué**. Plausible, cohérent avec le parcours déclaré, mais
  jamais collecté.

Pourquoi cette séparation est structurante pour ML-7
----------------------------------------------------
ML-7 mesure la capacité d'un modèle entraîné sur profils synthétiques à
généraliser à de vrais profils. Or les champs générés ici le sont à partir de
`src.ml.archetypes` — c'est-à-dire du **générateur qui a produit le jeu
d'entraînement**. Les utiliser pour évaluer reviendrait à demander au modèle
de retrouver ses propres hypothèses sur des données qu'on lui a soufflées :
le score serait excellent et ne voudrait rien dire.

D'où `utilisable_pour_evaluation`, qui n'est vrai que si l'étiquette est
fiable **et** qu'aucun trait exploité par le modèle n'a été fabriqué. Les
enregistrements complétés servent à la démonstration et au frontend, jamais à
la mesure.

`serie_bac` fait exception et reste utilisable : elle est générée à partir des
**prérequis officiels d'admission** (`backend/data/prerequis.json`), pas des
archétypes ML — et le modèle ne la voit jamais, elle n'appartient pas à son
espace de features. Elle n'alimente que la règle d'admissibilité
(`src.admission`), qui est déterministe.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.config import config
from src.schemas import ProfilCandidat

Provenance = Literal["declaree", "derivee", "generee"]

Population = Literal["etudiant", "professionnel"]

# Champs du profil qui entrent dans l'espace de features du modèle. Si l'un
# d'eux est généré, l'enregistrement sort du jeu d'évaluation.
CHAMPS_EXPLOITES_PAR_LE_MODELE: tuple[str, ...] = (
    "matieres_preferees",
    "resultats_scolaires",
    "competences_declarees",
    "centres_interet",
    "preferences_professionnelles",
    "environnement_travail_recherche",
)


class ReponseEnquete(BaseModel):
    """Une réponse d'enquête, avec la provenance de chacun de ses champs."""

    id: str
    population: Population
    # Étiquette : le parcours effectivement suivi, résolu vers un des 16 sigles.
    # `None` quand la réponse libre n'a pas pu être rattachée avec certitude.
    parcours_declare: str | None = None
    parcours_brut: str | None = Field(
        default=None, description="Réponse libre d'origine, conservée pour audit"
    )
    profil: ProfilCandidat
    provenance: dict[str, Provenance] = Field(default_factory=dict)

    satisfaction: int | None = Field(
        default=None, description="Échelle 1–5 telle que déclarée, jamais convertie"
    )
    metier_exerce: str | None = None
    adequation_formation_metier: int | None = Field(
        default=None, description="Échelle 1–5, professionnels uniquement"
    )

    utilisable_pour_evaluation: bool = False
    motif_exclusion: str | None = None

    @property
    def champs_generes(self) -> list[str]:
        return sorted(c for c, p in self.provenance.items() if p == "generee")


class RegistreCollecte(BaseModel):
    """Registre de collecte exigé au §5 du sujet (DATA-5).

    Les champs sont ceux que le sujet énumère explicitement. `limites` est
    obligatoire à l'usage : le sujet demande de nommer les biais plutôt que de
    les masquer.
    """

    source: str
    populations_visees: list[str] = Field(default_factory=list)
    mode_diffusion: str | None = None
    periode_collecte: str | None = None
    reponses_recues: int = 0
    reponses_retenues: int = 0
    reponses_ecartees: int = 0
    repartition_populations: dict[str, int] = Field(default_factory=dict)
    texte_consentement: str | None = None
    procedure_anonymisation: str | None = None
    traitements_posterieurs: list[str] = Field(default_factory=list)
    limites: list[str] = Field(default_factory=list)


def charger_reponses(nom_fichier: str = "reponses_anonymisees.json") -> list[ReponseEnquete]:
    """Charge les réponses d'enquête. Tolère un fichier absent (liste vide)."""
    chemin = config.dossier_data / "enquete" / nom_fichier
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return [ReponseEnquete.model_validate(r) for r in json.load(f)]


def charger_registre_collecte(
    nom_fichier: str = "registre_collecte.json",
) -> RegistreCollecte | None:
    chemin = config.dossier_data / "enquete" / nom_fichier
    if not chemin.exists():
        return None
    with open(chemin, encoding="utf-8") as f:
        return RegistreCollecte.model_validate(json.load(f))


def jeu_evaluation(reponses: list[ReponseEnquete] | None = None) -> list[ReponseEnquete]:
    """Le sous-ensemble mesurable : étiquette fiable et aucun trait fabriqué.

    C'est **la seule** entrée admissible pour ML-7. Passer par cette fonction
    plutôt que de filtrer à la main évite qu'un appelant oublie la condition.
    """
    reponses = reponses if reponses is not None else charger_reponses()
    return [r for r in reponses if r.utilisable_pour_evaluation]


def sauvegarder_reponses(
    reponses: list[ReponseEnquete], chemin: Path | None = None
) -> Path:
    chemin = chemin or (config.dossier_data / "enquete" / "reponses_anonymisees.json")
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in reponses], f, ensure_ascii=False, indent=2
        )
    return chemin
