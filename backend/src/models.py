"""Modèles de données du corpus pédagogique ISPM.

Deux couches, complémentaires plutôt que redondantes :

- `DocumentSource` : un article en texte libre (présentation d'un parcours,
  fiche de compétences, procédure d'admission...), utilisé par le RAG pour la
  recherche documentaire et la génération de réponses citées.
- `Mention`/`Parcours`/`Matiere`/`Competence`/`Prerequis`/`Metier` : les
  mêmes informations, mais structurées — utilisées par le modèle de Machine
  Learning (Phase 2 du sujet) et par l'ontologie (§12 du sujet, voir
  `ontologie.py`), qui ont besoin de relations explicites plutôt que de texte
  à interpréter.

Un même corpus source (site ISPM, brochures) alimente typiquement les deux :
le contenu brut devient des `DocumentSource` pour le RAG, et les faits qu'on
en extrait (quel parcours enseigne quelle matière, prépare à quel métier)
peuplent les modèles structurés ci-dessous. Le champ `source_id`, porté par
`DocumentSource`, `Mention` et `Parcours`, référence une entrée du registre
de traçabilité des sources (`src.sources.EntreeRegistreSource`, DATA-2) — voir
`src.sources.verifier_provenance()` pour le contrôle qui s'assure qu'aucune
entrée ne pointe vers une source inexistante.

Tous les chargeurs tolèrent un fichier absent (liste vide) : le corpus n'est
pas encore collecté à ce stade du projet, mais l'API doit pouvoir démarrer
sans lui.
"""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.config import config


class DocumentSource(BaseModel):
    id: str  # ex. "FORM-INFO-01"
    titre: str
    categorie: str
    contenu: str
    derniere_maj: datetime
    # Identifiant vers une entrée du futur registre des sources (DATA-2).
    # Optionnel pour l'instant : un corpus de démarrage peut ne pas encore
    # avoir de registre associé.
    source_id: str | None = None


def charger_corpus(
    nom_fichier: str = "corpus.json", chemin: Path | None = None
) -> list[DocumentSource]:
    """Charge le corpus documentaire (RAG) depuis un fichier JSON."""
    chemin = chemin or (config.dossier_data / nom_fichier)
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return [DocumentSource.model_validate(d) for d in json.load(f)]


# --- Modèles structurés (§3 du sujet, ML et ontologie) -----------------------


class Mention(BaseModel):
    id: str
    nom: str
    niveau: str  # ex. "Licence", "Master", ou une description si plusieurs niveaux s'enchaînent
    diplome: str | None = None
    # Identifiant vers une entrée du registre des sources (DATA-2).
    source_id: str | None = None


class Matiere(BaseModel):
    id: str
    nom: str
    source_id: str | None = None


class Competence(BaseModel):
    id: str
    nom: str
    source_id: str | None = None


class Prerequis(BaseModel):
    id: str
    description: str
    source_id: str | None = None


class Metier(BaseModel):
    id: str
    nom: str
    secteur: str | None = None
    source_id: str | None = None


class Parcours(BaseModel):
    id: str
    nom: str
    mention_id: str
    # Listes d'identifiants (Matiere.id, Competence.id...), pas d'objets
    # imbriqués : ça reste facile à sérialiser en JSON et c'est directement ce
    # dont l'ontologie (ontologie.py) a besoin pour peupler ses relations.
    matieres: list[str] = []
    competences: list[str] = []
    prerequis: list[str] = []
    debouches: list[str] = []  # identifiants de Metier
    # Autres parcours accessibles en passerelle (§3 du sujet).
    passerelles: list[str] = []
    # Identifiant vers une entrée du registre des sources (DATA-2).
    source_id: str | None = None


class CorpusFormations(BaseModel):
    """Toutes les ressources structurées du corpus, chargées en mémoire."""

    mentions: list[Mention] = []
    parcours: list[Parcours] = []
    matieres: list[Matiere] = []
    competences: list[Competence] = []
    prerequis: list[Prerequis] = []
    metiers: list[Metier] = []


def _charger_json(nom_fichier: str) -> list[dict]:
    chemin = config.dossier_data / nom_fichier
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def charger_corpus_formations(
    fichier_mentions: str = "mentions.json",
    fichier_parcours: str = "parcours.json",
    fichier_matieres: str = "matieres.json",
    fichier_competences: str = "competences.json",
    fichier_prerequis: str = "prerequis.json",
    fichier_metiers: str = "metiers.json",
) -> CorpusFormations:
    """Point d'entrée unique pour charger le corpus structuré au démarrage de
    l'API. Noms de fichiers à ajuster une fois le corpus réel collecté
    (DATA-1)."""
    return CorpusFormations(
        mentions=[Mention.model_validate(m) for m in _charger_json(fichier_mentions)],
        parcours=[Parcours.model_validate(p) for p in _charger_json(fichier_parcours)],
        matieres=[Matiere.model_validate(m) for m in _charger_json(fichier_matieres)],
        competences=[Competence.model_validate(c) for c in _charger_json(fichier_competences)],
        prerequis=[Prerequis.model_validate(p) for p in _charger_json(fichier_prerequis)],
        metiers=[Metier.model_validate(m) for m in _charger_json(fichier_metiers)],
    )
