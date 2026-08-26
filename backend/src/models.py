"""Modèle de document du corpus pédagogique, utilisé par le RAG.

`DocumentSource` reste volontairement générique (id/titre/catégorie/contenu) :
il correspond à un article du corpus ISPM (présentation d'un parcours, fiche
de compétences, procédure d'admission...). Le modèle de données métier complet
(Formation, Parcours, Matière, Compétence, Prérequis, Métier...) et le
registre de traçabilité des sources (URL, date de consultation, statut
officiel/institutionnel/externe) viendront s'y adosser dans un ticket dédié
une fois le corpus effectivement collecté (voir `BACKLOG.md`, tickets
DATA-1 à DATA-3) — `source_id` est prévu pour ce lien.
"""

import json
from datetime import datetime

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


def charger_corpus(nom_fichier: str = "corpus.json", chemin=None) -> list[DocumentSource]:
    """Charge le corpus pédagogique depuis un fichier JSON.

    Tolère un fichier absent (retourne une liste vide) : le corpus n'est pas
    encore livré à ce stade du projet, mais l'ingestion RAG doit pouvoir être
    appelée sans planter le démarrage de l'API.
    """
    chemin = chemin or (config.dossier_data / nom_fichier)
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return [DocumentSource.model_validate(d) for d in json.load(f)]
