"""Registre de traçabilité des sources du corpus pédagogique (§4 du sujet).

Règle non négociable du sujet : « une information non vérifiée ne devra pas
être présentée comme une information officielle ». Ce registre est ce qui
permet de la respecter — chaque source utilisée pour construire le corpus
(`DocumentSource`, `Mention`, `Parcours`) est enregistrée ici avec son statut,
ce qu'elle a permis d'extraire, et ses limites, et `verifier_provenance()`
contrôle qu'aucune entrée du corpus ne référence une source absente du
registre.
"""

import functools
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from src.config import config

StatutSource = Literal["officiel", "institutionnel", "externe"]


class EntreeRegistreSource(BaseModel):
    id: str
    titre: str
    url: str
    date_consultation: date
    statut: StatutSource
    donnees_extraites: list[str] = Field(default_factory=list)
    limites: list[str] = Field(
        default_factory=list,
        description="Limites ou incertitudes constatées sur cette source (§4 du sujet)",
    )


def charger_registre_sources(
    nom_fichier: str = "registre_sources.json",
) -> list[EntreeRegistreSource]:
    """Charge le registre des sources. Tolère un fichier absent (liste vide)."""
    chemin = config.dossier_data / nom_fichier
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return [EntreeRegistreSource.model_validate(e) for e in json.load(f)]


@functools.lru_cache(maxsize=1)
def _index_registre() -> dict[str, EntreeRegistreSource]:
    """Registre indexé par identifiant, chargé une fois par processus."""
    return {entree.id: entree for entree in charger_registre_sources()}


def statut_de_source(source_id: str | None) -> StatutSource | None:
    """Statut déclaré d'une source du registre, ou `None` si inconnue.

    Permet au reste du pipeline (RAG, agent) de distinguer une information
    officielle d'une information externe **au moment de la citer**. Sans cela,
    la règle non négociable du §4 (« une information non vérifiée ne devra pas
    être présentée comme une information officielle ») ne serait vérifiable
    qu'à la main, en relisant le registre.
    """
    entree = _index_registre().get(source_id) if source_id else None
    return entree.statut if entree else None


def verifier_provenance(
    source_ids: list[str | None], registre: list[EntreeRegistreSource]
) -> list[str]:
    """Retourne les `source_id` référencés (non `None`) qui n'existent pas
    dans le registre.

    Une liste vide signifie que toute donnée du corpus qui prétend avoir une
    source peut effectivement être retracée jusqu'à une entrée du registre —
    c'est le contrôle déterministe qui rend la règle non négociable du §4
    vérifiable plutôt que déclarative.
    """
    connus = {e.id for e in registre}
    return sorted({sid for sid in source_ids if sid is not None and sid not in connus})
