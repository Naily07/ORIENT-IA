"""Vectorisation d'un `ProfilCandidat` en entrée numérique du modèle (ML-2).

Encodage multi-hot sur le vocabulaire contrôlé dérivé des archétypes
(`archetypes.VOCAB_*`) pour les listes déclarées, plus une note par matière du
vocabulaire (0 si non déclarée) et un one-hot sur l'environnement recherché.

Le vocabulaire est fermé (issu des archétypes de génération, pas d'un corpus
libre) : un trait déclaré par un vrai candidat mais absent du vocabulaire est
silencieusement ignoré par le vecteur. C'est une limite connue tant que le
modèle n'est entraîné que sur des données synthétiques (voir ML-7) — un
vocabulaire ouvert (embeddings de texte libre, par exemple) est une évolution
naturelle une fois l'enquête réelle disponible.
"""

import numpy as np

from src.ml.archetypes import (
    VOCAB_CENTRES_INTERET,
    VOCAB_COMPETENCES,
    VOCAB_ENVIRONNEMENTS,
    VOCAB_MATIERES,
    VOCAB_PREFERENCES_PRO,
)
from src.schemas import ProfilCandidat


def _multi_hot(valeurs: list[str], vocabulaire: tuple[str, ...]) -> list[float]:
    presents = set(valeurs)
    return [1.0 if v in presents else 0.0 for v in vocabulaire]


def _notes(resultats: dict[str, float], vocabulaire: tuple[str, ...]) -> list[float]:
    # Note ramenée sur [0, 1] : une échelle homogène avec le multi-hot évite
    # qu'une matière notée sur 20 domine artificiellement les autres traits
    # dans les modèles sensibles à l'échelle (régression logistique).
    return [resultats.get(matiere, 0.0) / 20.0 for matiere in vocabulaire]


def _one_hot(valeur: str | None, vocabulaire: tuple[str, ...]) -> list[float]:
    return [1.0 if v == valeur else 0.0 for v in vocabulaire]


def noms_features() -> list[str]:
    """Nom lisible de chaque position du vecteur, dans l'ordre produit par
    `vectoriser()`. Utilisé par `outils.identifier_points_forts()` pour
    traduire une position de vecteur en trait compréhensible."""
    return (
        [f"matiere:{m}" for m in VOCAB_MATIERES]
        + [f"competence:{c}" for c in VOCAB_COMPETENCES]
        + [f"interet:{i}" for i in VOCAB_CENTRES_INTERET]
        + [f"preference_pro:{p}" for p in VOCAB_PREFERENCES_PRO]
        + [f"note:{m}" for m in VOCAB_MATIERES]
        + [f"environnement:{e}" for e in VOCAB_ENVIRONNEMENTS]
    )


def vectoriser(profil: ProfilCandidat) -> np.ndarray:
    """Transforme un profil en vecteur numérique de taille `len(noms_features())`."""
    vecteur = (
        _multi_hot(profil.matieres_preferees, VOCAB_MATIERES)
        + _multi_hot(profil.competences_declarees, VOCAB_COMPETENCES)
        + _multi_hot(profil.centres_interet, VOCAB_CENTRES_INTERET)
        + _multi_hot(profil.preferences_professionnelles, VOCAB_PREFERENCES_PRO)
        + _notes(profil.resultats_scolaires, VOCAB_MATIERES)
        + _one_hot(profil.environnement_travail_recherche, VOCAB_ENVIRONNEMENTS)
    )
    return np.array(vecteur, dtype=float)
