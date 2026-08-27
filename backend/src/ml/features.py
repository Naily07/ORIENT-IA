"""Vectorisation d'un `ProfilCandidat` en entrée numérique du modèle (ML-2).

Encodage multi-hot sur le vocabulaire contrôlé dérivé des archétypes
(`archetypes.VOCAB_*`) pour les listes déclarées, plus une note par matière du
vocabulaire (0 si non déclarée) et un one-hot sur l'environnement recherché.

Le vocabulaire reste fermé côté espace de features (le modèle est entraîné
dessus), mais l'entrée est désormais **ouverte** : `src.ml.vocabulaire.resoudre()`
ramène un terme déclaré librement (« maths », « Python », « SVT ») vers son
équivalent du vocabulaire, et remonte ce qu'il n'a pas su reconnaître au lieu de
l'ignorer en silence. `analyser_couverture()` expose ce diagnostic pour que
l'assistant puisse dire ce qu'il n'a pas compris — et refuser de produire un score
sur un profil qu'il n'a en réalité pas exploité.
"""

from dataclasses import dataclass, field

import numpy as np

from src.ml.archetypes import (
    VOCAB_CENTRES_INTERET,
    VOCAB_COMPETENCES,
    VOCAB_ENVIRONNEMENTS,
    VOCAB_MATIERES,
    VOCAB_PREFERENCES_PRO,
)
from src.ml.vocabulaire import correspondances, resoudre
from src.schemas import ProfilCandidat


@dataclass
class CouvertureProfil:
    """Ce que la vectorisation a réellement retenu d'un profil déclaré."""

    matieres: list[str] = field(default_factory=list)
    competences: list[str] = field(default_factory=list)
    centres_interet: list[str] = field(default_factory=list)
    preferences_professionnelles: list[str] = field(default_factory=list)
    environnement: str | None = None
    non_reconnus: list[str] = field(default_factory=list)

    @property
    def nb_traits_reconnus(self) -> int:
        return (
            len(self.matieres)
            + len(self.competences)
            + len(self.centres_interet)
            + len(self.preferences_professionnelles)
            + (1 if self.environnement else 0)
        )

    @property
    def exploitable(self) -> bool:
        """Vrai si le profil porte assez de signal pour que le modèle ait un sens.

        Un seul trait reconnu sur 156 dimensions ne permet pas de distinguer 16
        parcours : le modèle retomberait sur des scores proches de la distribution
        a priori, présentés à l'utilisateur comme s'ils étaient informatifs.
        """
        return self.nb_traits_reconnus >= 2


def _multi_hot(valeurs: list[str], vocabulaire: tuple[str, ...]) -> list[float]:
    presents = set(valeurs)
    return [1.0 if v in presents else 0.0 for v in vocabulaire]


def _notes(resultats: dict[str, float], vocabulaire: tuple[str, ...]) -> list[float]:
    """Notes ramenées sur [0, 1], sur les matières effectivement reconnues.

    Les clés de `resultats_scolaires` passent par la même résolution que les
    matières préférées : un candidat qui écrit « maths: 16 » doit voir sa note
    prise en compte comme `mathematiques`. La résolution se fait en **un seul
    lot** — une résolution par clé déclenchait autant de passages dans le modèle
    d'embedding qu'il y avait de matières hors vocabulaire.
    """
    if not resultats:
        return [0.0] * len(vocabulaire)

    trouvees, _ = correspondances(list(resultats), vocabulaire)

    notes_par_matiere: dict[str, float] = {}
    for matiere_declaree, note in resultats.items():
        reconnue = trouvees.get(matiere_declaree)
        if reconnue is None:
            continue
        # Si deux libellés retombent sur la même matière, on garde la meilleure
        # note plutôt que la dernière rencontrée (ordre de dict arbitraire).
        notes_par_matiere[reconnue] = max(notes_par_matiere.get(reconnue, 0.0), note)

    # Une échelle homogène avec le multi-hot évite qu'une note sur 20 domine
    # artificiellement les autres traits dans les modèles sensibles à l'échelle.
    return [min(notes_par_matiere.get(m, 0.0) / 20.0, 1.0) for m in vocabulaire]


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


def analyser_couverture(profil: ProfilCandidat) -> CouvertureProfil:
    """Résout chaque champ déclaré vers le vocabulaire et rapporte les écarts."""
    matieres, non_matieres = resoudre(profil.matieres_preferees, VOCAB_MATIERES)
    competences, non_competences = resoudre(profil.competences_declarees, VOCAB_COMPETENCES)
    interets, non_interets = resoudre(profil.centres_interet, VOCAB_CENTRES_INTERET)
    preferences, non_preferences = resoudre(
        profil.preferences_professionnelles, VOCAB_PREFERENCES_PRO
    )

    environnement = None
    non_environnement: list[str] = []
    if profil.environnement_travail_recherche:
        environnements, non_environnement = resoudre(
            [profil.environnement_travail_recherche], VOCAB_ENVIRONNEMENTS
        )
        environnement = environnements[0] if environnements else None

    return CouvertureProfil(
        matieres=matieres,
        competences=competences,
        centres_interet=interets,
        preferences_professionnelles=preferences,
        environnement=environnement,
        non_reconnus=(
            non_matieres + non_competences + non_interets + non_preferences + non_environnement
        ),
    )


def vectoriser(profil: ProfilCandidat, couverture: CouvertureProfil | None = None) -> np.ndarray:
    """Transforme un profil en vecteur numérique de taille `len(noms_features())`.

    `couverture` est injectable pour éviter de refaire la résolution quand
    l'appelant l'a déjà calculée (`outils.analyser_profil`) : chaque résolution
    peut déclencher le modèle d'embedding, et la refaire deux fois par appel
    doublait la latence sur un profil contenant un terme hors vocabulaire.
    """
    couverture = couverture if couverture is not None else analyser_couverture(profil)
    vecteur = (
        _multi_hot(couverture.matieres, VOCAB_MATIERES)
        + _multi_hot(couverture.competences, VOCAB_COMPETENCES)
        + _multi_hot(couverture.centres_interet, VOCAB_CENTRES_INTERET)
        + _multi_hot(couverture.preferences_professionnelles, VOCAB_PREFERENCES_PRO)
        + _notes(profil.resultats_scolaires, VOCAB_MATIERES)
        + _one_hot(couverture.environnement, VOCAB_ENVIRONNEMENTS)
    )
    return np.array(vecteur, dtype=float)
