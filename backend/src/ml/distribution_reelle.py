"""Distribution observée dans les enquêtes réelles, mesurée pour piloter la
génération synthétique (DATA-6 bis).

**Pourquoi ce module existe.** Le générateur d'origine
(`donnees_synthetiques.py`) tire le **même nombre** de profils par parcours et
remplit **tous** les champs de chaque profil. Les deux hypothèses sont
démenties par les enquêtes réellement collectées :

- *classes* — sur 93 étiquettes réelles résolues, IGGLIA en concentre 38,7 % et
  deux parcours (EMP, TEE) n'apparaissent jamais. Un jeu équiprobable apprend
  au modèle un a priori uniforme qu'aucun public réel ne présente ;
- *complétude* — 85 des 86 réponses de l'enquête courte ne renseignent qu'**un
  seul** des cinq champs multi-valeurs, là où un profil synthétique en renseigne
  quatre ou cinq. C'est la cause directe des 56/79 profils réels déclarés
  inexploitables par le garde-fou ML-9 : le modèle n'a jamais vu, à
  l'entraînement, de profil aussi mince que ceux qu'il reçoit en production.

Ce module **mesure** ces deux écarts sur les fichiers réels et expose de quoi
les répliquer. Il ne code aucun chiffre en dur : tout est recalculé depuis
`jeu_test_reel.json` et `reponses_orientia.json`, pour qu'une seconde vague de
collecte se répercute sans édition manuelle.

**Limite à ne pas masquer.** Les deux enquêtes ne sont pas un échantillon
aléatoire des futurs utilisateurs : l'une est concentrée sur les mentions
informatiques, l'autre compte 15 réponses. Le lissage (`prior_lisse`) est donc
un **choix assumé**, pas une estimation — recopier la distribution empirique
telle quelle laisserait quatre parcours à un ou deux profils d'entraînement et
deux parcours à zéro, ce qui remplacerait un biais par un pire.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.config import config

# Les cinq champs multi-valeurs d'un `ProfilCandidat`, dans l'ordre où
# `features.vectoriser` les consomme.
CHAMPS_MULTI_VALEURS: tuple[str, ...] = (
    "matieres_preferees",
    "competences_declarees",
    "centres_interet",
    "activites_projets",
    "preferences_professionnelles",
)

# Champs à valeur unique dont on mesure seulement la présence.
CHAMPS_SIMPLES: tuple[str, ...] = (
    "environnement_travail_recherche",
    "serie_bac",
)

# Part de la masse de probabilité laissée à la distribution empirique ; le
# complément revient à l'uniforme. Choisi (non mesuré) pour que la classe la
# plus fréquente reste nettement dominante sans qu'aucune classe ne descende
# sous le plancher — voir la limite documentée en tête de module.
ALPHA_LISSAGE = 0.6

# Part de l'effectif uniforme en dessous de laquelle aucune classe ne descend.
# 0,4 signifie qu'un parcours jamais observé conserve 40 % de ce qu'il aurait
# eu dans un jeu équiprobable, au lieu de zéro.
PLANCHER_RELATIF = 0.4


@dataclass(frozen=True)
class CompletudeChamp:
    """Ce qu'on observe réellement pour un champ donné dans une enquête."""

    taux_presence: float
    """Fraction des réponses où le champ est renseigné (non vide)."""

    tailles: tuple[int, ...] = ()
    """Tailles observées quand le champ est renseigné, une entrée par réponse.

    Conservée brute plutôt que résumée : rééchantillonner dans les tailles
    réellement observées évite de supposer une loi (normale, Poisson…) que
    93 réponses ne permettent pas de vérifier.
    """

    @property
    def taille_mediane(self) -> float:
        if not self.tailles:
            return 0.0
        triees = sorted(self.tailles)
        milieu = len(triees) // 2
        if len(triees) % 2:
            return float(triees[milieu])
        return (triees[milieu - 1] + triees[milieu]) / 2


@dataclass(frozen=True)
class DistributionReelle:
    """Photographie mesurée d'une ou plusieurs enquêtes réelles."""

    effectifs_parcours: dict[str, int] = field(default_factory=dict)
    completude: dict[str, CompletudeChamp] = field(default_factory=dict)
    n_reponses: int = 0
    n_etiquetees: int = 0
    sources: tuple[str, ...] = ()

    def prior(
        self,
        parcours_connus: tuple[str, ...],
        alpha: float = ALPHA_LISSAGE,
        plancher_relatif: float = PLANCHER_RELATIF,
    ) -> dict[str, float]:
        """Raccourci vers `prior_lisse` sur les effectifs mesurés."""
        return prior_lisse(
            self.effectifs_parcours,
            parcours_connus,
            alpha=alpha,
            plancher_relatif=plancher_relatif,
        )


def _profils_etiquetes(entrees: list[dict], cle_label: str) -> list[tuple[str | None, dict]]:
    """Extrait `(parcours_id, profil)` d'une liste d'enregistrements d'enquête.

    Tolère les deux formats produits par le projet : `jeu_test_reel.json`
    (`parcours_id`) et `reponses_orientia.json` (`parcours_declare`).
    """
    extraits: list[tuple[str | None, dict]] = []
    for entree in entrees:
        profil = entree.get("profil")
        if not isinstance(profil, dict):
            continue
        extraits.append((entree.get(cle_label) or None, profil))
    return extraits


def _charger(chemin: Path) -> list[dict]:
    """Charge un fichier d'enquête, ou une liste vide s'il n'existe pas.

    Un fichier absent est un cas normal (dépôt fraîchement cloné, collecte pas
    encore faite) : il ne doit pas faire échouer la génération, seulement la
    priver de ce qu'il aurait apporté.
    """
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        contenu = json.load(f)
    return contenu if isinstance(contenu, list) else []


def mesurer(
    chemin_jeu_test_reel: Path | None = None,
    chemin_reponses_enquete: Path | None = None,
) -> DistributionReelle:
    """Mesure effectifs par parcours et complétude par champ sur les enquêtes.

    Les deux sources sont additionnées : elles proviennent de deux
    questionnaires différents (l'un court, l'autre à 21 questions), et c'est
    précisément cette hétérogénéité qu'on veut répliquer — un système en
    production reçoit aussi bien une réponse d'une ligne qu'un formulaire
    complet.
    """
    dossier = config.dossier_data
    chemin_jeu_test_reel = chemin_jeu_test_reel or (dossier / "ml" / "jeu_test_reel.json")
    chemin_reponses_enquete = chemin_reponses_enquete or (
        dossier / "enquete" / "reponses_orientia.json"
    )

    sources: list[str] = []
    profils: list[tuple[str | None, dict]] = []

    externes = _charger(chemin_jeu_test_reel)
    if externes:
        profils += _profils_etiquetes(externes, "parcours_id")
        sources.append(chemin_jeu_test_reel.name)

    internes = _charger(chemin_reponses_enquete)
    if internes:
        profils += _profils_etiquetes(internes, "parcours_declare")
        sources.append(chemin_reponses_enquete.name)

    effectifs = Counter(pid for pid, _ in profils if pid)

    completude: dict[str, CompletudeChamp] = {}
    total = len(profils)
    for champ in CHAMPS_MULTI_VALEURS:
        tailles = [len(p.get(champ) or []) for _, p in profils]
        non_vides = tuple(t for t in tailles if t > 0)
        completude[champ] = CompletudeChamp(
            taux_presence=(len(non_vides) / total) if total else 0.0,
            tailles=non_vides,
        )
    for champ in (*CHAMPS_SIMPLES, "resultats_scolaires"):
        presents = sum(1 for _, p in profils if p.get(champ))
        completude[champ] = CompletudeChamp(
            taux_presence=(presents / total) if total else 0.0,
        )

    return DistributionReelle(
        effectifs_parcours=dict(effectifs),
        completude=completude,
        n_reponses=total,
        n_etiquetees=sum(effectifs.values()),
        sources=tuple(sources),
    )


def prior_lisse(
    effectifs: dict[str, int],
    parcours_connus: tuple[str, ...],
    alpha: float = ALPHA_LISSAGE,
    plancher_relatif: float = PLANCHER_RELATIF,
) -> dict[str, float]:
    """Mélange la distribution empirique et l'uniforme, avec un plancher.

    `alpha = 1` recopie l'enquête (et affame les parcours rares), `alpha = 0`
    revient au jeu équiprobable d'origine. Le plancher s'applique **après** le
    mélange, puis la distribution est renormalisée : sans renormalisation, un
    relèvement de plancher ferait sortir la somme de 1.

    Sans aucun effectif observé, retourne l'uniforme — le module se comporte
    alors exactement comme le générateur d'origine plutôt que d'échouer.
    """
    if not parcours_connus:
        return {}

    uniforme = 1.0 / len(parcours_connus)
    total = sum(effectifs.get(p, 0) for p in parcours_connus)
    if total <= 0:
        return dict.fromkeys(parcours_connus, uniforme)

    alpha = min(max(alpha, 0.0), 1.0)
    plancher = plancher_relatif * uniforme

    brut = {
        p: max(alpha * (effectifs.get(p, 0) / total) + (1 - alpha) * uniforme, plancher)
        for p in parcours_connus
    }
    masse = sum(brut.values())
    return {p: v / masse for p, v in brut.items()}


def effectifs_cibles(prior: dict[str, float], n_total: int) -> dict[str, int]:
    """Convertit un prior en effectifs entiers dont la somme vaut `n_total`.

    Méthode des plus forts restes : arrondir chaque classe indépendamment
    donnerait un total dérivant de `n_total`, ce qui rendrait la taille du jeu
    dépendante du prior et les tests non reproductibles.
    """
    if not prior or n_total <= 0:
        return {}

    exacts = {p: prior[p] * n_total for p in prior}
    effectifs = {p: int(v) for p, v in exacts.items()}
    reste = n_total - sum(effectifs.values())

    # Les classes dont la partie fractionnaire est la plus grande récupèrent les
    # unités restantes ; à égalité, l'ordre alphabétique tranche pour rester
    # déterministe d'une exécution à l'autre.
    ordre = sorted(prior, key=lambda p: (-(exacts[p] - effectifs[p]), p))
    for p in ordre[:reste]:
        effectifs[p] += 1
    return effectifs
