"""Analyse exploratoire du jeu de données (ML-1, §7 du sujet).

Le sujet attend « une analyse exploratoire des données » et, parmi les
livrables, « les notebooks d'analyse et d'entraînement ». Le calcul vit ici
plutôt que dans le notebook, pour trois raisons :

- il est **testable** (`backend/tests/ml/test_exploration.py`), là où le code
  enfermé dans des cellules ne l'est pas ;
- il est **reproductible en une commande**, sans dépendre de l'ordre
  d'exécution des cellules ni d'un état de kernel ;
- le notebook (`notebooks/01_analyse_exploratoire.ipynb`) reste le livrable
  attendu, mais se contente d'appeler ces fonctions et de commenter les
  résultats — ce qu'un notebook fait de mieux.

Ce que l'exploration cherche, sur un jeu **synthétique** : moins à découvrir
des faits sur le monde qu'à **vérifier les hypothèses de génération** et à
repérer les fuites. Une variable qui identifierait à elle seule la classe est
le défaut le plus coûteux ici — un a déjà été trouvé et corrigé de cette façon
(`environnement_travail_recherche`, voir `donnees_synthetiques.py`), et
`pouvoir_discriminant_par_champ()` existe pour que le suivant se voie.
"""

from collections import Counter
from typing import Any

import numpy as np

from src.ml.archetypes import (
    VOCAB_CENTRES_INTERET,
    VOCAB_COMPETENCES,
    VOCAB_ENVIRONNEMENTS,
    VOCAB_MATIERES,
    VOCAB_PREFERENCES_PRO,
)
from src.ml.features import noms_features, vectoriser
from src.schemas import ProfilCandidat

CHAMPS_LISTE = (
    ("matieres_preferees", VOCAB_MATIERES),
    ("competences_declarees", VOCAB_COMPETENCES),
    ("centres_interet", VOCAB_CENTRES_INTERET),
    ("preferences_professionnelles", VOCAB_PREFERENCES_PRO),
)


def _profils(exemples: list[dict]) -> list[ProfilCandidat]:
    return [
        e["profil"]
        if isinstance(e["profil"], ProfilCandidat)
        else ProfilCandidat.model_validate(e["profil"])
        for e in exemples
    ]


def distribution_des_classes(exemples: list[dict]) -> dict[str, Any]:
    """Effectif par parcours, et déséquilibre éventuel (§7 : « déséquilibres »)."""
    compte = Counter(e["parcours_id"] for e in exemples)
    effectifs = list(compte.values())
    return {
        "nombre_de_classes": len(compte),
        "effectif_total": len(exemples),
        "par_classe": dict(sorted(compte.items())),
        "effectif_min": min(effectifs) if effectifs else 0,
        "effectif_max": max(effectifs) if effectifs else 0,
        "equilibre": len(set(effectifs)) == 1,
    }


def completude_des_champs(exemples: list[dict]) -> dict[str, Any]:
    """Taux de renseignement de chaque champ du profil.

    Un champ jamais renseigné n'est pas une statistique anodine : il signale
    une capacité que le modèle ne peut pas apprendre. `serie_bac` en est
    l'exemple vivant — jamais généré, donc les règles d'admission du volet
    hybride restent inertes sur ce jeu (voir `eval_ml.py`).
    """
    profils = _profils(exemples)
    total = len(profils) or 1
    champs = [nom for nom, _ in CHAMPS_LISTE] + [
        "resultats_scolaires",
        "environnement_travail_recherche",
        "activites_projets",
        "serie_bac",
    ]
    return {
        champ: {
            "renseigne": sum(1 for p in profils if getattr(p, champ)),
            "taux": sum(1 for p in profils if getattr(p, champ)) / total,
        }
        for champ in champs
    }


def traits_les_plus_frequents(exemples: list[dict], top_n: int = 10) -> dict[str, list]:
    """Traits les plus déclarés, par champ — la « distribution » du §7."""
    profils = _profils(exemples)
    resultat = {}
    for champ, _ in CHAMPS_LISTE:
        compte = Counter(trait for p in profils for trait in getattr(p, champ))
        resultat[champ] = [
            {"trait": t, "occurrences": n} for t, n in compte.most_common(top_n)
        ]
    compte_env = Counter(
        p.environnement_travail_recherche for p in profils if p.environnement_travail_recherche
    )
    resultat["environnement_travail_recherche"] = [
        {"trait": t, "occurrences": n} for t, n in compte_env.most_common(top_n)
    ]
    return resultat


def pouvoir_discriminant_par_champ(exemples: list[dict]) -> dict[str, Any]:
    """Détecte une variable qui identifierait à elle seule le parcours.

    Pour chaque champ, on mesure la part des profils dont la **valeur la plus
    fréquente pour leur classe** est unique à cette classe. Une valeur proche
    de 1 signale une fuite : le modèle n'aurait qu'à lire ce champ.

    C'est exactement le défaut trouvé pendant le calibrage — un
    `environnement_travail_recherche` déterministe par archétype donnait 100 %
    d'exactitude à n'importe quel modèle. Ce contrôle existe pour que le
    prochain se voie sans avoir à ré-entraîner quoi que ce soit.
    """
    profils_et_classes = list(zip(_profils(exemples), [e["parcours_id"] for e in exemples],
                                 strict=True))

    resultat: dict[str, Any] = {}
    for champ, _ in CHAMPS_LISTE + (("environnement_travail_recherche", VOCAB_ENVIRONNEMENTS),):
        # trait -> ensemble des classes où il apparaît
        classes_par_trait: dict[str, set[str]] = {}
        for profil, classe in profils_et_classes:
            valeur = getattr(profil, champ)
            traits = valeur if isinstance(valeur, list) else ([valeur] if valeur else [])
            for trait in traits:
                classes_par_trait.setdefault(trait, set()).add(classe)

        if not classes_par_trait:
            resultat[champ] = {"traits": 0, "part_traits_exclusifs": 0.0}
            continue

        exclusifs = sum(1 for classes in classes_par_trait.values() if len(classes) == 1)
        resultat[champ] = {
            "traits": len(classes_par_trait),
            "traits_exclusifs_a_une_classe": exclusifs,
            "part_traits_exclusifs": exclusifs / len(classes_par_trait),
            "classes_moyennes_par_trait": float(
                np.mean([len(c) for c in classes_par_trait.values()])
            ),
        }
    return resultat


def correlations_traits_classe(exemples: list[dict], top_n: int = 15) -> list[dict]:
    """Traits dont la présence est la plus corrélée à une classe précise.

    Corrélation point-bisériale entre la dimension du vecteur (0/1) et
    l'appartenance à une classe : c'est la lecture « quelles variables portent
    le signal » attendue au §7, calculée sur l'espace de features réellement
    utilisé par le modèle plutôt que sur les champs bruts.
    """
    profils = _profils(exemples)
    classes = np.array([e["parcours_id"] for e in exemples])
    X = np.array([vectoriser(p) for p in profils])
    noms = noms_features()

    correlations = []
    for classe in sorted(set(classes)):
        cible = (classes == classe).astype(float)
        for i, nom in enumerate(noms):
            colonne = X[:, i]
            if colonne.std() == 0 or cible.std() == 0:
                continue
            r = float(np.corrcoef(colonne, cible)[0, 1])
            correlations.append({"parcours": classe, "trait": nom, "correlation": r})

    correlations.sort(key=lambda c: abs(c["correlation"]), reverse=True)
    return correlations[:top_n]


def analyser(exemples: list[dict]) -> dict[str, Any]:
    """Analyse exploratoire complète, sérialisable en JSON (ML-1)."""
    return {
        "distribution_des_classes": distribution_des_classes(exemples),
        "completude_des_champs": completude_des_champs(exemples),
        "traits_les_plus_frequents": traits_les_plus_frequents(exemples),
        "pouvoir_discriminant_par_champ": pouvoir_discriminant_par_champ(exemples),
        "correlations_traits_classe": correlations_traits_classe(exemples),
        "dimension_de_l_espace_de_features": len(noms_features()),
    }
