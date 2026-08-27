"""Le modèle exposé comme outil appelable, pas isolé dans un notebook (§6 du sujet).

Reprend exactement les 4 signatures citées par le sujet :
`analyser_profil(candidat)`, `classer_parcours(profil)`,
`calculer_adequation(profil, parcours)`, `identifier_points_forts(profil)`.

Modèle de production : la régression logistique (voir `entrainement.py` pour
la comparaison mesurée qui justifie ce choix face à la forêt aléatoire), dont
les coefficients par classe permettent en prime des justifications propres à
chaque parcours (quels traits déclarés pèsent pour *ce* parcours précisément),
plutôt qu'une importance globale non spécifique à la classe.

Le modèle est entraîné une fois par processus (mis en cache) sur le jeu de
données synthétique, plutôt que persisté en fichier binaire : sur un jeu de
cette taille l'entraînement prend une fraction de seconde, et ça évite les
problèmes de compatibilité d'un fichier `.joblib` figé entre versions de
scikit-learn sur les machines de l'équipe. `entrainement.entrainer_et_sauvegarder()`
reste disponible pour qui veut un artefact persisté (livrable alternatif
accepté par le sujet).
"""

import functools
import logging

import numpy as np

from src.ml.entrainement import (
    entrainer_baseline,
    entrainer_baseline_calibree,
    preparer_jeu_entrainement,
)
from src.ml.features import CouvertureProfil, analyser_couverture, noms_features, vectoriser
from src.ml.hybride import appliquer_regles_admission
from src.schemas import AnalyseProfil, ProfilCandidat, RecommandationParcours

logger = logging.getLogger(__name__)

PREFIXES_LISIBLES = {
    "matiere": "l'intérêt déclaré pour la matière",
    "competence": "la compétence déclarée",
    "interet": "le centre d'intérêt déclaré",
    "preference_pro": "la préférence professionnelle déclarée",
    "note": "le bon résultat scolaire en",
    "environnement": "l'environnement de travail recherché",
}


def _libelle_lisible(nom_feature: str) -> str:
    prefixe, _, valeur = nom_feature.partition(":")
    libelle_prefixe = PREFIXES_LISIBLES.get(prefixe, prefixe)
    return f"{libelle_prefixe} : {valeur.replace('_', ' ')}"


def _jeu_ou_erreur():
    X, y = preparer_jeu_entrainement()
    if X.size == 0:
        raise RuntimeError(
            "Aucun jeu de données ML disponible : lancer "
            "`python -m src.ml.donnees_synthetiques` pour le générer."
        )
    return X, y


@functools.lru_cache(maxsize=1)
def _modele():
    """Modèle servant les **probabilités**, calibré (voir `entrainement`).

    Sans calibration, le score affiché à un candidat était systématiquement
    sous-estimé de 12 points et ne correspondait à aucune fréquence réelle.
    """
    X, y = _jeu_ou_erreur()
    return entrainer_baseline_calibree(X, y)


@functools.lru_cache(maxsize=1)
def _modele_explicatif():
    """Modèle servant les **justifications**, non calibré.

    `CalibratedClassifierCV` n'expose pas de `coef_` : les contributions par
    trait, qui répondent à « pourquoi CE parcours ? », viennent donc de la
    régression logistique brute entraînée sur les mêmes données.

    Ce n'est pas un contournement. La calibration isotonique est une
    transformation **monotone** du score : elle change la valeur affichée, pas
    l'ordre ni le signe des contributions qui l'ont produite. L'explication
    reste donc celle de la décision réellement prise.
    """
    X, y = _jeu_ou_erreur()
    return entrainer_baseline(X, y)


# Modèle imposé de l'extérieur, uniquement pour l'évaluation (voir ci-dessous).
_modele_impose = None


def imposer_modele_pour_evaluation(modele) -> None:
    """Force le modèle utilisé par les outils, le temps d'une évaluation.

    `_modele()` entraîne sur **tout** le jeu de données disponible, ce qui est
    le bon comportement en production mais rendrait sans valeur toute mesure du
    chemin de production : les profils de test auraient été vus à
    l'entraînement. `evaluer_chemin_de_production` (ml.evaluation) impose donc
    ici un modèle entraîné sur le seul jeu d'entraînement.

    `None` rétablit le modèle de production. À n'utiliser que depuis un script
    d'évaluation, et toujours dans un `try/finally`.
    """
    global _modele_impose
    _modele_impose = modele


def _modele_courant():
    return _modele_impose if _modele_impose is not None else _modele()


@functools.lru_cache(maxsize=1)
def _graphe_admission():
    """Graphe de connaissances utilisé pour les règles d'admission (ONTO-2).

    Construit paresseusement et mis en cache : les prérequis ne changent pas
    au fil des requêtes. Retourne `None` si le corpus ou le graphe ne sont pas
    disponibles — le volet hybride est un bonus, il ne doit jamais empêcher le
    modèle de répondre (`hybride.appliquer_regles_admission` laisse alors le
    classement intact).
    """
    try:
        from src.graphe import construire_graphe
        from src.models import charger_corpus_formations

        return construire_graphe(charger_corpus_formations())
    except Exception:  # noqa: BLE001 — l'enrichissement symbolique est optionnel
        logger.warning("Graphe d'admission indisponible : classement ML non filtré", exc_info=True)
        return None


def _contributions_pour_classe(vecteur: np.ndarray, modele, indice_classe: int) -> np.ndarray:
    """Contribution de chaque trait déclaré au score de la classe `indice_classe`.

    Le poids (`coef_`) est spécifique à la régression logistique multinomiale :
    chaque classe a ses propres coefficients, donc cette contribution répond
    à « pourquoi CE parcours ? », pas à une importance globale du modèle.
    """
    return vecteur * modele.coef_[indice_classe]


def _justification(vecteur: np.ndarray, parcours: str, score: float) -> str:
    """Explique le score d'un parcours par les traits déclarés qui y pèsent.

    Le parcours est désigné par son **nom** et non par un indice : le modèle de
    probabilités (calibré) et le modèle explicatif (brut) n'ont aucune raison
    de ranger leurs classes dans le même ordre, et croiser un indice de l'un
    avec les coefficients de l'autre attribuerait silencieusement l'explication
    d'un parcours à un autre.
    """
    explicatif = _modele_explicatif()
    classes = list(explicatif.classes_)
    if parcours not in classes:
        return f"Score d'adéquation de {score:.0%} estimé par le modèle de recommandation."

    contributions = _contributions_pour_classe(vecteur, explicatif, classes.index(parcours))
    noms = noms_features()
    meilleur = int(np.argmax(contributions))
    if contributions[meilleur] <= 0:
        return f"Score d'adéquation de {score:.0%} estimé par le modèle de recommandation."
    return (
        f"Score d'adéquation de {score:.0%}, porté notamment par "
        f"{_libelle_lisible(noms[meilleur])}."
    )


def analyser_profil(candidat: ProfilCandidat) -> AnalyseProfil:
    """Sortie brute du modèle sur un profil (§6 du sujet) : tous les parcours
    connus, classés par score d'adéquation décroissant.

    **Refuse d'affirmer sur un profil qu'il n'a pas exploité.** Si trop peu de
    traits déclarés ont pu être ramenés au vocabulaire du modèle (voir
    `features.analyser_couverture`), les probabilités retombent sur la
    distribution a priori des classes : les scores existent numériquement mais
    ne portent aucune information sur *ce candidat-là*. La confiance est alors
    mise à zéro, ce qui déclenche naturellement l'escalade en aval
    (`agent._appliquer_controles_deterministes` compare au seuil configuré) au
    lieu de présenter un classement d'apparence normale.
    """
    modele = _modele_courant()
    # Une seule résolution du profil, réutilisée par la vectorisation : chaque
    # résolution peut déclencher le modèle d'embedding pour un terme hors
    # vocabulaire, et la refaire doublait la latence de l'appel.
    couverture = analyser_couverture(candidat)
    vecteur = vectoriser(candidat, couverture)
    probabilites = modele.predict_proba(vecteur.reshape(1, -1))[0]

    classes_et_probas = enumerate(zip(modele.classes_, probabilites, strict=True))
    candidats = sorted(
        (
            RecommandationParcours(
                parcours=parcours,
                score_adequation=float(proba),
                justification=_justification(vecteur, parcours, float(proba)),
            )
            for indice, (parcours, proba) in classes_et_probas
        ),
        key=lambda c: c.score_adequation,
        reverse=True,
    )

    # Volet hybride (§6 du sujet) : les parcours auxquels le candidat n'est pas
    # admissible passent derrière ceux qui lui sont accessibles. Le modèle ne
    # voit jamais la série de baccalauréat, il peut donc placer en tête une
    # formation dans laquelle le candidat ne pourrait pas s'inscrire.
    candidats = appliquer_regles_admission(candidats, candidat, _graphe_admission())

    if not couverture.exploitable:
        return AnalyseProfil(
            # L'avertissement est recopié dans la justification de **chaque**
            # candidat : `classer_parcours` et `calculer_adequation` ne renvoient
            # pas l'`AnalyseProfil`, et un score nu sorti d'ici serait
            # indiscernable d'un score informatif.
            parcours_candidats=[
                c.model_copy(update={"justification": AVERTISSEMENT_NON_EXPLOITABLE})
                for c in candidats
            ],
            confiance=0.0,
            justification=_justification_profil_inexploitable(couverture),
            profil_exploitable=False,
            elements_non_reconnus=couverture.non_reconnus,
        )

    confiance = candidats[0].score_adequation
    justification = (
        f"Le parcours {candidats[0].parcours} obtient le score le plus élevé "
        f"({confiance:.0%}) parmi les parcours accessibles au candidat, d'après le "
        "modèle entraîné sur le profil déclaré."
    )
    if couverture.non_reconnus:
        justification += (
            " Éléments déclarés non rattachés au vocabulaire du modèle, donc non pris "
            f"en compte dans ce score : {', '.join(couverture.non_reconnus)}."
        )
    return AnalyseProfil(
        parcours_candidats=candidats,
        confiance=confiance,
        justification=justification,
        profil_exploitable=True,
        elements_non_reconnus=couverture.non_reconnus,
    )


AVERTISSEMENT_NON_EXPLOITABLE = (
    "Score non informatif : le profil déclaré n'a pas pu être rattaché au "
    "vocabulaire du modèle. Cette valeur reflète la distribution générale des "
    "parcours, pas ce candidat, et ne doit pas fonder une recommandation."
)


def _justification_profil_inexploitable(couverture: CouvertureProfil) -> str:
    if couverture.non_reconnus:
        return (
            "Profil insuffisamment exploitable par le modèle : les éléments déclarés "
            f"({', '.join(couverture.non_reconnus)}) n'ont pas pu être rattachés à son "
            "vocabulaire. Les scores ci-dessous reflètent la distribution générale des "
            "parcours, pas ce candidat — ils ne doivent pas fonder une recommandation."
        )
    return (
        "Profil trop peu renseigné pour que le modèle produise un score informatif. "
        "Les scores ci-dessous reflètent la distribution générale des parcours, pas ce "
        "candidat — ils ne doivent pas fonder une recommandation."
    )


# Un parcours n'est proposé que si son score atteint cette fraction du score de
# tête. Valeur **mesurée**, pas choisie : voir `selectionner_significatifs`.
PART_MINIMALE_DU_LEADER = 0.20

# Plafond de parcours proposés, même quand plusieurs sont significatifs.
MAXIMUM_PROPOSES = 3


def selectionner_significatifs(
    candidats: list[RecommandationParcours],
    part_minimale: float = PART_MINIMALE_DU_LEADER,
    maximum: int = MAXIMUM_PROPOSES,
) -> list[RecommandationParcours]:
    """Ne garde que les parcours réellement distinguables du reste.

    **Le problème que ça corrige.** Un top-3 systématique présentait du bruit
    comme une recommandation : sur le jeu de test, le score médian est de 92 %
    au rang 1 mais de 2,0 % au rang 2 et 1,1 % au rang 3, séparés par six
    dixièmes de point. 94 % des profils ont un rang 3 sous 5 %. Ces places-là
    ne portent aucune information — et c'est exactement ce qui rendait le
    classement instable : retirer un seul trait déclaré changeait la
    composition du top-3 pour **34 %** des profils.

    La bonne réponse n'est pas de stabiliser du bruit, c'est de cesser de le
    présenter. Le critère est **relatif au leader** plutôt qu'un seuil absolu :
    il s'adapte à la forme de la distribution au lieu de supposer une échelle.

    **Le seuil est mesuré.** Balayage sur les 200 profils de test, stabilité de
    la sélection sous perturbation (retrait d'un trait déclaré) :

        part du leader   sélection stable   profils avec >1 option   rappel
              0,10             84 %                  15 %            100 %
              0,15             88 %                   8 %            100 %
        ->    0,20             94 %                   4 %            100 %
              0,30             97 %                   2 %            100 %

    0,20 retenu : le rappel reste à 100 % à tous les seuils (le bon parcours
    est quasi toujours en tête), donc restreindre ne coûte rien en justesse ;
    au-delà de 0,20 le gain de stabilité se paie en n'affichant plus jamais
    d'alternative, y compris quand le modèle hésite vraiment.

    Toujours au moins un parcours : une liste vide priverait l'appelant de la
    sortie du modèle sans rien dire de plus.
    """
    if not candidats:
        return []
    tete = candidats[0].score_adequation
    if tete <= 0:
        return candidats[:1]
    retenus = [c for c in candidats[:maximum] if c.score_adequation >= part_minimale * tete]
    return retenus or candidats[:1]


def classer_parcours(
    profil: ProfilCandidat, top_k: int = MAXIMUM_PROPOSES
) -> list[RecommandationParcours]:
    """Les parcours à proposer pour ce profil.

    Ne retourne **pas** systématiquement `top_k` éléments : seuls les parcours
    significativement distinguables du leader sont proposés (voir
    `selectionner_significatifs`). Un profil sur lequel le modèle est net
    donne une seule proposition, un profil ambigu en donne plusieurs — ce qui
    est l'information utile, là où un top-3 constant la masquait.
    """
    return selectionner_significatifs(
        analyser_profil(profil).parcours_candidats, maximum=top_k
    )


def calculer_adequation(profil: ProfilCandidat, parcours: str) -> float:
    """Score d'adéquation pour un parcours précis. 0.0 si le parcours est
    inconnu du modèle (jamais une erreur non gérée)."""
    for candidat in analyser_profil(profil).parcours_candidats:
        if candidat.parcours == parcours:
            return candidat.score_adequation
    return 0.0


def identifier_points_forts(
    profil: ProfilCandidat, top_n: int = 3, couverture: CouvertureProfil | None = None
) -> list[str]:
    """Les traits déclarés qui pèsent le plus dans la meilleure recommandation.

    Utilise les coefficients de la classe recommandée en tête (pas une
    importance globale toutes classes confondues), croisés avec les traits
    effectivement présents dans ce profil.

    `couverture` est injectable pour la même raison que dans `vectoriser` :
    éviter une seconde résolution du même profil quand l'appelant l'a déjà.
    """
    modele = _modele_courant()
    vecteur = vectoriser(profil, couverture)
    probabilites = modele.predict_proba(vecteur.reshape(1, -1))[0]
    # Le parcours de tête est déterminé par le modèle de production, mais ses
    # contributions sont lues dans le modèle explicatif : on passe donc par le
    # **nom** de la classe, les deux modèles pouvant les ordonner différemment.
    parcours_de_tete = list(modele.classes_)[int(np.argmax(probabilites))]

    explicatif = _modele_explicatif()
    classes_explicatives = list(explicatif.classes_)
    if parcours_de_tete not in classes_explicatives:
        return []

    contributions = _contributions_pour_classe(
        vecteur, explicatif, classes_explicatives.index(parcours_de_tete)
    )
    noms = noms_features()

    ordre = np.argsort(contributions)[::-1]
    retenus = [i for i in ordre if contributions[i] > 0][:top_n]
    # `_libelle_lisible` est pensé pour s'insérer en milieu de phrase
    # (`_justification`) : la première lettre est mise en capitale ici pour
    # un usage en liste autonome (`.capitalize()` mettrait aussi en
    # minuscule le reste de la phrase, ce qu'on ne veut pas).
    def _en_debut_de_liste(libelle: str) -> str:
        return libelle[0].upper() + libelle[1:]

    return [_en_debut_de_liste(_libelle_lisible(noms[i])) for i in retenus]
