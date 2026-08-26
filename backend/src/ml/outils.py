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

from src.ml.entrainement import entrainer_baseline, preparer_jeu_entrainement
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


@functools.lru_cache(maxsize=1)
def _modele():
    X, y = preparer_jeu_entrainement()
    if X.size == 0:
        raise RuntimeError(
            "Aucun jeu de données ML disponible : lancer "
            "`python -m src.ml.donnees_synthetiques` pour le générer."
        )
    return entrainer_baseline(X, y)


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


def _justification(vecteur: np.ndarray, modele, indice_classe: int, score: float) -> str:
    contributions = _contributions_pour_classe(vecteur, modele, indice_classe)
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
    modele = _modele()
    couverture = analyser_couverture(candidat)
    vecteur = vectoriser(candidat)
    probabilites = modele.predict_proba(vecteur.reshape(1, -1))[0]

    classes_et_probas = enumerate(zip(modele.classes_, probabilites, strict=True))
    candidats = sorted(
        (
            RecommandationParcours(
                parcours=parcours,
                score_adequation=float(proba),
                justification=_justification(vecteur, modele, indice, float(proba)),
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
            parcours_candidats=candidats,
            confiance=0.0,
            justification=_justification_profil_inexploitable(couverture),
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


def classer_parcours(profil: ProfilCandidat, top_k: int = 3) -> list[RecommandationParcours]:
    """Les `top_k` parcours les mieux classés pour ce profil."""
    return analyser_profil(profil).parcours_candidats[:top_k]


def calculer_adequation(profil: ProfilCandidat, parcours: str) -> float:
    """Score d'adéquation pour un parcours précis. 0.0 si le parcours est
    inconnu du modèle (jamais une erreur non gérée)."""
    for candidat in analyser_profil(profil).parcours_candidats:
        if candidat.parcours == parcours:
            return candidat.score_adequation
    return 0.0


def identifier_points_forts(profil: ProfilCandidat, top_n: int = 3) -> list[str]:
    """Les traits déclarés qui pèsent le plus dans la meilleure recommandation.

    Utilise les coefficients de la classe recommandée en tête (pas une
    importance globale toutes classes confondues), croisés avec les traits
    effectivement présents dans ce profil.
    """
    modele = _modele()
    vecteur = vectoriser(profil)
    probabilites = modele.predict_proba(vecteur.reshape(1, -1))[0]
    indice_meilleur = int(np.argmax(probabilites))

    contributions = _contributions_pour_classe(vecteur, modele, indice_meilleur)
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
