"""Entraînement des modèles de recommandation de parcours (ML-3, ML-4).

Deux approches comparées (§6 du sujet : choix justifié par le besoin métier,
comparaison d'au moins deux approches) :

- **Baseline** : régression logistique multinomiale. Linéaire, rapide,
  directement interprétable (coefficients par classe).
- **Modèle comparé** : forêt aléatoire. Capture des interactions entre traits
  qu'un modèle linéaire ne peut pas représenter, et fournit des probabilités
  par classe utilisables comme score d'adéquation.

Les deux tâches sont formulées comme une classification multi-classes (16
parcours) : la probabilité prédite pour chaque classe sert directement de
« score d'adéquation » (`RecommandationParcours.score_adequation`), ce qui
couvre à la fois la classification et le classement des parcours par
compatibilité (§6 du sujet) sans dupliquer le modèle.

Choix volontaire : pas de LightGBM ni de dépendance de boosting
supplémentaire. Sur un jeu de cette taille (quelques centaines de profils
synthétiques), une forêt aléatoire de scikit-learn donne une deuxième
approche suffisamment différente de la baseline linéaire, sans ajouter de
dépendance à la chaîne d'installation d'une équipe de hackathon.

**Résultat mesuré, pas supposé** (voir `backend/tests/eval_results_ml.json`) :
sur le jeu synthétique calibré (voir la note de biais dans
`donnees_synthetiques.py`), la régression logistique généralise mieux
(≈ 99,5 % d'exactitude) que la forêt aléatoire (≈ 86 %), qui overfitte
davantage sur un jeu de cette taille avec ce niveau de bruit. C'est pourquoi
`outils.py` (ML-8) utilise la régression logistique en production — le choix
suit la mesure, pas l'inverse. La forêt aléatoire reste le second modèle de
comparaison exigé par le sujet, et redevient un candidat sérieux si le jeu
de données grossit (enquête réelle, ML-7) ou si des interactions entre
traits s'avèrent importantes.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import config
from src.ml.donnees_synthetiques import charger_jeu_de_donnees
from src.ml.features import vectoriser
from src.schemas import ProfilCandidat

RANDOM_STATE_DEFAUT = 42

# Deux probabilités plus proches que cela ne désignent plus un vainqueur : la
# différence est du bruit de virgule flottante, pas un écart de modèle.
_TOLERANCE_EX_AEQUO = 1e-9
CHEMIN_MODELE_DEFAUT = config.dossier_data / "ml" / "modele_recommandation.joblib"


def preparer_jeu_entrainement(exemples: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorise le jeu de données synthétique en `(X, y)`.

    `exemples` est injectable pour les tests ; par défaut, charge le jeu
    généré par `donnees_synthetiques.py` depuis `backend/data/ml/`.
    """
    exemples = exemples if exemples is not None else charger_jeu_de_donnees()
    if not exemples:
        return np.empty((0, 0)), np.empty((0,), dtype=str)

    X = np.array([vectoriser(ProfilCandidat.model_validate(e["profil"])) for e in exemples])
    y = np.array([e["parcours_id"] for e in exemples])
    return X, y


def separer_train_test(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = RANDOM_STATE_DEFAUT
):
    """Split stratifié : chaque parcours doit être représenté dans le test,
    sans quoi les métriques par classe (ML-5) seraient incalculables pour lui.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def separer_indices(
    y: np.ndarray, test_size: float = 0.25, random_state: int = RANDOM_STATE_DEFAUT
):
    """Indices du même découpage stratifié que `separer_train_test`.

    Permet de découper en parallèle des structures que `separer_train_test` ne
    voit pas — en particulier les profils d'origine, dont l'évaluation du
    chemin de production a besoin (elle appelle `analyser_profil` sur un
    `ProfilCandidat`, pas sur un vecteur). Mêmes paramètres et même graine :
    la partition est garantie identique.
    """
    return train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state
    )


def entrainer_baseline(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_STATE_DEFAUT
) -> LogisticRegression:
    modele = LogisticRegression(max_iter=1000, random_state=random_state)
    modele.fit(X_train, y_train)
    return modele


def entrainer_baseline_calibree(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_STATE_DEFAUT
) -> CalibratedClassifierCV:
    """Baseline dont les probabilités sont recalées sur la réalité observée.

    **Le défaut corrigé.** La régression logistique brute est nettement
    **sous-confiante** sur ce jeu : confiance moyenne 0,879 pour une exactitude
    de 0,995, soit un écart de −0,116 (ECE 0,120). Toutes les tranches
    au-dessus de 0,4 affichent 100 % de réussite pour une confiance annoncée
    bien inférieure. Un « 62 % d'adéquation » montré à un candidat ne
    correspondait donc à rien de mesurable.

    **Méthode choisie sur mesure, pas par habitude** — ECE sur le même
    découpage :

        brut (aucune calibration)   0,120
        Platt / sigmoïde            0,196   (aggrave : écartée)
        ->  isotonique              0,029

    L'isotonique divise l'écart par quatre. `cv=5` est indispensable : ajuster
    la calibration sur les données d'entraînement du modèle la ferait
    surapprendre, d'autant plus avec 16 classes.

    **Effet contrôlé sur le garde-fou d'escalade** : la part de profils sous
    `orchestrateur_seuil_confiance` passe de 1,0 % à 0,5 % — le mécanisme
    reste actif, la calibration ne le neutralise pas.

    **Limite à ne pas masquer.** Cette calibration est ajustée sur des profils
    **synthétiques**, où la tâche est presque parfaitement séparable. Elle
    apprend donc « ce modèle a presque toujours raison », ce qui sera faux sur
    de vrais candidats : appliquée telle quelle, elle rendrait le système
    sur-confiant là où il l'était peu. Elle doit être réajustée sur les
    réponses d'enquête dès qu'elles existent (DATA-4, ML-7).
    """
    calibre = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, random_state=random_state),
        method="isotonic",
        cv=5,
    ).fit(X_train, y_train)
    # Modèle de secours pour les profils que l'isotonique ne sait pas départager
    # (voir `ModeleBorne.predict_proba`). Entraîné sur les mêmes données, donc
    # sans fuite : c'est la même information, lue avant l'escalier.
    secours = entrainer_baseline(X_train, y_train, random_state=random_state)
    return ModeleBorne(calibre, len(y_train), secours)


class ModeleBorne:
    """Enveloppe qui empêche une probabilité d'atteindre exactement 0 ou 1.

    **Pourquoi c'est nécessaire.** L'isotonique est une fonction en escalier :
    sur une tâche presque parfaitement séparable comme le jeu synthétique, sa
    dernière marche vaut 1,0 et **68 % des profils ressortaient à exactement
    100 %**. Ces profils sont effectivement classés juste 100 % du temps, donc
    la calibration n'est pas fausse au sens statistique — mais annoncer « 100 %
    d'adéquation » à un candidat revient à affirmer une certitude sur son
    avenir, ce qu'aucun modèle entraîné sur 600 profils ne peut soutenir. Le
    sujet demande une recommandation *prudente* (§2) et un système qui
    *reconnaît l'incertitude* (§9).

    **La borne est dérivée des données, pas choisie.** Règle de trois : n
    observations sans erreur ne justifient pas p = 1, mais au mieux
    p ≈ 1 − 3/n à 95 % de confiance. Avec n exemples d'entraînement, la marge
    vaut donc `3/n` — soit 0,5 % ici, assez pour supprimer la certitude
    absolue sans déformer la mesure (l'ECE bouge de moins d'un millième).

    L'enveloppe est appliquée **avant** l'évaluation comme avant la production :
    mesurer le modèle non borné pendant qu'on en sert un borné recréerait
    l'écart « modèle évalué ≠ modèle servi » que le §8 interdit.
    """

    def __init__(self, modele, n_entrainement: int, secours=None):
        self._modele = modele
        # Régression logistique non calibrée, consultée uniquement quand
        # l'escalier isotonique ne départage plus rien (voir `predict_proba`).
        self._secours = secours
        # Règle de trois, bornée pour rester sensée sur un très petit jeu.
        self.marge = min(3.0 / max(n_entrainement, 1), 0.05)
        self.classes_ = modele.classes_

    def _rangs_perdus(self, probabilites: np.ndarray) -> np.ndarray:
        """Lignes où la sortie calibrée ne désigne plus de premier parcours.

        L'isotonique est une fonction en escalier, ajustée un-contre-tous :
        chacun des 16 calibrateurs projette le score brut sur une marche. Quand
        les scores bruts d'un profil tombent tous dans la même bande basse,
        les 16 calibrateurs renvoient la même valeur et la normalisation rend
        la distribution **exactement uniforme**. L'`argmax` échoit alors au
        premier parcours par ordre alphabétique — AEE — quel que soit le profil.
        """
        if probabilites.shape[1] < 2:
            return np.zeros(len(probabilites), dtype=bool)
        deux_premiers = np.partition(probabilites, -2, axis=1)[:, -2:]
        return np.isclose(deux_premiers[:, 1], deux_premiers[:, 0], atol=_TOLERANCE_EX_AEQUO)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilités calibrées, bornées, et réparées quand elles sont muettes.

        **Le mélange uniforme** `(1-m)·p + m/K` empêche d'atteindre 0 ou 1. Un
        simple `clip` par le bas serait faux ici : avec K = 16 classes, un
        plancher de 0,005 force 7,5 % de masse dans la queue, et la
        renormalisation fait tomber le sommet à 0,93 — l'ECE remontait alors à
        0,094. Le mélange, lui, somme exactement à 1 par construction, préserve
        l'ordre des classes, et ne déplace la masse qu'à hauteur de `marge`.

        **La réparation** substitue les probabilités brutes sur les seules
        lignes où la calibration ne distingue plus le premier parcours du
        deuxième. Mesuré : ce cas ne survient **jamais** sur le jeu synthétique
        (0/200 profils de test), mais frappe **4 profils réels sur 14** — les
        profils peu fournis, que le sujet traite comme normaux. Le défaut était
        donc invisible à la mesure synthétique tout en produisant, en
        production, un classement sans rapport avec le candidat.

        Le repli ne fabrique aucune confiance : les probabilités brutes sont
        **sous-confiantes** (0,22 là où l'escalier annonçait 0,0625 pour tout le
        monde), restent sous `orchestrateur_seuil_confiance` et déclenchent donc
        toujours l'escalade vers un conseiller. Ce qui change est l'**ordre** —
        le candidat voit enfin le parcours qui lui correspond, signalé comme
        à confirmer, au lieu d'un parcours tiré par ordre alphabétique.
        """
        probabilites = self._modele.predict_proba(X)
        if self._secours is not None:
            a_reparer = self._rangs_perdus(probabilites)
            if a_reparer.any():
                probabilites = probabilites.copy()
                probabilites[a_reparer] = self._secours.predict_proba(X[a_reparer])
        nombre_de_classes = probabilites.shape[1]
        return (1.0 - self.marge) * probabilites + self.marge / nombre_de_classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classe retenue, **toujours** l'`argmax` de `predict_proba`.

        Déléguer au modèle calibré serait un piège : sur une ligne réparée, il
        renverrait le parcours alphabétique que la réparation vient d'écarter,
        et le système servirait un `predict` en contradiction avec les scores
        qu'il affiche.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def entrainer_foret(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_STATE_DEFAUT
) -> RandomForestClassifier:
    modele = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state, class_weight="balanced"
    )
    modele.fit(X_train, y_train)
    return modele


def entrainer_et_sauvegarder(chemin: Path | None = None) -> Path:
    """Entraîne le modèle de production et le persiste en `.joblib`.

    Alternative acceptée par le sujet au script reproductible : un artefact
    déjà entraîné, prêt à charger sans ré-entraînement. Ce fichier n'est pas
    versionné (voir `.gitignore`) — un pickle scikit-learn ne garantit pas
    d'être rechargeable avec une version différente de la bibliothèque, il
    doit donc rester reproductible à la demande plutôt que figé dans le dépôt.

    **Correctif d'audit (AUDIT-ML-7)** : cette fonction persistait la
    régression logistique **non calibrée** (`entrainer_baseline`), alors que
    `ml.outils._modele()` sert en production la version **calibrée**
    (`entrainer_baseline_calibree`) — les scores affichés à un candidat sont
    recalés sur la réalité observée, voir la note de biais de
    `entrainer_baseline_calibree`. Le `.joblib` produit ici était donc un
    autre modèle que celui réellement mesuré et servi. Corrigé pour entraîner
    la même fonction que la production, verrouillé par
    `test_entrainer_et_sauvegarder_persiste_le_meme_modele_que_la_production`.
    """
    X, y = preparer_jeu_entrainement()
    if X.size == 0:
        raise RuntimeError(
            "Aucun jeu de données ML disponible : lancer "
            "`python -m src.ml.donnees_synthetiques` pour le générer."
        )
    modele = entrainer_baseline_calibree(X, y)
    chemin = chemin or CHEMIN_MODELE_DEFAUT
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, chemin)
    return chemin


if __name__ == "__main__":
    chemin = entrainer_et_sauvegarder()
    print(f"Modèle entraîné et sauvegardé dans {chemin}")
