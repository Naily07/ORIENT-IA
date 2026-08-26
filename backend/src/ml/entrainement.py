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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import config
from src.ml.donnees_synthetiques import charger_jeu_de_donnees
from src.ml.features import vectoriser
from src.schemas import ProfilCandidat

RANDOM_STATE_DEFAUT = 42
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


def entrainer_baseline(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_STATE_DEFAUT
) -> LogisticRegression:
    modele = LogisticRegression(max_iter=1000, random_state=random_state)
    modele.fit(X_train, y_train)
    return modele


def entrainer_foret(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_STATE_DEFAUT
) -> RandomForestClassifier:
    modele = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state, class_weight="balanced"
    )
    modele.fit(X_train, y_train)
    return modele


def entrainer_et_sauvegarder(chemin: Path | None = None) -> Path:
    """Entraîne le modèle de production (régression logistique, sur tout le
    jeu de données disponible) et le persiste en `.joblib`.

    Alternative acceptée par le sujet au script reproductible : un artefact
    déjà entraîné, prêt à charger sans ré-entraînement. Ce fichier n'est pas
    versionné (voir `.gitignore`) — un pickle scikit-learn ne garantit pas
    d'être rechargeable avec une version différente de la bibliothèque, il
    doit donc rester reproductible à la demande plutôt que figé dans le dépôt.
    """
    X, y = preparer_jeu_entrainement()
    if X.size == 0:
        raise RuntimeError(
            "Aucun jeu de données ML disponible : lancer "
            "`python -m src.ml.donnees_synthetiques` pour le générer."
        )
    modele = entrainer_baseline(X, y)
    chemin = chemin or CHEMIN_MODELE_DEFAUT
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, chemin)
    return chemin


if __name__ == "__main__":
    chemin = entrainer_et_sauvegarder()
    print(f"Modèle entraîné et sauvegardé dans {chemin}")
