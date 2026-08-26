"""Évaluation des modèles de recommandation (ML-5, ML-6).

Le sujet est explicite : « une simple valeur d'accuracy ne constitue pas une
évaluation suffisante » (§7). `evaluer_modele()` calcule donc :

- l'exactitude globale, seulement comme repère, jamais seule ;
- précision/rappel/F1 par classe et macro-moyennés (matrice de confusion
  implicite) — indispensable avec 16 classes dont certaines se ressemblent ;
- le top-k accuracy (k=3) : pour un système qui propose plusieurs parcours,
  la bonne réponse dans le top 3 est le critère pertinent, pas seulement le
  rang 1 ;
- une mesure de calibration simplifiée : la confiance moyenne du modèle sur
  ses prédictions correctes doit être supérieure à celle sur ses erreurs,
  sans quoi la confiance affichée à l'utilisateur ne voudrait rien dire ;
- la stabilité : le même pipeline ré-entraîné sur plusieurs découpages
  train/test doit produire une exactitude qui ne varie pas dans tous les sens.

`analyser_erreurs()` isole les confusions les plus fréquentes — utile pour
distinguer une confusion « attendue » (deux parcours de la même mention,
voir `donnees_synthetiques.py`) d'une confusion qui signalerait un problème
du générateur ou du modèle.
"""

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def _top_k_accuracy(modele, X_test: np.ndarray, y_test: np.ndarray, k: int = 3) -> float:
    probabilites = modele.predict_proba(X_test)
    classes = modele.classes_
    top_k_indices = np.argsort(probabilites, axis=1)[:, -k:]
    top_k_classes = classes[top_k_indices]
    return float(np.mean([y in preds for y, preds in zip(y_test, top_k_classes, strict=True)]))


def _calibration(modele, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    probabilites = modele.predict_proba(X_test)
    classes = modele.classes_
    predictions = classes[np.argmax(probabilites, axis=1)]
    confiances = np.max(probabilites, axis=1)

    correctes = predictions == y_test
    confiance_correctes = float(confiances[correctes].mean()) if correctes.any() else None
    confiance_erreurs = float(confiances[~correctes].mean()) if (~correctes).any() else None
    return {
        "confiance_moyenne_predictions_correctes": confiance_correctes,
        "confiance_moyenne_predictions_erronees": confiance_erreurs,
    }


def evaluer_modele(modele, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Retourne les métriques d'évaluation d'un modèle déjà entraîné."""
    predictions = modele.predict(X_test)
    exactitude = float(np.mean(predictions == y_test))

    classes = sorted(set(y_test) | set(predictions))
    precision, rappel, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=classes, zero_division=0
    )
    par_classe = {
        classe: {
            "precision": float(p),
            "rappel": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for classe, p, r, f, s in zip(classes, precision, rappel, f1, support, strict=True)
    }

    return {
        "exactitude": exactitude,
        "precision_macro": float(precision.mean()),
        "rappel_macro": float(rappel.mean()),
        "f1_macro": float(f1.mean()),
        "par_classe": par_classe,
        "top_3_accuracy": _top_k_accuracy(modele, X_test, y_test, k=3),
        "calibration": _calibration(modele, X_test, y_test),
    }


def analyser_erreurs(modele, X_test: np.ndarray, y_test: np.ndarray, top_n: int = 5) -> list[dict]:
    """Les `top_n` confusions (vrai, prédit) les plus fréquentes."""
    predictions = modele.predict(X_test)
    confusions = Counter(
        (vrai, predit)
        for vrai, predit in zip(y_test, predictions, strict=True)
        if vrai != predit
    )
    return [
        {"vrai": vrai, "predit": predit, "occurrences": nb}
        for (vrai, predit), nb in confusions.most_common(top_n)
    ]


def mesurer_stabilite(
    entrainer_fn, X: np.ndarray, y: np.ndarray, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
) -> dict[str, float]:
    """Ré-entraîne et ré-évalue sur plusieurs découpages train/test.

    Une exactitude qui varie peu d'un `seed` à l'autre est un signe de
    stabilité (§7 du sujet, « stabilité des recommandations ») ; une variance
    élevée signale un modèle sensible au découpage, peu fiable en démo.
    """
    from src.ml.entrainement import separer_train_test

    exactitudes = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = separer_train_test(X, y, random_state=seed)
        modele = entrainer_fn(X_train, y_train, random_state=seed)
        exactitudes.append(float(np.mean(modele.predict(X_test) == y_test)))

    return {
        "exactitude_moyenne": float(np.mean(exactitudes)),
        "exactitude_ecart_type": float(np.std(exactitudes)),
        "exactitudes": exactitudes,
    }
