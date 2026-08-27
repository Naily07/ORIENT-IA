"""Évaluation des modèles de recommandation (ML-5, ML-6).

Le sujet est explicite : « une simple valeur d'accuracy ne constitue pas une
évaluation suffisante » (§7). `evaluer_modele()` calcule donc :

- l'exactitude globale, seulement comme repère, jamais seule ;
- précision/rappel/F1 par classe et macro-moyennés — indispensable avec 16
  classes dont certaines se ressemblent — et la **matrice de confusion**
  complète (`matrice_confusion()`) ;
- des métriques de **classement**, parce que le système propose des parcours
  ordonnés et non une réponse unique : top-3 accuracy, **MRR**, **NDCG@3**,
  rang médian de la bonne classe, et **PR-AUC** macro (préférée à la ROC-AUC :
  chaque problème un-contre-tous est déséquilibré à 1 contre 15) ;
- une **calibration** au sens propre : séparation de confiance, **ECE** par
  tranche et **score de Brier** multiclasse. La seule séparation de confiance
  dit si la confiance discrimine, pas si le « 90 % » affiché à un candidat
  correspond à 90 % de réussite réelle ;
- deux notions de **stabilité**, distinctes et toutes deux utiles :
  `mesurer_stabilite()` (variance de l'exactitude entre découpages
  train/test — une propriété de l'entraînement) et
  `mesurer_stabilite_des_recommandations()` (la recommandation résiste-t-elle
  au retrait d'un trait déclaré — la propriété qui compte pour un candidat,
  et celle que le §7 nomme « stabilité des recommandations »).

`analyser_erreurs()` isole les confusions les plus fréquentes — utile pour
distinguer une confusion « attendue » (deux parcours de la même mention,
voir `donnees_synthetiques.py`) d'une confusion qui signalerait un problème
du générateur ou du modèle.

**`evaluer_chemin_de_production()` mesure ce que l'assistant appelle
réellement** (`ml.outils.analyser_profil`), résolution de vocabulaire, garde-fou
d'exploitabilité et règles d'admission comprises — et non le seul estimateur
scikit-learn, qui n'est qu'un maillon du chemin servi (§8, §14).
"""

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize


def _top_k_accuracy(modele, X_test: np.ndarray, y_test: np.ndarray, k: int = 3) -> float:
    probabilites = modele.predict_proba(X_test)
    classes = modele.classes_
    top_k_indices = np.argsort(probabilites, axis=1)[:, -k:]
    top_k_classes = classes[top_k_indices]
    return float(np.mean([y in preds for y, preds in zip(y_test, top_k_classes, strict=True)]))


def _rangs(modele, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Rang (à partir de 1) de la bonne classe dans le classement produit.

    Base commune du MRR et du NDCG : dans ce problème, chaque profil n'a
    **qu'une seule** classe pertinente, ce qui simplifie les deux mesures — le
    gain est binaire et le classement idéal place la bonne classe au rang 1.
    """
    probabilites = modele.predict_proba(X_test)
    classes = list(modele.classes_)
    indices_vrais = np.array([classes.index(y) for y in y_test])
    # Nombre de classes strictement mieux notées que la bonne, +1.
    probas_vraies = probabilites[np.arange(len(y_test)), indices_vrais]
    return (probabilites > probas_vraies[:, None]).sum(axis=1) + 1


def _mrr(rangs: np.ndarray) -> float:
    """Mean Reciprocal Rank (§7 du sujet).

    Mesure ce que le top-k accuracy ignore : *à quelle hauteur* la bonne
    réponse est classée. Deux modèles peuvent avoir le même top-3 en plaçant
    l'un la bonne réponse au rang 1, l'autre au rang 3.
    """
    return float(np.mean(1.0 / rangs))


def _ndcg(rangs: np.ndarray, k: int = 3) -> float:
    """NDCG@k (§7 du sujet), pour une unique classe pertinente par profil.

    Gain binaire : DCG = 1/log2(rang+1) si la bonne classe est dans le top-k,
    0 sinon. Le DCG idéal vaut 1 (bonne classe au rang 1), donc le NDCG se
    réduit au DCG moyen.
    """
    gains = np.where(rangs <= k, 1.0 / np.log2(rangs + 1), 0.0)
    return float(np.mean(gains))


def _pr_auc_macro(modele, X_test: np.ndarray, y_test: np.ndarray) -> float | None:
    """PR-AUC macro-moyennée, en un-contre-tous (§7 du sujet).

    PR-AUC plutôt que ROC-AUC : avec 16 classes, chaque problème
    un-contre-tous est fortement déséquilibré (1 classe positive contre 15),
    cas où la ROC-AUC reste optimiste là où la PR-AUC reste informative.

    `None` si une seule classe est présente dans le test — la mesure n'a alors
    pas de sens et vaut mieux qu'un 0 trompeur.
    """
    classes = list(modele.classes_)
    if len(set(y_test)) < 2:
        return None
    y_binaire = label_binarize(y_test, classes=classes)
    probabilites = modele.predict_proba(X_test)
    return float(average_precision_score(y_binaire, probabilites, average="macro"))


def _calibration(modele, X_test: np.ndarray, y_test: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration de la confiance affichée (§7 du sujet).

    Trois mesures complémentaires, parce que la première seule ne suffit pas :

    - la **séparation** de confiance (correctes vs erronées) dit si la
      confiance discrimine, pas si elle est juste ;
    - l'**ECE** (Expected Calibration Error) mesure l'écart moyen entre
      confiance annoncée et exactitude réelle, par tranche de confiance :
      c'est la mesure de calibration au sens propre. Un modèle qui annonce
      90 % et a raison 60 % du temps est discriminant mais mal calibré, et
      c'est exactement ce qui rendrait un score d'adéquation trompeur pour un
      candidat ;
    - le **score de Brier** multiclasse, erreur quadratique sur le vecteur de
      probabilités complet, qui pénalise à la fois l'erreur et la sur-confiance.
    """
    probabilites = modele.predict_proba(X_test)
    classes = list(modele.classes_)
    predictions = np.array(classes)[np.argmax(probabilites, axis=1)]
    confiances = np.max(probabilites, axis=1)

    correctes = predictions == y_test
    confiance_correctes = float(confiances[correctes].mean()) if correctes.any() else None
    confiance_erreurs = float(confiances[~correctes].mean()) if (~correctes).any() else None

    # ECE : moyenne pondérée de |confiance moyenne − exactitude| par tranche.
    bornes = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    tranches = []
    for debut, fin in zip(bornes[:-1], bornes[1:], strict=True):
        dans_tranche = (confiances > debut) & (confiances <= fin)
        effectif = int(dans_tranche.sum())
        if effectif == 0:
            continue
        confiance_moyenne = float(confiances[dans_tranche].mean())
        exactitude_tranche = float(correctes[dans_tranche].mean())
        ece += (effectif / len(y_test)) * abs(confiance_moyenne - exactitude_tranche)
        tranches.append(
            {
                "intervalle": [round(float(debut), 2), round(float(fin), 2)],
                "effectif": effectif,
                "confiance_moyenne": confiance_moyenne,
                "exactitude": exactitude_tranche,
            }
        )

    y_binaire = label_binarize(y_test, classes=classes)
    if y_binaire.shape[1] == 1:  # cas dégénéré : une seule classe
        brier = None
    else:
        brier = float(np.mean(np.sum((probabilites - y_binaire) ** 2, axis=1)))

    return {
        "confiance_moyenne_predictions_correctes": confiance_correctes,
        "confiance_moyenne_predictions_erronees": confiance_erreurs,
        "ece": float(ece),
        # **Écart signé**, indispensable pour ne pas se tromper de diagnostic :
        # l'ECE est une valeur absolue et ne dit pas dans quel sens le modèle
        # se trompe. Positif = sur-confiance (annonce plus que sa réussite
        # réelle, le cas dangereux) ; négatif = sous-confiance. Une première
        # version de l'alerte affirmait « sur-confiance » sans avoir mesuré le
        # sens, alors que ce modèle était franchement sous-confiant.
        "ecart_signe_confiance_moins_exactitude": float(confiances.mean() - correctes.mean()),
        "score_de_brier": brier,
        "tranches": tranches,
    }


def matrice_confusion(modele, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Matrice de confusion complète (§7 du sujet), sérialisable en JSON.

    `analyser_erreurs()` n'en donnait que les cinq cases les plus peuplées ;
    le sujet demande la matrice, qui seule montre aussi les confusions rares et
    les classes jamais prédites.
    """
    predictions = modele.predict(X_test)
    labels = sorted(set(y_test) | set(predictions))
    matrice = confusion_matrix(y_test, predictions, labels=labels)
    return {
        "labels": labels,
        "matrice": matrice.tolist(),
        # Forme lisible sans recroiser les indices à la main.
        "par_vraie_classe": {
            vraie: {predite: int(n) for predite, n in zip(labels, ligne, strict=True) if n}
            for vraie, ligne in zip(labels, matrice, strict=True)
        },
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

    rangs = _rangs(modele, X_test, y_test)

    return {
        "exactitude": exactitude,
        "precision_macro": float(precision.mean()),
        "rappel_macro": float(rappel.mean()),
        "f1_macro": float(f1.mean()),
        "par_classe": par_classe,
        "top_3_accuracy": _top_k_accuracy(modele, X_test, y_test, k=3),
        # Métriques de **classement** : le système propose plusieurs parcours
        # ordonnés, le sujet mesure « la qualité du classement ou de la
        # recommandation » (§14) — l'exactitude au rang 1 n'en dit rien.
        "mrr": _mrr(rangs),
        "ndcg_3": _ndcg(rangs, k=3),
        "rang_median_bonne_classe": float(np.median(rangs)),
        "pr_auc_macro": _pr_auc_macro(modele, X_test, y_test),
        "calibration": _calibration(modele, X_test, y_test),
        "matrice_confusion": matrice_confusion(modele, X_test, y_test),
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


# --- Évaluation du chemin réellement servi (§8, §14) --------------------------


def _rang_dans(candidats: list, parcours_attendu: str) -> int | None:
    for rang, candidat in enumerate(candidats, start=1):
        if candidat.parcours == parcours_attendu:
            return rang
    return None


def evaluer_chemin_de_production(exemples: list[dict], analyser_fn) -> dict[str, Any]:
    """Mesure `analyser_profil()` — ce que l'assistant appelle réellement.

    `evaluer_modele()` note un estimateur scikit-learn sur des vecteurs. Or ce
    n'est pas ce que le système sert : entre les deux s'intercalent la
    résolution du vocabulaire ouvert (`ml.vocabulaire`), le garde-fou
    d'exploitabilité et les règles d'admission du volet hybride
    (`ml.hybride`), qui **réordonnent** le classement. Mesurer uniquement
    l'estimateur reviendrait à publier les chiffres d'un modèle que personne
    n'exécute — exactement ce que le §8 interdit (« le modèle ne devra pas
    rester isolé ») et ce que le §14 vérifie (« cohérence entre le modèle ML et
    la réponse finale »).

    `analyser_fn` est injecté plutôt qu'importé : `ml.outils` entraîne son
    modèle au premier appel, et l'évaluation doit pouvoir lui substituer un
    modèle entraîné sur le seul jeu d'entraînement pour ne pas mesurer sur des
    profils déjà vus.
    """
    rangs: list[int] = []
    non_classes = 0
    inexploitables = 0
    for exemple in exemples:
        analyse = analyser_fn(exemple["profil"])
        if not analyse.profil_exploitable:
            inexploitables += 1
        rang = _rang_dans(analyse.parcours_candidats, exemple["parcours_id"])
        if rang is None:
            non_classes += 1
        else:
            rangs.append(rang)

    if not rangs:
        return {"effectif": len(exemples), "avertissement": "aucun profil classable"}

    tableau = np.array(rangs)
    return {
        "effectif": len(exemples),
        "top_1": float(np.mean(tableau == 1)),
        "top_3": float(np.mean(tableau <= 3)),
        "mrr": _mrr(tableau),
        "ndcg_3": _ndcg(tableau, k=3),
        "rang_median_bonne_classe": float(np.median(tableau)),
        "profils_juges_inexploitables": inexploitables,
        "parcours_absents_du_classement": non_classes,
    }


def mesurer_stabilite_des_recommandations(
    exemples: list[dict], analyser_fn, graine: int = 42
) -> dict[str, Any]:
    """« Stabilité des recommandations » au sens du §7 du sujet.

    À distinguer de `mesurer_stabilite()`, qui mesure la variance de
    l'exactitude entre découpages train/test — une propriété de
    l'*entraînement*. Ici on mesure la propriété qui compte pour un candidat :
    **une petite variation de ce qu'il déclare change-t-elle la
    recommandation ?** Un assistant dont le premier parcours bascule parce que
    l'utilisateur a retiré un centre d'intérêt sur cinq n'est pas utilisable,
    quelle que soit son exactitude.

    Perturbation appliquée : retrait d'un seul trait déclaré, tiré au hasard
    parmi les champs multi-valués non vides.
    """
    from src.ml.outils import selectionner_significatifs

    rng = np.random.default_rng(graine)
    champs = (
        "matieres_preferees",
        "competences_declarees",
        "centres_interet",
        "preferences_professionnelles",
    )

    identiques_top1 = 0
    identiques_top3 = 0
    identiques_selection = 0
    comparables = 0

    for exemple in exemples:
        profil = exemple["profil"]
        disponibles = [c for c in champs if len(getattr(profil, c)) > 1]
        if not disponibles:
            continue

        champ = disponibles[int(rng.integers(len(disponibles)))]
        valeurs = list(getattr(profil, champ))
        valeurs.pop(int(rng.integers(len(valeurs))))
        perturbe = profil.model_copy(update={champ: valeurs})

        avant = analyser_fn(profil).parcours_candidats
        apres = analyser_fn(perturbe).parcours_candidats
        if not avant or not apres:
            continue

        comparables += 1
        identiques_top1 += int(avant[0].parcours == apres[0].parcours)
        # Stabilité de ce qui est **réellement présenté**, et non d'un top-3
        # constant : au-delà du leader, les scores tombent sous 2 % et se
        # séparent de fractions de point. Mesurer la permanence de ce bruit
        # renseignait sur le bruit, pas sur la recommandation.
        identiques_selection += int(
            {c.parcours for c in selectionner_significatifs(avant)}
            == {c.parcours for c in selectionner_significatifs(apres)}
        )
        identiques_top3 += int(
            {c.parcours for c in avant[:3]} == {c.parcours for c in apres[:3]}
        )

    if not comparables:
        return {"avertissement": "aucun profil perturbable"}

    return {
        "profils_compares": comparables,
        "perturbation": "retrait d'un trait déclaré au hasard",
        "top_1_inchange": identiques_top1 / comparables,
        # Métrique de référence : la stabilité de la liste effectivement
        # proposée au candidat (`outils.selectionner_significatifs`).
        "selection_presentee_inchangee": identiques_selection / comparables,
        # Conservée pour comparaison : ce que mesurait la version précédente,
        # un top-3 constant dont les rangs 2 et 3 sont du bruit.
        "top_3_fixe_inchange": identiques_top3 / comparables,
    }
