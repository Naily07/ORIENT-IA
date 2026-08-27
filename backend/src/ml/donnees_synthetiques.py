"""Génération du jeu de données synthétique (DATA-6, §5 du sujet).

Documentation exigée par le sujet pour toute donnée synthétique :

**Méthode** : pour chaque parcours, on tire `n_par_parcours` profils. Chaque
profil retient une **minorité** des traits de l'archétype du parcours
(`archetypes.ARCHETYPES` ; 30 à 60 % de chaque catégorie — matières,
compétences, centres d'intérêt, préférences professionnelles), complétée par
0 à 2 traits tirés d'un archétype *différent* par catégorie, et par un
environnement de travail recherché parfois emprunté à un autre parcours
(30 % de chances). Les notes scolaires sont simulées par un bruit gaussien
autour d'une moyenne fixe.

**Hypothèses** : les archétypes eux-mêmes (voir `archetypes.py`) sont
l'hypothèse principale — un lien plausible mais non vérifié entre traits
déclarés et parcours réel. Le tirage suppose qu'un candidat qui correspond à
un parcours n'en déclare jamais qu'une partie de ses traits caractéristiques,
mêlés à des traits sans rapport — pas un profil « pur ».

**Biais introduit et corrigé pendant le calibrage, à nommer plutôt qu'à
masquer** : la première version de ce générateur laissait
`environnement_travail_recherche` fixe et unique par archétype ; avec cette
seule fuite, n'importe quel modèle atteignait 100 % d'exactitude sur
16 classes, quel que soit le bruit ajouté ailleurs — la variable à elle seule
suffisait à identifier le parcours. Mesuré avant d'être documenté ici (voir
l'historique de calibrage dans le journal des commits), pas supposé.
Randomiser cette variable a fait tomber l'exactitude de la forêt aléatoire de
100 % à environ 86 % (voir `backend/tests/eval_results_ml.json`) — un résultat
bien plus crédible pour un jeu de données censé rester ambigu.

**Contrôle de cohérence** : chaque profil généré est revalidé par
`src.schemas.ProfilCandidat` avant d'être conservé — un profil qui ne
respecte pas le schéma (ex. type invalide) est rejeté, pas silencieusement
corrigé.

**Limite non négociable** : ce jeu ne remplace pas l'enquête réelle
(DATA-4/DATA-5/DATA-7). Un modèle entraîné et évalué uniquement dessus mesure
sa capacité à retrouver les hypothèses ci-dessus, pas sa capacité à orienter
de vrais candidats — voir ML-7.

---

## Second mode : génération calée sur l'enquête (DATA-6 bis, AUDIT-ML-2)

Le mode décrit ci-dessus fait deux hypothèses que les enquêtes réellement
collectées démentent : que les 16 parcours sont **équiprobables**, et que
chaque candidat renseigne **toutes** les dimensions de son profil. Sur 101
réponses réelles, IGGLIA concentre 38,7 % des étiquettes, deux parcours
n'apparaissent jamais, et 85 réponses sur 86 de l'enquête courte ne
renseignent qu'un seul des cinq champs multi-valeurs.

`generer_jeu_cale_sur_enquete()` produit un jeu qui réplique les deux :

- **effectifs par classe** — tirés du `prior` mesuré par
  `distribution_reelle.mesurer()`, puis **lissés** vers l'uniforme
  (`ALPHA_LISSAGE`) avec un plancher par classe. Recopier l'empirique tel quel
  laisserait quatre parcours à un ou deux profils et deux parcours à zéro : ce
  serait remplacer un biais par un pire. Le lissage est donc un **choix
  documenté**, pas une estimation ;
- **complétude** — chaque profil tire un `RegimeCompletude` (voir
  `REGIMES_REALISTES`) qui décide quels champs sont renseignés et jusqu'où.
  Les formes viennent des deux questionnaires réellement passés ; les
  **poids** entre formes sont, eux aussi, un choix assumé — l'objectif est de
  couvrir l'étendue de complétude attendue en production, pas d'imiter les
  proportions d'un échantillon de 101 réponses non aléatoire.

Ce mode ajoute aussi `serie_bac`, jamais générée jusqu'ici (constat ML-1) et
pourtant nécessaire à la règle d'admission hybride (ML-10), qui restait inerte
faute de la voir. Une part `PROBABILITE_SERIE_INADMISSIBLE` des séries est
délibérément **incompatible** avec le parcours : sans elle, la série
deviendrait une clé déterministe de la classe — exactement la fuite
`environnement_travail_recherche` corrigée plus haut — et la règle hybride
n'aurait aucun cas à rétrograder.

**Effet mesuré, chemin de production identique, seules les données
d'entraînement changeant** (79 profils réels, jamais vus à l'entraînement) :
top-1 15,2 % → 31,6 %, top-3 41,8 % → 63,3 %, MRR 0,351 → 0,488.

**Ce que ça ne corrige pas, contrairement à ce qu'on pourrait attendre** : le
nombre de profils réels jugés inexploitables par le garde-fou ML-9 ne bouge pas
(56/79 avant comme après). Ce garde-fou se prononce sur la couverture du
**vocabulaire en entrée**, pas sur ce que le modèle a vu à l'entraînement —
aucune distribution d'entraînement ne peut le faire baisser. Voir AUDIT-ML-1.

Le jeu équiprobable reste généré par défaut et **conservé intact** : il sert de
point de comparaison, et le remplacer effacerait la mesure de l'écart.
"""

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.admission import serie_satisfait_prerequis
from src.config import config
from src.ml.archetypes import ARCHETYPES, PARCOURS_CONNUS
from src.ml.distribution_reelle import effectifs_cibles
from src.schemas import ProfilCandidat

NOTE_MOYENNE_ARCHETYPE = 15.0  # /20, note moyenne simulée pour une matière de l'archétype
ECART_TYPE_NOTE = 2.5

# Bornes de calibrage retenues après mesure (voir la note de biais ci-dessus) :
# une rétention trop large des traits de l'archétype (>60 %) rend le jeu
# trivialement séparable, quel que soit le bruit ajouté par ailleurs.
RETENTION_MIN = 0.3
RETENTION_MAX = 0.6
BRUIT_CROISE_MAX_PAR_CHAMP = 2  # nb de traits d'un autre archétype ajoutés, tiré dans [0, ce max]
PROBABILITE_BRUIT_ENVIRONNEMENT = 0.3

CHAMPS_MULTI_VALEURS = (
    "matieres",
    "competences",
    "centres_interet",
    "preferences_professionnelles",
)

# Séries de baccalauréat effectivement rencontrées dans les réponses d'enquête.
SERIES_BAC: tuple[str, ...] = ("A1", "A2", "C", "D", "G", "S")

# Part des profils dont la série déclarée ne satisfait *pas* les prérequis du
# parcours. Sans elle, la série serait une clé déterministe du parcours — la
# même fuite que `environnement_travail_recherche` documentée ci-dessus — et la
# règle d'admission hybride (ML-10) n'aurait jamais rien à rétrograder.
PROBABILITE_SERIE_INADMISSIBLE = 0.15


@dataclass(frozen=True)
class RegimeCompletude:
    """Forme de remplissage d'un profil : quels champs sont renseignés, et
    jusqu'où.

    Un candidat réel ne remplit pas les cinq dimensions d'un `ProfilCandidat`.
    Les deux questionnaires réellement passés produisent deux formes très
    différentes (mesurées par `distribution_reelle.mesurer()`), et un système en
    production reçoit les deux — plus, en conversation, des profils encore plus
    minces. Générer uniquement des profils complets est ce qui rendait 56 des
    79 profils réels « inexploitables » au sens de ML-9 : le modèle n'avait
    jamais vu, à l'entraînement, un profil aussi mince que ceux qu'il reçoit.
    """

    nom: str
    poids: float
    champs: frozenset[str]
    """Sous-ensemble de `CHAMPS_MULTI_VALEURS` effectivement renseigné."""
    notes: bool
    environnement: bool
    serie_bac: bool
    plafond_traits: int | None = None
    """Nombre maximum de traits conservés par champ, `None` pour illimité."""


# Trois formes calquées sur ce qui a réellement été collecté, plus la forme
# d'origine. Les **poids** sont un choix assumé, pas une mesure : les deux
# enquêtes ne sont pas un échantillon aléatoire des futurs utilisateurs, et
# recopier leurs proportions (85 % de profils à un seul champ) priverait le
# modèle de tout signal sur les autres dimensions. L'objectif est de couvrir
# l'étendue de complétude attendue en production, pas d'imiter un échantillon.
REGIMES_REALISTES: tuple[RegimeCompletude, ...] = (
    RegimeCompletude(
        # Forme de l'enquête courte : 1 à 4 matières, une note, rien d'autre.
        nom="enquete_courte",
        poids=0.35,
        champs=frozenset({"matieres"}),
        notes=True,
        environnement=False,
        serie_bac=False,
        plafond_traits=2,
    ),
    RegimeCompletude(
        # Forme de notre questionnaire à 21 questions : matières, centres
        # d'intérêt, quelques compétences, environnement et série — mais aucune
        # note chiffrée (la question n'est pas posée).
        nom="enquete_longue",
        poids=0.30,
        champs=frozenset({"matieres", "competences", "centres_interet"}),
        notes=False,
        environnement=True,
        serie_bac=True,
        plafond_traits=3,
    ),
    RegimeCompletude(
        # Profil recueilli au fil d'une conversation : quelques traits épars,
        # pas nécessairement les mêmes champs d'un candidat à l'autre.
        nom="conversation_partielle",
        poids=0.15,
        champs=frozenset({"matieres", "centres_interet"}),
        notes=False,
        environnement=False,
        serie_bac=True,
        plafond_traits=2,
    ),
    RegimeCompletude(
        # Forme d'origine (DATA-6) : toutes les dimensions renseignées.
        nom="formulaire_complet",
        poids=0.20,
        champs=frozenset(CHAMPS_MULTI_VALEURS),
        notes=True,
        environnement=True,
        serie_bac=True,
        plafond_traits=None,
    ),
)

# Régime unique reproduisant exactement le comportement d'origine, utilisé par
# défaut pour que le jeu de référence reste bit-à-bit identique.
REGIME_ORIGINE = RegimeCompletude(
    nom="formulaire_complet",
    poids=1.0,
    champs=frozenset(CHAMPS_MULTI_VALEURS),
    notes=True,
    environnement=True,
    serie_bac=False,
    plafond_traits=None,
)


def _sous_ensemble(valeurs: list[str], rng: random.Random) -> list[str]:
    """Retient une minorité (30 à 60 %) d'une liste de traits — jamais la
    totalité, sans quoi le profil serait trivialement identifiable."""
    if not valeurs:
        return []
    k = max(1, round(len(valeurs) * rng.uniform(RETENTION_MIN, RETENTION_MAX)))
    return rng.sample(valeurs, k)


def _traits_bruit(champ: str, parcours_id: str, rng: random.Random) -> list[str]:
    """0 à `BRUIT_CROISE_MAX_PAR_CHAMP` traits plausibles tirés d'archétypes
    *différents*, pour éviter que chaque classe occupe une région disjointe
    de l'espace des traits."""
    autres = [p for p in ARCHETYPES if p != parcours_id]
    n = rng.randint(0, BRUIT_CROISE_MAX_PAR_CHAMP)
    traits = []
    for _ in range(n):
        valeurs = ARCHETYPES[rng.choice(autres)][champ]
        if valeurs:
            traits.append(rng.choice(valeurs))
    return traits


def _environnement_bruite(parcours_id: str, rng: random.Random) -> str:
    """L'environnement de l'archétype, ou (30 % du temps) celui d'un autre
    parcours tiré au hasard — voir la note de biais du module."""
    archetype = ARCHETYPES[parcours_id]
    if rng.random() < PROBABILITE_BRUIT_ENVIRONNEMENT:
        autres = [p for p in ARCHETYPES if p != parcours_id]
        return ARCHETYPES[rng.choice(autres)]["environnement"]
    return archetype["environnement"]


@lru_cache(maxsize=1)
def _series_admissibles_par_parcours() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Pour chaque parcours : `(séries admissibles, séries inadmissibles)`.

    Dérivé du corpus réel en testant chaque série candidate contre les
    descriptions de prérequis via `admission.serie_satisfait_prerequis` — la
    **même** fonction que celle utilisée en production par `verifier_prerequis`
    et `ml.hybride`. Recopier ici une table série → parcours la ferait diverger
    de la règle réelle au premier changement de corpus.

    Corpus absent ou parcours sans prérequis connu : toutes les séries sont
    considérées admissibles, ce qui revient à ne rien affirmer — la génération
    ne doit pas inventer une contrainte que le corpus ne porte pas.
    """
    from src.models import charger_corpus_formations

    try:
        corpus = charger_corpus_formations()
    except Exception:
        return {}

    descriptions_par_id = {p.id: p.description for p in corpus.prerequis}
    resultat: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for parcours in corpus.parcours:
        descriptions = [
            descriptions_par_id[pid] for pid in parcours.prerequis if pid in descriptions_par_id
        ]
        admissibles = tuple(
            s for s in SERIES_BAC if serie_satisfait_prerequis(s, descriptions) is not False
        )
        inadmissibles = tuple(s for s in SERIES_BAC if s not in admissibles)
        resultat[parcours.id] = (admissibles or SERIES_BAC, inadmissibles)
    return resultat


def _serie_bac(parcours_id: str, rng: random.Random) -> str | None:
    """Une série plausible pour ce parcours, parfois inadmissible.

    Voir `PROBABILITE_SERIE_INADMISSIBLE` : un jeu où la série serait toujours
    compatible ferait de ce champ un raccourci vers la classe, et laisserait la
    règle d'admission hybride sans aucun cas à traiter.
    """
    admissibles, inadmissibles = _series_admissibles_par_parcours().get(
        parcours_id, (SERIES_BAC, ())
    )
    if inadmissibles and rng.random() < PROBABILITE_SERIE_INADMISSIBLE:
        return rng.choice(inadmissibles)
    return rng.choice(admissibles)


def _plafonner(traits: list[str], plafond: int | None, rng: random.Random) -> list[str]:
    """Ramène une liste de traits à `plafond` éléments, tirés au hasard.

    Tronquer les premiers éléments plutôt qu'échantillonner biaiserait le jeu
    vers les traits placés en tête de l'archétype.
    """
    if plafond is None or len(traits) <= plafond:
        return traits
    return rng.sample(traits, plafond)


def generer_profil(
    parcours_id: str,
    rng: random.Random,
    regime: RegimeCompletude | None = None,
) -> ProfilCandidat:
    """Génère un profil plausible mais ambigu pour un parcours donné.

    `regime` décrit **quels champs sont renseignés** et jusqu'où (voir
    `RegimeCompletude`). Par défaut, `REGIME_ORIGINE` : tous les champs, comme
    avant l'introduction des régimes — le jeu de référence reste inchangé.
    """
    regime = regime or REGIME_ORIGINE
    archetype = ARCHETYPES[parcours_id]

    traits: dict[str, list[str]] = {
        "matieres": _sous_ensemble(archetype["matieres"], rng),
        "competences": _sous_ensemble(archetype["competences"], rng),
        "centres_interet": _sous_ensemble(archetype["centres_interet"], rng),
        "preferences_professionnelles": _sous_ensemble(
            archetype["preferences_professionnelles"], rng
        ),
    }
    for champ in CHAMPS_MULTI_VALEURS:
        for trait in _traits_bruit(champ, parcours_id, rng):
            if trait not in traits[champ]:
                traits[champ].append(trait)

    # Le tirage ci-dessus est fait pour **tous** les champs avant filtrage, afin
    # que la suite de nombres aléatoires consommée ne dépende pas du régime :
    # deux régimes différents partant de la même graine restent comparables.
    for champ in CHAMPS_MULTI_VALEURS:
        if champ not in regime.champs:
            traits[champ] = []
        else:
            traits[champ] = _plafonner(traits[champ], regime.plafond_traits, rng)

    resultats = (
        {
            matiere: round(
                max(0.0, min(20.0, rng.gauss(NOTE_MOYENNE_ARCHETYPE, ECART_TYPE_NOTE))), 1
            )
            for matiere in traits["matieres"]
        }
        if regime.notes
        else {}
    )

    environnement = _environnement_bruite(parcours_id, rng) if regime.environnement else None
    serie = _serie_bac(parcours_id, rng) if regime.serie_bac else None

    return ProfilCandidat(
        matieres_preferees=traits["matieres"],
        resultats_scolaires=resultats,
        competences_declarees=traits["competences"],
        centres_interet=traits["centres_interet"],
        preferences_professionnelles=traits["preferences_professionnelles"],
        environnement_travail_recherche=environnement,
        serie_bac=serie,
    )


def _tirer_regime(
    regimes: tuple[RegimeCompletude, ...], rng: random.Random
) -> RegimeCompletude:
    return rng.choices(regimes, weights=[r.poids for r in regimes], k=1)[0]


def generer_jeu_de_donnees(
    n_par_parcours: int = 50,
    seed: int = 42,
    prior: dict[str, float] | None = None,
    n_total: int | None = None,
    regimes: tuple[RegimeCompletude, ...] | None = None,
) -> list[dict]:
    """Génère le jeu complet.

    Deux modes :

    - **équiprobable** (par défaut) — `n_par_parcours` profils pour chacun des
      16 parcours, tous les champs renseignés. C'est le jeu de référence
      historique (DATA-6), conservé tel quel pour servir de point de comparaison ;
    - **calé sur l'enquête** — en passant `prior` (et `n_total`), les effectifs
      par classe suivent la distribution réellement observée, lissée par
      `distribution_reelle.prior_lisse`. En passant `regimes`, chaque profil
      tire en plus une forme de complétude (voir `REGIMES_REALISTES`).

    Retourne une liste de `{"profil": ..., "parcours_id": ...}`, sérialisable
    directement en JSON — c'est le format lu par `charger_jeu_de_donnees()`.
    """
    rng = random.Random(seed)

    if prior is not None:
        total = n_total if n_total is not None else n_par_parcours * len(ARCHETYPES)
        effectifs = effectifs_cibles(prior, total)
    else:
        effectifs = dict.fromkeys(ARCHETYPES, n_par_parcours)

    exemples = []
    for parcours_id in sorted(ARCHETYPES):
        for _ in range(effectifs.get(parcours_id, 0)):
            regime = _tirer_regime(regimes, rng) if regimes else None
            profil = generer_profil(parcours_id, rng, regime)
            exemple = {"profil": profil.model_dump(), "parcours_id": parcours_id}
            if regime is not None:
                # Tracé dans le jeu produit : sans ça, impossible de vérifier
                # après coup que les proportions demandées ont bien été tirées.
                exemple["regime"] = regime.nom
            exemples.append(exemple)
    return exemples


def generer_jeu_cale_sur_enquete(
    n_total: int = 800,
    seed: int = 42,
    regimes: tuple[RegimeCompletude, ...] = REGIMES_REALISTES,
) -> list[dict]:
    """Jeu synthétique dont classes et complétude imitent les enquêtes réelles.

    Raccourci qui mesure la distribution réelle puis génère — c'est le point
    d'entrée de l'amélioration « entraîner sur une distribution qui ressemble à
    l'enquête ». Si aucune enquête n'est disponible sur le disque, le prior
    retombe sur l'uniforme et seuls les régimes de complétude s'appliquent.
    """
    from src.ml.distribution_reelle import mesurer

    prior = mesurer().prior(PARCOURS_CONNUS)
    return generer_jeu_de_donnees(
        seed=seed, prior=prior, n_total=n_total, regimes=regimes
    )


def sauvegarder_jeu_de_donnees(exemples: list[dict], chemin: Path | None = None) -> Path:
    chemin = chemin or (config.dossier_data / "ml" / "profils_synthetiques.json")
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(exemples, f, ensure_ascii=False, indent=2)
    return chemin


def charger_jeu_de_donnees(chemin: Path | None = None) -> list[dict]:
    """Charge le jeu de données synthétique. Tolère un fichier absent (liste vide)."""
    chemin = chemin or (config.dossier_data / "ml" / "profils_synthetiques.json")
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument(
        "--cale-sur-enquete",
        action="store_true",
        help=(
            "Génère le jeu challenger (classes et complétude calquées sur les "
            "enquêtes réelles) dans profils_synthetiques_realistes.json, "
            "au lieu du jeu de référence équiprobable."
        ),
    )
    parseur.add_argument("--n-total", type=int, default=800)
    parseur.add_argument("--seed", type=int, default=42)
    arguments = parseur.parse_args()

    if arguments.cale_sur_enquete:
        exemples = generer_jeu_cale_sur_enquete(
            n_total=arguments.n_total, seed=arguments.seed
        )
        destination = config.dossier_data / "ml" / "profils_synthetiques_realistes.json"
    else:
        exemples = generer_jeu_de_donnees(seed=arguments.seed)
        destination = None

    chemin = sauvegarder_jeu_de_donnees(exemples, destination)
    print(f"{len(exemples)} profils générés dans {chemin}")

    if arguments.cale_sur_enquete:
        classes = Counter(e["parcours_id"] for e in exemples)
        regimes = Counter(e.get("regime") for e in exemples)
        print("\nEffectifs par parcours :")
        for parcours_id, n in classes.most_common():
            print(f"  {parcours_id:9s} {n:4d}")
        print("\nRégimes de complétude tirés :")
        for nom, n in regimes.most_common():
            print(f"  {nom:24s} {n:4d}  ({100 * n / len(exemples):4.1f} %)")
