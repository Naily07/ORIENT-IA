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
"""

import json
import random
from pathlib import Path

from src.config import config
from src.ml.archetypes import ARCHETYPES
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


def generer_profil(parcours_id: str, rng: random.Random) -> ProfilCandidat:
    """Génère un profil plausible mais ambigu pour un parcours donné."""
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

    resultats = {
        matiere: round(max(0.0, min(20.0, rng.gauss(NOTE_MOYENNE_ARCHETYPE, ECART_TYPE_NOTE))), 1)
        for matiere in traits["matieres"]
    }

    return ProfilCandidat(
        matieres_preferees=traits["matieres"],
        resultats_scolaires=resultats,
        competences_declarees=traits["competences"],
        centres_interet=traits["centres_interet"],
        preferences_professionnelles=traits["preferences_professionnelles"],
        environnement_travail_recherche=_environnement_bruite(parcours_id, rng),
    )


def generer_jeu_de_donnees(n_par_parcours: int = 50, seed: int = 42) -> list[dict]:
    """Génère le jeu complet : `n_par_parcours` profils par parcours connu.

    Retourne une liste de `{"profil": ..., "parcours_id": ...}`, sérialisable
    directement en JSON — c'est le format lu par `charger_jeu_de_donnees()`.
    """
    rng = random.Random(seed)
    exemples = []
    for parcours_id in sorted(ARCHETYPES):
        for _ in range(n_par_parcours):
            profil = generer_profil(parcours_id, rng)
            exemples.append({"profil": profil.model_dump(), "parcours_id": parcours_id})
    return exemples


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
    exemples = generer_jeu_de_donnees()
    chemin = sauvegarder_jeu_de_donnees(exemples)
    print(f"{len(exemples)} profils générés dans {chemin}")
