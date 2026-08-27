"""Résolution d'un terme déclaré librement vers le vocabulaire contrôlé (AGT/ML).

**Le problème que ce module corrige.** `features.vectoriser()` encode un profil en
multi-hot sur un vocabulaire fermé dérivé des archétypes. Avant ce module, un terme
absent du vocabulaire était **silencieusement ignoré** : un candidat déclarant
« maths », « info » ou « Python » produisait un vecteur entièrement nul, et le système
émettait malgré tout un score d'adéquation d'apparence normale. Trouvé en évaluant le
bloc ML (voir `backend/tests/eval_analyse.md`) : ce n'est pas une simple limite de
couverture, c'est un défaut visible dès qu'un humain tape un profil à la main.

**Quatre couches, de la plus précise à la plus tolérante :**

1. **Normalisation** — casse, accents, séparateurs (`« Mathématiques »` →
   `mathematiques`). Déterministe, sans coût.
2. **Appartenance directe au vocabulaire du champ** — si le terme normalisé est
   déjà dans le vocabulaire cible, il est retenu tel quel, **avant toute
   réécriture**. Cette couche existe parce que son absence provoquait un défaut
   réel : `communication` est une matière (archétype IMTICIA), et l'alias
   `communication → communication_numerique`, prévu pour les compétences, la
   remplaçait par un terme absent de `VOCAB_MATIERES` — 80 des 800 profils
   d'entraînement partaient alors au repli sémantique pour un terme parfaitement
   valide.
3. **Alias curés** — abréviations et synonymes français courants (`maths`, `info`,
   `SVT`, `python`), retenus seulement si leur cible appartient au vocabulaire du
   champ interrogé. Haute précision : c'est cette couche qui rattrape les cas que la
   couche 4 rate, mesuré (« info » ne ressort qu'à 0,386 de similarité sémantique,
   sous le seuil, alors que c'est un synonyme évident).
4. **Repli sémantique** — plus proche voisin dans le vocabulaire du champ, via le
   modèle d'embedding ONNX déjà présent pour le RAG. Chargé **paresseusement** :
   aucun coût tant que les couches 1 à 3 suffisent.

**Le seuil est mesuré, pas supposé.** À 0,50 : « maths » (0,72), « physique-chimie »
(0,72) et « anglais » (0,51) sont acceptés ; « SVT » (0,33 vers *gestion*),
« philosophie » (0,36 vers *comptabilité*) et « cuisine » (0,38 vers *langues*) sont
rejetés — ces trois-là n'ont effectivement pas d'équivalent dans le domaine, et les
mapper de force aurait fabriqué un profil faux. Un terme rejeté n'est pas perdu : il
est remonté dans `non_reconnus` pour que l'assistant puisse le signaler ou poser une
question, plutôt que de faire comme s'il n'avait jamais été déclaré.
"""

import functools
import unicodedata

import numpy as np

from src.config import config

# Seuil du repli sémantique : lu dans `config`, avec les autres seuils de décision.
# Au-dessus on mappe ; en dessous on préfère déclarer le terme non reconnu plutôt
# que de fabriquer une correspondance.
SEUIL_SIMILARITE_SEMANTIQUE = config.ml_seuil_similarite_semantique

# Abréviations et synonymes français courants → terme du vocabulaire contrôlé.
# Volontairement curé et non exhaustif : chaque entrée est un cas qu'un candidat a
# des chances réelles de taper. La couche sémantique couvre la longue traîne.
ALIAS: dict[str, str] = {
    # Matières
    "maths": "mathematiques",
    "math": "mathematiques",
    "mathematique": "mathematiques",
    "info": "informatique",
    "informatiques": "informatique",
    "svt": "biologie",
    "sciences_de_la_vie_et_de_la_terre": "biologie",
    "bio": "biologie",
    "physique_chimie": "physique",
    "anglais": "langues",
    "francais": "langues",
    "malgache": "langues",
    "espagnol": "langues",
    "allemand": "langues",
    "compta": "comptabilite",
    "eco": "economie",
    "geo": "geographie",
    "electro": "electronique",
    "meca": "mecanique",
    # Compétences
    "python": "programmation",
    "java": "programmation",
    "javascript": "programmation",
    "c++": "programmation",
    "coder": "programmation",
    "codage": "programmation",
    "developpement": "programmation",
    "algo": "algorithmique",
    "stats": "statistiques",
    "statistique": "statistiques",
    "data": "analyse_de_donnees",
    "excel": "analyse_de_donnees",
    "vente": "negociation",
    "communication": "communication_numerique",
    # Centres d'intérêt
    "ia": "technologie",
    "intelligence_artificielle": "technologie",
    "jeux_video": "logiciels",
    "gaming": "logiciels",
    "informatique_materielle": "materiel_informatique",
    "ecologie": "nature",
    "environnement": "nature",
    "medecine": "sante",
    "tourisme": "voyage",
    "business": "entrepreneuriat",
}


def normaliser(terme: str) -> str:
    """Réduit un terme à sa forme canonique : minuscules, sans accents, séparateurs
    unifiés en `_`. `« Physique-Chimie »` → `physique_chimie`."""
    if not terme:
        return ""
    decompose = unicodedata.normalize("NFKD", terme.strip().casefold())
    sans_accent = "".join(c for c in decompose if not unicodedata.combining(c))
    # Tout ce qui n'est ni alphanumérique ni `+` (pour « c++ ») devient un séparateur.
    caracteres = [c if (c.isalnum() or c == "+") else " " for c in sans_accent]
    return "_".join("".join(caracteres).split())


@functools.lru_cache(maxsize=1)
def _fonction_embedding():
    """Modèle d'embedding, chargé paresseusement.

    Même modèle que le RAG (`rag.py`) : aucune dépendance supplémentaire, et le
    coût de chargement (~1,3 s) n'est payé que si un terme échappe aux couches
    déterministes.
    """
    from chromadb.utils import embedding_functions

    return embedding_functions.ONNXMiniLM_L6_V2()


def _normaliser_vecteurs(matrice: np.ndarray) -> np.ndarray:
    normes = np.linalg.norm(matrice, axis=1, keepdims=True)
    # Un vecteur nul ne doit pas produire de division par zéro : sa similarité sera
    # nulle, donc sous le seuil, donc rejeté — le comportement voulu.
    normes[normes == 0] = 1.0
    return matrice / normes


@functools.lru_cache(maxsize=8)
def _embeddings_vocabulaire(vocabulaire: tuple[str, ...]) -> np.ndarray:
    """Embeddings du vocabulaire d'un champ, calculés une fois par vocabulaire.

    Les `_` sont remplacés par des espaces : le modèle est entraîné sur du langage
    naturel, `analyse_de_donnees` s'encode mieux en « analyse de donnees ».
    """
    textes = [terme.replace("_", " ") for terme in vocabulaire]
    return _normaliser_vecteurs(np.array(_fonction_embedding()(textes), dtype=float))


def _resoudre_semantiquement(
    termes: list[str], vocabulaire: tuple[str, ...], seuil: float
) -> dict[str, str]:
    """Associe chaque terme au plus proche voisin du vocabulaire, au-dessus du seuil.

    Un seul appel au modèle pour tout le lot : embarquer les termes un par un
    multiplierait la latence sans rien apporter.
    """
    if not termes:
        return {}

    requetes = _normaliser_vecteurs(
        np.array(_fonction_embedding()([t.replace("_", " ") for t in termes]), dtype=float)
    )
    similarites = requetes @ _embeddings_vocabulaire(vocabulaire).T

    correspondances = {}
    for i, terme in enumerate(termes):
        meilleur = int(np.argmax(similarites[i]))
        if similarites[i][meilleur] >= seuil:
            correspondances[terme] = vocabulaire[meilleur]
    return correspondances


def correspondances(
    termes: list[str],
    vocabulaire: tuple[str, ...],
    seuil: float = SEUIL_SIMILARITE_SEMANTIQUE,
    avec_semantique: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """Associe chaque terme déclaré au terme du vocabulaire qui lui correspond.

    Retourne `(trouvees, non_reconnus)` où les **clés de `trouvees` sont les termes
    tels qu'ils ont été déclarés** — c'est ce qui permet à un appelant de rattacher
    une valeur au terme d'origine (`features._notes` doit savoir que la note saisie
    sous « maths » appartient à `mathematiques`).

    Les termes inconnus sont résolus en **un seul lot** : un appel au modèle
    d'embedding par terme multiplierait la latence sans rien apporter.
    """
    ensemble_vocabulaire = set(vocabulaire)
    trouvees: dict[str, str] = {}
    # normalisé -> terme d'origine, pour pouvoir remonter le terme tel que tapé.
    a_tenter: dict[str, str] = {}

    for terme in termes:
        normalise = normaliser(terme)
        if not normalise:
            continue

        # **Le vocabulaire du champ prime sur les alias.** Un terme qui figure déjà
        # dans le vocabulaire cible ne doit jamais être réécrit : `communication`
        # est une matière réelle (archétype IMTICIA), que l'alias
        # `communication → communication_numerique` — prévu pour les compétences —
        # envoyait vers un terme absent de VOCAB_MATIERES. La couche « haute
        # précision » détruisait alors un terme valide, à charge pour le repli
        # sémantique de le rattraper, ce qu'il ne garantit pas.
        if normalise in ensemble_vocabulaire:
            trouvees[terme] = normalise
            continue

        alias = ALIAS.get(normalise)
        if alias is not None and alias in ensemble_vocabulaire:
            trouvees[terme] = alias
            continue

        a_tenter[normalise] = terme

    non_reconnus: list[str] = []
    if a_tenter:
        resolues = (
            _resoudre_semantiquement(list(a_tenter), vocabulaire, seuil)
            if avec_semantique
            else {}
        )
        for normalise, origine in a_tenter.items():
            correspondance = resolues.get(normalise)
            if correspondance is not None:
                trouvees[origine] = correspondance
            else:
                non_reconnus.append(origine)

    return trouvees, non_reconnus


def resoudre(
    termes: list[str],
    vocabulaire: tuple[str, ...],
    seuil: float = SEUIL_SIMILARITE_SEMANTIQUE,
    avec_semantique: bool = True,
) -> tuple[list[str], list[str]]:
    """Résout des termes déclarés librement vers le vocabulaire d'un champ.

    Retourne `(reconnus, non_reconnus)` : `reconnus` ne contient que des termes du
    vocabulaire (dédoublonnés, ordre d'apparition préservé), `non_reconnus` garde les
    termes d'origine tels que déclarés, pour pouvoir les citer à l'utilisateur.

    `avec_semantique=False` désactive la couche 3 — utile pour tester les couches
    déterministes sans charger le modèle d'embedding.
    """
    trouvees, non_reconnus = correspondances(termes, vocabulaire, seuil, avec_semantique)

    reconnus: list[str] = []
    for terme in termes:
        correspondance = trouvees.get(terme)
        if correspondance is not None and correspondance not in reconnus:
            reconnus.append(correspondance)
    return reconnus, non_reconnus
