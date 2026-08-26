"""Résolution d'un terme déclaré librement vers le vocabulaire contrôlé (AGT/ML).

**Le problème que ce module corrige.** `features.vectoriser()` encode un profil en
multi-hot sur un vocabulaire fermé dérivé des archétypes. Avant ce module, un terme
absent du vocabulaire était **silencieusement ignoré** : un candidat déclarant
« maths », « info » ou « Python » produisait un vecteur entièrement nul, et le système
émettait malgré tout un score d'adéquation d'apparence normale. Trouvé en évaluant le
bloc ML (voir `backend/tests/eval_analyse.md`) : ce n'est pas une simple limite de
couverture, c'est un défaut visible dès qu'un humain tape un profil à la main.

**Trois couches, de la plus précise à la plus tolérante :**

1. **Normalisation** — casse, accents, séparateurs (`« Mathématiques »` →
   `mathematiques`). Déterministe, sans coût.
2. **Alias curés** — abréviations et synonymes français courants (`maths`, `info`,
   `SVT`, `python`). Haute précision : c'est cette couche qui rattrape les cas que la
   couche 3 rate, mesuré (« info » ne ressort qu'à 0,386 de similarité sémantique,
   sous le seuil, alors que c'est un synonyme évident).
3. **Repli sémantique** — plus proche voisin dans le vocabulaire du champ, via le
   modèle d'embedding ONNX déjà présent pour le RAG. Chargé **paresseusement** :
   aucun coût tant que les couches 1 et 2 suffisent.

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

# Seuil de similarité cosinus du repli sémantique. Calibré à la main sur des termes
# réalistes (voir le docstring du module) : au-dessus, on mappe ; en dessous, on
# préfère déclarer le terme non reconnu plutôt que de fabriquer une correspondance.
SEUIL_SIMILARITE_SEMANTIQUE = 0.50

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
    reconnus: list[str] = []
    non_reconnus: list[str] = []
    a_tenter_semantiquement: list[str] = []
    # Le terme d'origine est conservé pour pouvoir le remonter tel qu'il a été tapé.
    origine: dict[str, str] = {}

    ensemble_vocabulaire = set(vocabulaire)

    for terme in termes:
        normalise = normaliser(terme)
        if not normalise:
            continue

        candidat = ALIAS.get(normalise, normalise)
        if candidat in ensemble_vocabulaire:
            if candidat not in reconnus:
                reconnus.append(candidat)
        else:
            a_tenter_semantiquement.append(normalise)
            origine[normalise] = terme

    if a_tenter_semantiquement:
        if avec_semantique:
            correspondances = _resoudre_semantiquement(
                a_tenter_semantiquement, vocabulaire, seuil
            )
        else:
            correspondances = {}
        for normalise in a_tenter_semantiquement:
            correspondance = correspondances.get(normalise)
            if correspondance is not None:
                if correspondance not in reconnus:
                    reconnus.append(correspondance)
            else:
                non_reconnus.append(origine[normalise])

    return reconnus, non_reconnus
