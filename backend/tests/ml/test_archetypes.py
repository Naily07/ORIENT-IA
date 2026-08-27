"""Tests des archétypes et de leur cohérence avec le corpus réel."""

from src.ml.archetypes import (
    ARCHETYPES,
    PARCOURS_CONNUS,
    VOCAB_CENTRES_INTERET,
    VOCAB_COMPETENCES,
    VOCAB_ENVIRONNEMENTS,
    VOCAB_MATIERES,
    VOCAB_PREFERENCES_PRO,
)
from src.models import charger_corpus_formations

CHAMPS_REQUIS = {
    "matieres",
    "competences",
    "centres_interet",
    "preferences_professionnelles",
    "environnement",
}


def test_chaque_archetype_a_tous_les_champs_requis():
    for parcours_id, archetype in ARCHETYPES.items():
        assert CHAMPS_REQUIS <= set(archetype), f"{parcours_id} incomplet"


def test_chaque_archetype_a_au_moins_une_matiere_et_une_competence():
    for parcours_id, archetype in ARCHETYPES.items():
        assert archetype["matieres"], f"{parcours_id} sans matière"
        assert archetype["competences"], f"{parcours_id} sans compétence"


def test_vocabulaires_derives_sont_non_vides_et_sans_doublon():
    assert len(VOCAB_MATIERES) > 0
    assert len(VOCAB_MATIERES) == len(set(VOCAB_MATIERES))
    assert len(VOCAB_ENVIRONNEMENTS) > 0


def test_les_archetypes_couvrent_les_vrais_parcours_du_corpus():
    """Les 16 parcours réels collectés en DATA-1 doivent tous avoir un
    archétype : un modèle entraîné sur un sous-ensemble ne pourrait pas
    recommander les parcours manquants."""
    corpus = charger_corpus_formations()
    ids_reels = {p.id for p in corpus.parcours}
    assert ids_reels, "le corpus réel de backend/data/ ne doit pas être vide"
    assert ids_reels <= set(PARCOURS_CONNUS)


def test_le_vocabulaire_ne_contient_aucun_critere_sensible():
    """Garantie structurelle du sujet (§16, SEC-3) : le modèle ML ne doit
    jamais pouvoir se servir du genre, de l'origine ou de l'âge d'un
    candidat, quoi que l'utilisateur déclare. La défense la plus solide est
    qu'aucune de ces dimensions n'existe dans le vocabulaire de
    vectorisation — pas seulement un filtre a posteriori sur le texte généré
    (voir `src.securite`, qui n'est qu'un filet de sécurité complémentaire).

    Comparaison par *mot entier* du vocabulaire (chaque terme y est déjà un
    token séparé par des espaces, pas du texte libre) : une simple
    sous-chaîne ferait échouer ce test sur « age » trouvé dans « voy**age** »
    — même piège que celui déjà rencontré et corrigé dans `tools.py`
    (`verifier_prerequis`, comparaison de série de baccalauréat).
    """
    termes_interdits = {
        "homme", "femme", "genre", "origine", "ethnie", "religion", "age",
        "handicap", "orientation_sexuelle",
    }
    mots_du_vocabulaire = set(
        " ".join(
            VOCAB_MATIERES + VOCAB_COMPETENCES + VOCAB_CENTRES_INTERET
            + VOCAB_PREFERENCES_PRO + VOCAB_ENVIRONNEMENTS
        )
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )
    for terme in termes_interdits:
        assert terme not in mots_du_vocabulaire, f"terme sensible trouvé : {terme}"
