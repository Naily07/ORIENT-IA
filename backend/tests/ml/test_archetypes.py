"""Tests des archétypes et de leur cohérence avec le corpus réel."""

from src.ml.archetypes import ARCHETYPES, PARCOURS_CONNUS, VOCAB_ENVIRONNEMENTS, VOCAB_MATIERES
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
