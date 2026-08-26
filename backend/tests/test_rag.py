"""Tests du RAG.

Le découpage et les garde-fous de citation sont testés sans réseau ni index :
ce sont eux qui protègent contre l'invention d'une formation ou d'une règle
d'admission (§16 du sujet), ils doivent être vérifiables à chaque exécution.

Les tests sur index réel (recherche vectorielle) sont marqués `index` :
ils téléchargent le modèle d'embedding ONNX au premier lancement et sont donc
exclus par défaut (voir `pyproject.toml`).
"""

import pytest

from src.rag import ReponseRAG, chunker, generer_reponse_rag

# --- Découpage ---------------------------------------------------------------


def test_texte_court_reste_en_un_fragment():
    assert chunker("Une phrase courte. Une autre.", taille_max=200) == [
        "Une phrase courte. Une autre."
    ]


def test_texte_vide_ne_produit_aucun_fragment():
    assert chunker("") == []
    assert chunker("   \n  ") == []


def test_texte_long_est_decoupe():
    texte = " ".join(f"Ceci est la phrase numero {i} du document." for i in range(60))
    fragments = chunker(texte, taille_max=50, chevauchement=10)
    assert len(fragments) > 1


def test_le_decoupage_ne_coupe_pas_au_milieu_d_une_phrase():
    texte = " ".join(f"Condition numero {i} a respecter avec soin." for i in range(40))
    for fragment in chunker(texte, taille_max=30, chevauchement=5):
        assert fragment.endswith(".")


def test_les_fragments_se_chevauchent():
    phrases = [f"Phrase distincte numero {i} avec du contenu." for i in range(40)]
    fragments = chunker(" ".join(phrases), taille_max=40, chevauchement=15)
    fin_premier = fragments[0].split(".")[-2].strip()
    assert fin_premier in fragments[1]


def test_un_chevauchement_excessif_ne_fait_pas_exploser_l_index():
    """Régression connue : avec un chevauchement supérieur à la taille du
    fragment, chaque nouveau fragment repart presque du début du précédent."""
    texte = " ".join(f"Phrase numero {i} ici." for i in range(12))
    fragments = chunker(texte, taille_max=20, chevauchement=30)

    mots_source = len(texte.split())
    mots_indexes = sum(len(f.split()) for f in fragments)
    assert mots_indexes < 2 * mots_source
    assert all(len(f.split()) <= 20 for f in fragments)


def test_tout_le_contenu_est_present_dans_les_fragments():
    texte = " ".join(f"Information capitale numero {i}." for i in range(30))
    fragments = chunker(texte, taille_max=25, chevauchement=5)
    concatenation = " ".join(fragments)
    for i in range(30):
        assert f"Information capitale numero {i}." in concatenation


# --- Absence de source ---------------------------------------------------


def test_aucun_fragment_donne_une_reponse_incertaine_sans_appel_llm(monkeypatch):
    """Sans passage, interroger le LLM l'inviterait à répondre de mémoire —
    précisément le risque de formation inventée."""

    def interdit(*_args, **_kwargs):
        raise AssertionError("le LLM ne doit pas être appelé sans passage")

    monkeypatch.setattr("src.rag.llm_call", interdit)

    reponse = generer_reponse_rag("une question quelconque", [])
    assert reponse.incertain is True
    assert reponse.sources == []


# --- Garde-fou sur les citations ----------------------------------------------


@pytest.fixture
def passages():
    return [
        {"source_id": "FORM-INFO-01", "titre": "Informatique", "contenu": "...", "distance": 0.3},
        {"source_id": "FORM-INFO-02", "titre": "Réseaux", "contenu": "...", "distance": 0.4},
    ]


def _simuler_reponse(monkeypatch, **champs):
    valeurs = {"reponse": "réponse simulée", "sources": [], "incertain": False}
    valeurs.update(champs)
    modele = ReponseRAG(**valeurs)
    monkeypatch.setattr("src.rag.llm_call", lambda *a, **k: modele)


def test_sources_valides_sont_conservees(monkeypatch, passages):
    _simuler_reponse(monkeypatch, sources=["FORM-INFO-01"])
    reponse = generer_reponse_rag("quel parcours en informatique ?", passages)
    assert reponse.sources == ["FORM-INFO-01"]
    assert reponse.incertain is False


def test_source_inventee_est_retiree_et_declenche_l_incertitude(monkeypatch, passages):
    """Le modèle peut produire un identifiant plausible mais absent des
    passages. On ne se fie pas à la consigne du prompt : on vérifie."""
    _simuler_reponse(monkeypatch, sources=["FORM-INFO-01", "FORM-INVENTE-99"])
    reponse = generer_reponse_rag("quel parcours en informatique ?", passages)

    assert reponse.sources == ["FORM-INFO-01"]
    assert reponse.incertain is True
    assert "FORM-INVENTE-99" in reponse.reponse


def test_toutes_les_sources_inventees_laisse_une_reponse_sans_source(monkeypatch, passages):
    _simuler_reponse(monkeypatch, sources=["FORM-FAUX-01", "FORM-FAUX-02"])
    reponse = generer_reponse_rag("question", passages)
    assert reponse.sources == []
    assert reponse.incertain is True


# --- Recherche sur index réel -------------------------------------------------

_CORPUS_TEST = [
    {
        "id": "FORM-INFO-01",
        "titre": "Mention Informatique",
        "categorie": "informatique",
        "contenu": (
            "La mention Informatique forme aux métiers du développement logiciel "
            "et de l'intelligence artificielle. Les prérequis incluent de bonnes "
            "bases en mathématiques et en algorithmique."
        ),
        "derniere_maj": "2026-01-01T00:00:00",
    },
    {
        "id": "FORM-GESTION-01",
        "titre": "Mention Gestion",
        "categorie": "gestion",
        "contenu": (
            "La mention Gestion forme aux métiers de la comptabilité, de la "
            "finance et du management des organisations."
        ),
        "derniere_maj": "2026-01-01T00:00:00",
    },
]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    """Index isolé, reconstruit dans un dossier temporaire."""
    from src import rag
    from src.config import config
    from src.models import DocumentSource

    config.dossier_chroma = tmp_path_factory.mktemp("chroma")
    config.rag_collection = "test-corpus-pedagogique"
    rag._collection.cache_clear()
    documents = [DocumentSource.model_validate(d) for d in _CORPUS_TEST]
    rag.ingerer(documents)
    yield
    rag._collection.cache_clear()


@pytest.mark.index
def test_recherche_retrouve_le_document_attendu(index):
    from src.rag import retrieve_context

    question = "quels sont les prérequis en informatique ?"
    fragments = retrieve_context(question, categorie="informatique")
    assert "FORM-INFO-01" in {f["source_id"] for f in fragments}


@pytest.mark.index
def test_repli_sans_filtre_quand_la_categorie_est_erronee(index):
    from src.rag import retrieve_context

    fragments = retrieve_context("prérequis en informatique", categorie="gestion")
    assert "FORM-INFO-01" in {f["source_id"] for f in fragments}
