"""Tests du chargement du corpus pédagogique."""

import json

from src.models import DocumentSource, Parcours, charger_corpus, charger_corpus_formations


def test_fichier_absent_retourne_une_liste_vide(tmp_path):
    assert charger_corpus(chemin=tmp_path / "inexistant.json") == []


def test_chargement_d_un_corpus_valide(tmp_path):
    fichier = tmp_path / "corpus.json"
    fichier.write_text(
        json.dumps(
            [
                {
                    "id": "FORM-INFO-01",
                    "titre": "Mention Informatique",
                    "categorie": "informatique",
                    "contenu": "Contenu de démonstration.",
                    "derniere_maj": "2026-01-01T00:00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    documents = charger_corpus(chemin=fichier)
    assert len(documents) == 1
    assert isinstance(documents[0], DocumentSource)
    assert documents[0].id == "FORM-INFO-01"
    assert documents[0].source_id is None


# --- Corpus structuré (Mentions, Parcours, Matières...) ----------------------


def test_corpus_formations_vide_si_aucun_fichier(monkeypatch, tmp_path):
    from src.config import config

    monkeypatch.setattr(config, "dossier_data", tmp_path)
    corpus = charger_corpus_formations()
    assert corpus.mentions == []
    assert corpus.parcours == []
    assert corpus.matieres == []


def test_corpus_formations_charge_les_fichiers_presents(monkeypatch, tmp_path):
    from src.config import config

    monkeypatch.setattr(config, "dossier_data", tmp_path)
    (tmp_path / "parcours.json").write_text(
        json.dumps(
            [
                {
                    "id": "PAR-INFO-IA",
                    "nom": "Intelligence Artificielle",
                    "mention_id": "MENTION-INFO",
                    "matieres": ["MAT-ALGO"],
                    "competences": ["COMP-ML"],
                    "prerequis": ["PREREQ-MATHS"],
                    "debouches": ["METIER-DEV-IA"],
                }
            ]
        ),
        encoding="utf-8",
    )

    corpus = charger_corpus_formations()
    assert len(corpus.parcours) == 1
    assert isinstance(corpus.parcours[0], Parcours)
    assert corpus.parcours[0].id == "PAR-INFO-IA"
    assert corpus.parcours[0].debouches == ["METIER-DEV-IA"]
