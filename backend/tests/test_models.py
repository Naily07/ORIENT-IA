"""Tests du chargement du corpus pédagogique."""

import json

from src.models import DocumentSource, charger_corpus


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
