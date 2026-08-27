"""Tests du registre de traçabilité des sources."""

import json

from src.sources import charger_registre_sources, verifier_provenance


def test_fichier_absent_retourne_une_liste_vide(monkeypatch, tmp_path):
    from src.config import config

    monkeypatch.setattr(config, "dossier_data", tmp_path)
    assert charger_registre_sources() == []


def test_chargement_d_une_entree_valide(monkeypatch, tmp_path):
    from src.config import config

    monkeypatch.setattr(config, "dossier_data", tmp_path)
    (tmp_path / "registre_sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "SRC-TEST",
                    "titre": "Page de test",
                    "url": "https://exemple.mg/test",
                    "date_consultation": "2026-08-26",
                    "statut": "officiel",
                    "donnees_extraites": ["une info"],
                    "limites": ["résumé automatique, à revérifier"],
                }
            ]
        ),
        encoding="utf-8",
    )

    registre = charger_registre_sources()
    assert len(registre) == 1
    assert registre[0].id == "SRC-TEST"
    assert registre[0].statut == "officiel"


def test_verifier_provenance_ne_signale_rien_quand_tout_est_trace():
    from src.sources import EntreeRegistreSource

    registre = [
        EntreeRegistreSource(
            id="SRC-A", titre="A", url="https://x", date_consultation="2026-08-26",
            statut="officiel",
        )
    ]
    assert verifier_provenance(["SRC-A", None, "SRC-A"], registre) == []


def test_verifier_provenance_signale_les_references_orphelines():
    from src.sources import EntreeRegistreSource

    registre = [
        EntreeRegistreSource(
            id="SRC-A", titre="A", url="https://x", date_consultation="2026-08-26",
            statut="officiel",
        )
    ]
    orphelins = verifier_provenance(["SRC-A", "SRC-INEXISTANTE", None], registre)
    assert orphelins == ["SRC-INEXISTANTE"]


# --- Cohérence du corpus réel livré avec le projet ---------------------------
# Ces tests tournent sur les vraies données de `backend/data/` (pas de
# fichiers temporaires) : ils garantissent qu'aucune donnée ajoutée au corpus
# ne référence une source absente du registre, conformément à la règle non
# négociable du §4 du sujet.


def test_le_corpus_reel_ne_contient_aucune_source_orpheline():
    from src.models import charger_corpus_formations, charger_corpus_rag

    registre = charger_registre_sources()
    # `charger_corpus_rag()` : corpus rédigé + fiches générées (DATA-3), pour
    # qu'une fiche générée référençant une source hors registre soit refusée
    # au même titre qu'une fiche rédigée à la main.
    documents = charger_corpus_rag()
    corpus = charger_corpus_formations()

    source_ids = (
        [d.source_id for d in documents]
        + [m.source_id for m in corpus.mentions]
        + [p.source_id for p in corpus.parcours]
        + [p.source_id for p in corpus.prerequis]
    )
    assert verifier_provenance(source_ids, registre) == []


def test_le_registre_reel_couvre_les_statuts_attendus():
    registre = charger_registre_sources()
    assert registre, "le registre des sources ne doit pas être vide"
    assert {e.statut for e in registre} <= {"officiel", "institutionnel", "externe"}
    assert any(e.statut == "officiel" for e in registre)
