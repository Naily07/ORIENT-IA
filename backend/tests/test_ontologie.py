"""Tests du schéma d'entités/relations de l'ontologie."""

from src.ontologie import ENTITES, RELATIONS, SCHEMA_RELATIONS, relation_valide, relations_depuis


def test_toutes_les_entites_du_sujet_sont_presentes():
    attendu = {
        "Etudiant", "Formation", "Mention", "Parcours", "Matiere",
        "Competence", "Prerequis", "Metier", "CentreInteret",
    }
    assert attendu <= set(ENTITES)


def test_toutes_les_relations_du_sujet_sont_presentes():
    attendu = {
        "enseigne", "developpe", "prepareA", "necessite", "possede", "prefere", "estRequisePour",
    }
    assert attendu <= set(RELATIONS)


def test_schema_relations_n_utilise_que_des_types_declares():
    """Auto-cohérence : chaque triplet du schéma référence des entités et une
    relation elles-mêmes déclarées dans TypeEntite/TypeRelation."""
    for r in SCHEMA_RELATIONS:
        assert r.source in ENTITES
        assert r.relation in RELATIONS
        assert r.cible in ENTITES


def test_relation_connue_est_validee():
    assert relation_valide("Parcours", "enseigne", "Matiere") is True
    assert relation_valide("Competence", "estRequisePour", "Metier") is True


def test_relation_inconnue_est_rejetee():
    assert relation_valide("Parcours", "enseigne", "Metier") is False
    assert relation_valide("Etudiant", "enseigne", "Matiere") is False


def test_relations_depuis_parcours_ne_retourne_que_les_relations_sources():
    relations = relations_depuis("Parcours")
    assert relations  # le Parcours est le pivot du graphe : jamais vide
    assert all(r.source == "Parcours" for r in relations)
    # Les quatre relations du §12 dont Parcours est la source, nommées plutôt
    # que comptées : un compte brut casse à chaque ajout au schéma sans rien
    # dire de ce qui compte réellement.
    assert {"enseigne", "developpe", "prepareA", "necessite"} <= {r.relation for r in relations}


def test_relations_depuis_entite_sans_relation_sortante():
    assert relations_depuis("CentreInteret") == ()
