"""Tests de la stratégie de retry sur sortie non conforme au schéma."""

import pytest
from pydantic import BaseModel, ValidationError

from src.sortie import generer_avec_retry


class SchemaTest(BaseModel):
    valeur: int


def test_reussite_du_premier_coup():
    resultat = generer_avec_retry(lambda erreur_precedente: {"valeur": 42}, SchemaTest)
    assert resultat.valeur == 42


def test_regeneration_apres_un_echec():
    appels = []

    def prompt_fn(erreur_precedente):
        appels.append(erreur_precedente)
        if erreur_precedente is None:
            return {"valeur": "pas un entier"}  # échoue la validation
        return {"valeur": 7}

    resultat = generer_avec_retry(prompt_fn, SchemaTest, max_essais=2)
    assert resultat.valeur == 7
    assert len(appels) == 2
    assert appels[0] is None
    assert isinstance(appels[1], ValidationError)


def test_echec_persistant_leve_runtimeerror():
    def toujours_invalide(erreur_precedente):
        return {"valeur": "toujours invalide"}

    with pytest.raises(RuntimeError):
        generer_avec_retry(toujours_invalide, SchemaTest, max_essais=2)
