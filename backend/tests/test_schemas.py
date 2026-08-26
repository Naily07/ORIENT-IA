"""Tests des schémas du domaine d'orientation pédagogique."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    ACTIONS,
    AnalyseProfil,
    ProfilCandidat,
    RecommandationDecision,
    RecommandationParcours,
)


def test_profil_candidat_a_des_defauts_vides():
    profil = ProfilCandidat()
    assert profil.matieres_preferees == []
    assert profil.resultats_scolaires == {}
    assert profil.informations_manquantes == []


def test_profil_candidat_accepte_des_valeurs_declarees():
    profil = ProfilCandidat(
        matieres_preferees=["mathématiques", "informatique"],
        resultats_scolaires={"mathématiques": 16.5},
        competences_declarees=["algorithmique"],
    )
    assert profil.matieres_preferees == ["mathématiques", "informatique"]
    assert profil.resultats_scolaires["mathématiques"] == 16.5


def test_score_adequation_hors_bornes_est_rejete():
    with pytest.raises(ValidationError):
        RecommandationParcours(parcours="Informatique", score_adequation=1.5, justification="x")


def test_analyse_profil_valide():
    analyse = AnalyseProfil(
        parcours_candidats=[
            RecommandationParcours(
                parcours="Informatique", score_adequation=0.82, justification="x"
            )
        ],
        confiance=0.7,
        justification="Profil orienté sciences",
    )
    assert analyse.confiance == 0.7
    assert analyse.parcours_candidats[0].parcours == "Informatique"


def test_toutes_les_actions_du_sujet_sont_couvertes():
    attendu = {
        "information", "recommandation", "demande_information",
        "escalade_conseiller", "renvoi_administration",
    }
    assert attendu == set(ACTIONS)


def test_recommandation_decision_action_hors_vocabulaire_est_rejetee():
    with pytest.raises(ValidationError):
        RecommandationDecision(
            resume="x",
            confiance=0.5,
            explication="x",
            action="filiere_inventee",
            incertitude_declaree=False,
        )


def test_recommandation_decision_valide():
    decision = RecommandationDecision(
        resume="Profil orienté sciences et informatique",
        parcours_recommandes=[
            RecommandationParcours(parcours="Informatique", score_adequation=0.9, justification="x")
        ],
        confiance=0.85,
        explication="Fort intérêt déclaré pour les mathématiques et la programmation",
        sources=["FORM-INFO-01"],
        action="recommandation",
        incertitude_declaree=False,
    )
    assert decision.action == "recommandation"
    assert decision.incertitude_declaree is False
