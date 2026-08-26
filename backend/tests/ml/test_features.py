"""Tests de la vectorisation d'un profil candidat."""

from src.ml.archetypes import VOCAB_ENVIRONNEMENTS, VOCAB_MATIERES
from src.ml.features import noms_features, vectoriser
from src.schemas import ProfilCandidat


def test_taille_du_vecteur_correspond_aux_noms_de_features():
    profil = ProfilCandidat()
    assert len(vectoriser(profil)) == len(noms_features())


def test_profil_vide_donne_un_vecteur_nul_sauf_environnement_absent():
    profil = ProfilCandidat()
    vecteur = vectoriser(profil)
    assert vecteur.sum() == 0.0


def test_matiere_declaree_active_le_bon_bit():
    matiere = VOCAB_MATIERES[0]
    profil = ProfilCandidat(matieres_preferees=[matiere])
    vecteur = vectoriser(profil)
    noms = noms_features()
    index = noms.index(f"matiere:{matiere}")
    assert vecteur[index] == 1.0
    assert vecteur.sum() == 1.0


def test_note_scolaire_est_normalisee_sur_20():
    matiere = VOCAB_MATIERES[0]
    profil = ProfilCandidat(matieres_preferees=[matiere], resultats_scolaires={matiere: 10.0})
    vecteur = vectoriser(profil)
    noms = noms_features()
    index_note = noms.index(f"note:{matiere}")
    assert vecteur[index_note] == 0.5


def test_environnement_active_un_seul_bit_one_hot():
    environnement = VOCAB_ENVIRONNEMENTS[0]
    profil = ProfilCandidat(environnement_travail_recherche=environnement)
    vecteur = vectoriser(profil)
    noms = noms_features()
    indices_env = [i for i, n in enumerate(noms) if n.startswith("environnement:")]
    actifs = [i for i in indices_env if vecteur[i] == 1.0]
    assert actifs == [noms.index(f"environnement:{environnement}")]


def test_trait_hors_vocabulaire_est_ignore_sans_erreur():
    profil = ProfilCandidat(matieres_preferees=["matiere_totalement_inconnue"])
    vecteur = vectoriser(profil)
    assert vecteur.sum() == 0.0
