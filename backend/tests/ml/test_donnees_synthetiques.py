"""Tests du générateur de profils synthétiques (DATA-6)."""

from src.ml.archetypes import ARCHETYPES
from src.ml.donnees_synthetiques import generer_jeu_de_donnees, generer_profil
from src.schemas import ProfilCandidat


def test_generer_profil_produit_un_profil_valide():
    import random

    profil = generer_profil("IGGLIA", random.Random(1))
    assert isinstance(profil, ProfilCandidat)


def test_jeu_de_donnees_couvre_tous_les_parcours_avec_le_bon_volume():
    exemples = generer_jeu_de_donnees(n_par_parcours=5, seed=1)
    assert len(exemples) == 5 * len(ARCHETYPES)
    comptage = {}
    for e in exemples:
        comptage[e["parcours_id"]] = comptage.get(e["parcours_id"], 0) + 1
    assert set(comptage) == set(ARCHETYPES)
    assert all(n == 5 for n in comptage.values())


def test_meme_graine_produit_le_meme_jeu_de_donnees():
    a = generer_jeu_de_donnees(n_par_parcours=5, seed=7)
    b = generer_jeu_de_donnees(n_par_parcours=5, seed=7)
    assert a == b


def test_graines_differentes_produisent_des_jeux_differents():
    a = generer_jeu_de_donnees(n_par_parcours=5, seed=1)
    b = generer_jeu_de_donnees(n_par_parcours=5, seed=2)
    assert a != b


def test_la_retention_moyenne_des_traits_de_l_archetype_reste_partielle():
    """Un profil qui reprendrait systématiquement tous les traits de son
    archétype serait trivialement identifiable — voir la note de biais du
    module. Le bruit croisé peut occasionnellement faire coïncider un trait
    ajouté avec un trait d'origine (plusieurs archétypes partagent des
    matières comme "informatique") : on vérifie donc une moyenne sur de
    nombreux tirages, pas chaque tirage individuellement.
    """
    import random

    rng = random.Random(3)
    archetype = ARCHETYPES["IGGLIA"]
    total_matieres = len(archetype["matieres"])
    ratios = []
    for _ in range(200):
        profil = generer_profil("IGGLIA", rng)
        conserves = set(profil.matieres_preferees) & set(archetype["matieres"])
        ratios.append(len(conserves) / total_matieres)

    assert sum(ratios) / len(ratios) < 0.9


def test_chaque_matiere_declaree_a_une_note():
    import random

    profil = generer_profil("ISAIA", random.Random(5))
    assert set(profil.resultats_scolaires) == set(profil.matieres_preferees)
    assert all(0 <= note <= 20 for note in profil.resultats_scolaires.values())
