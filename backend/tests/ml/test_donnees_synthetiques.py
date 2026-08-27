"""Tests du générateur de profils synthétiques (DATA-6)."""

import random

from src.ml.archetypes import ARCHETYPES, PARCOURS_CONNUS
from src.ml.distribution_reelle import prior_lisse
from src.ml.donnees_synthetiques import (
    REGIMES_REALISTES,
    RegimeCompletude,
    generer_jeu_de_donnees,
    generer_profil,
)
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


# --- Génération calée sur l'enquête réelle (DATA-6 bis) --------------------


def test_un_prior_desequilibre_produit_des_effectifs_desequilibres():
    prior = prior_lisse({"IGGLIA": 40, "ESIIA": 10}, PARCOURS_CONNUS)
    exemples = generer_jeu_de_donnees(prior=prior, n_total=800, seed=1)

    comptage: dict[str, int] = {}
    for e in exemples:
        comptage[e["parcours_id"]] = comptage.get(e["parcours_id"], 0) + 1

    assert len(exemples) == 800
    assert comptage["IGGLIA"] > comptage["ESIIA"]
    # Aucune classe ne disparaît : c'est le rôle du plancher de `prior_lisse`.
    assert set(comptage) == set(ARCHETYPES)
    assert all(n > 0 for n in comptage.values())


def test_le_regime_restreint_les_champs_renseignes():
    regime = RegimeCompletude(
        nom="matieres_seules",
        poids=1.0,
        champs=frozenset({"matieres"}),
        notes=True,
        environnement=False,
        serie_bac=False,
    )
    profil = generer_profil("IGGLIA", random.Random(11), regime)

    assert profil.matieres_preferees
    assert profil.competences_declarees == []
    assert profil.centres_interet == []
    assert profil.preferences_professionnelles == []
    assert profil.environnement_travail_recherche is None
    assert profil.serie_bac is None


def test_le_plafond_de_traits_borne_chaque_champ():
    regime = RegimeCompletude(
        nom="borne",
        poids=1.0,
        champs=frozenset({"matieres", "competences"}),
        notes=False,
        environnement=False,
        serie_bac=False,
        plafond_traits=1,
    )
    for graine in range(20):
        profil = generer_profil("IGGLIA", random.Random(graine), regime)
        assert len(profil.matieres_preferees) <= 1
        assert len(profil.competences_declarees) <= 1


def test_le_regime_sans_notes_ne_produit_aucun_resultat_scolaire():
    regime = RegimeCompletude(
        nom="sans_notes",
        poids=1.0,
        champs=frozenset({"matieres"}),
        notes=False,
        environnement=False,
        serie_bac=False,
    )
    profil = generer_profil("IGGLIA", random.Random(3), regime)
    assert profil.resultats_scolaires == {}


def test_la_serie_bac_generee_appartient_au_vocabulaire_observe():
    from src.ml.donnees_synthetiques import SERIES_BAC

    regime = RegimeCompletude(
        nom="avec_serie",
        poids=1.0,
        champs=frozenset({"matieres"}),
        notes=False,
        environnement=False,
        serie_bac=True,
    )
    for graine in range(30):
        profil = generer_profil("GCA", random.Random(graine), regime)
        assert profil.serie_bac in SERIES_BAC


def test_la_serie_bac_n_est_pas_une_cle_deterministe_du_parcours():
    """Une série toujours admissible ferait de ce champ un raccourci vers la
    classe — la fuite déjà rencontrée sur `environnement_travail_recherche` —
    et laisserait la règle d'admission hybride (ML-10) sans cas à traiter."""
    regime = RegimeCompletude(
        nom="avec_serie",
        poids=1.0,
        champs=frozenset({"matieres"}),
        notes=False,
        environnement=False,
        serie_bac=True,
    )
    rng = random.Random(0)
    series = {generer_profil("GCA", rng, regime).serie_bac for _ in range(200)}
    # GCA n'admet que C/D/S : voir des séries hors de cet ensemble prouve que
    # la part inadmissible est bien tirée.
    assert series - {"C", "D", "S"}


def test_les_regimes_realistes_couvrent_plusieurs_formes_de_completude():
    exemples = generer_jeu_de_donnees(
        n_par_parcours=20, seed=5, regimes=REGIMES_REALISTES
    )
    noms = {e["regime"] for e in exemples}
    assert noms == {r.nom for r in REGIMES_REALISTES}


def test_le_jeu_cale_sur_l_enquete_est_plus_mince_que_le_jeu_de_reference():
    """La raison d'être des régimes : un profil réel ne renseigne pas cinq
    dimensions, le jeu d'entraînement ne doit donc pas le supposer."""
    champs = (
        "matieres_preferees",
        "competences_declarees",
        "centres_interet",
        "preferences_professionnelles",
    )

    def moyenne_champs_remplis(exemples):
        total = sum(sum(1 for c in champs if e["profil"][c]) for e in exemples)
        return total / len(exemples)

    reference = generer_jeu_de_donnees(n_par_parcours=20, seed=5)
    challenger = generer_jeu_de_donnees(n_par_parcours=20, seed=5, regimes=REGIMES_REALISTES)

    assert moyenne_champs_remplis(challenger) < moyenne_champs_remplis(reference)


def test_meme_graine_et_memes_regimes_produisent_le_meme_jeu():
    a = generer_jeu_de_donnees(n_par_parcours=5, seed=9, regimes=REGIMES_REALISTES)
    b = generer_jeu_de_donnees(n_par_parcours=5, seed=9, regimes=REGIMES_REALISTES)
    assert a == b
