"""Tests des garde-fous de sortie : biais et profilage psychologique
(SEC-3, SEC-4)."""

from src.securite import (
    detecter_criteres_discriminatoires,
    detecter_profilage_psychologique,
    verifier_sortie,
)

# --- Critères discriminatoires (SEC-3) ---------------------------------------


def test_critere_sensible_utilise_comme_justification_est_detecte():
    texte = "Ce parcours est déconseillé car c'est une femme et le métier est physique."
    resultat = detecter_criteres_discriminatoires(texte)
    assert resultat["detecte"] is True
    assert "genre" in resultat["criteres"]


def test_critere_sensible_mentionne_sans_justification_causale_n_est_pas_signale():
    """Mentionner un fait neutre n'est pas la même chose que s'en servir
    comme raison — sinon on ne pourrait plus jamais mentionner l'âge ou le
    genre d'un candidat dans un résumé, même de façon neutre."""
    texte = "Le candidat est un homme de 22 ans intéressé par l'informatique."
    resultat = detecter_criteres_discriminatoires(texte)
    assert resultat["detecte"] is False


def test_origine_utilisee_comme_justification_est_detectee():
    texte = "Recommandation écartée en raison de son origine nationale."
    resultat = detecter_criteres_discriminatoires(texte)
    assert resultat["detecte"] is True
    assert "origine" in resultat["criteres"]


def test_texte_sans_aucun_critere_sensible_n_est_pas_signale():
    texte = "Ce parcours convient bien car le profil déclare un fort intérêt pour l'informatique."
    resultat = detecter_criteres_discriminatoires(texte)
    assert resultat["detecte"] is False


def test_texte_vide_n_est_pas_signale():
    assert detecter_criteres_discriminatoires("")["detecte"] is False


# --- Profilage psychologique (SEC-4) -----------------------------------------


def test_inference_de_personnalite_est_detectee():
    texte = "D'après votre façon d'écrire, vous semblez être quelqu'un de très rigoureux."
    resultat = detecter_profilage_psychologique(texte)
    assert resultat["detecte"] is True


def test_reference_a_un_profil_psychologique_est_detectee():
    texte = "Votre profil psychologique correspond bien à ce métier."
    assert detecter_profilage_psychologique(texte)["detecte"] is True


def test_preference_declaree_explicitement_n_est_pas_signalee():
    """Le cœur de l'outil : reformuler une préférence *déclarée* par
    l'utilisateur ne doit jamais être confondu avec une inférence."""
    texte = "Vous avez déclaré un intérêt pour l'informatique et la programmation."
    assert detecter_profilage_psychologique(texte)["detecte"] is False


def test_texte_neutre_n_est_pas_signale():
    texte = "Le parcours IGGLIA correspond à votre profil déclaré."
    assert detecter_profilage_psychologique(texte)["detecte"] is False


# --- verifier_sortie (agrégation) --------------------------------------------


def test_verifier_sortie_sans_probleme_sur_plusieurs_textes():
    verdict = verifier_sortie("Résumé neutre.", "Explication neutre.", "Justification neutre.")
    assert verdict["danger"] is False


def test_verifier_sortie_detecte_un_probleme_dans_un_texte_parmi_plusieurs():
    verdict = verifier_sortie(
        "Résumé neutre.",
        "Ce choix est déconseillé car c'est une femme.",
        "Justification neutre.",
    )
    assert verdict["danger"] is True
    assert "genre" in verdict["raison"]


def test_verifier_sortie_ignore_les_textes_vides_ou_none():
    verdict = verifier_sortie("", None, "Tout va bien.")
    assert verdict["danger"] is False
