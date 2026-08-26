"""Tests de `features.analyser_couverture()` et du refus d'affirmer sur un profil
non exploité (`outils.analyser_profil`).

Régression majeure couverte ici : avant l'ouverture du vocabulaire, un profil
réaliste (« maths », « Python ») produisait un vecteur entièrement nul et le
système émettait quand même un score d'adéquation d'apparence normale.
"""

from src.ml.features import analyser_couverture, vectoriser
from src.ml.outils import analyser_profil
from src.schemas import ProfilCandidat


def test_profil_en_langage_naturel_est_desormais_exploite():
    """Le cas exact qui était cassé : un candidat qui écrit comme un humain."""
    profil = ProfilCandidat(
        matieres_preferees=["maths", "info"],
        competences_declarees=["Python"],
    )
    couverture = analyser_couverture(profil)

    assert "mathematiques" in couverture.matieres
    assert "informatique" in couverture.matieres
    assert "programmation" in couverture.competences
    assert couverture.exploitable is True
    assert vectoriser(profil).sum() > 0


def test_profil_vide_reste_non_exploitable():
    couverture = analyser_couverture(ProfilCandidat())
    assert couverture.nb_traits_reconnus == 0
    assert couverture.exploitable is False


def test_un_seul_trait_ne_suffit_pas():
    """Un trait sur 156 dimensions ne distingue pas 16 parcours : le modèle
    retomberait sur la distribution a priori."""
    couverture = analyser_couverture(ProfilCandidat(matieres_preferees=["maths"]))
    assert couverture.nb_traits_reconnus == 1
    assert couverture.exploitable is False


def test_notes_declarees_avec_un_alias_sont_prises_en_compte():
    """« maths: 17 » doit alimenter la note de `mathematiques`, pas être ignorée."""
    avec_alias = vectoriser(
        ProfilCandidat(matieres_preferees=["maths"], resultats_scolaires={"maths": 17.0})
    )
    avec_terme_exact = vectoriser(
        ProfilCandidat(
            matieres_preferees=["mathematiques"], resultats_scolaires={"mathematiques": 17.0}
        )
    )
    assert (avec_alias == avec_terme_exact).all()


def test_note_hors_bornes_est_plafonnee():
    """Une note aberrante ne doit pas produire une feature hors de [0, 1] et
    dominer toutes les autres."""
    vecteur = vectoriser(
        ProfilCandidat(matieres_preferees=["maths"], resultats_scolaires={"maths": 200.0})
    )
    assert vecteur.max() <= 1.0


# --- Refus d'affirmer sur un profil non exploité ------------------------------


def test_profil_inexploitable_donne_une_confiance_nulle():
    """Le cœur du correctif : plutôt qu'un classement d'apparence normale, le
    modèle déclare explicitement qu'il n'a rien pu exploiter. La confiance à 0
    déclenche l'escalade en aval via le seuil déjà en place."""
    analyse = analyser_profil(ProfilCandidat())
    assert analyse.confiance == 0.0
    assert "trop peu renseigné" in analyse.justification


def test_profil_exploitable_conserve_une_confiance_reelle():
    analyse = analyser_profil(
        ProfilCandidat(
            matieres_preferees=["maths", "info"],
            competences_declarees=["Python"],
        )
    )
    assert analyse.confiance > 0.0
    assert len(analyse.parcours_candidats) == 16


def test_termes_non_reconnus_sont_signales_dans_la_justification():
    """Un terme ignoré doit être dit, pas disparaître : c'est ce qui permet à
    l'assistant de poser une question plutôt que de faire comme si de rien
    n'était."""
    analyse = analyser_profil(
        ProfilCandidat(matieres_preferees=["philosophie"], centres_interet=["cuisine"])
    )
    assert analyse.confiance == 0.0
    assert "philosophie" in analyse.justification
    assert "cuisine" in analyse.justification
