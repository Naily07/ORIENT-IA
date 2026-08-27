"""Tests de la complétion de profil depuis le chat (`src.extraction_profil`).

L'appel LLM est simulé : ces tests portent sur la fusion déterministe et sur la
frontière « déclaration explicite » vs « profilage » (§16, SEC-4), pas sur la
qualité d'extraction du modèle.
"""

import pytest

from src.extraction_profil import (
    extraire_profil_declare,
    fusionner_profils,
)
from src.llm_client import LLMError
from src.schemas import (
    NoteMatiereDeclaree,
    ProfilCandidat,
    ProfilDeclareExtrait,
    TourConversation,
)

# --- Fusion déterministe ----------------------------------------------------


def test_fusion_ajoute_sans_ecraser():
    base = ProfilCandidat(matieres_preferees=["maths"], serie_bac="D")
    extrait = ProfilDeclareExtrait(
        matieres_preferees=["physique"],
        competences_declarees=["python"],
        serie_bac="C",  # ignoré : la série de l'appelant fait foi
    )

    profil, completes = fusionner_profils(base, extrait)

    assert profil.matieres_preferees == ["maths", "physique"]
    assert profil.competences_declarees == ["python"]
    assert profil.serie_bac == "D"
    assert "matières préférées" in completes
    assert "compétences" in completes
    assert "série du baccalauréat" not in completes


def test_fusion_dedoublonne_a_la_casse_et_aux_espaces_pres():
    base = ProfilCandidat(matieres_preferees=["Maths"])
    extrait = ProfilDeclareExtrait(matieres_preferees=["  maths ", "Informatique"])

    profil, completes = fusionner_profils(base, extrait)

    assert profil.matieres_preferees == ["Maths", "Informatique"]
    assert completes == ["matières préférées"]


def test_fusion_normalise_la_serie_en_forme_courte():
    """« bac D » extrait du message est stocké « D » : sinon il ne correspond à
    aucun prérequis (« série C, D, S ») et fausse la rétrogradation d'admission."""
    profil, completes = fusionner_profils(
        ProfilCandidat(), ProfilDeclareExtrait(serie_bac="bac D")
    )

    assert profil.serie_bac == "D"
    assert "série du baccalauréat" in completes


def test_fusion_remplit_un_champ_scalaire_seulement_s_il_manque():
    base = ProfilCandidat()
    extrait = ProfilDeclareExtrait(
        serie_bac="D", environnement_travail_recherche="laboratoire"
    )

    profil, completes = fusionner_profils(base, extrait)

    assert profil.serie_bac == "D"
    assert profil.environnement_travail_recherche == "laboratoire"
    assert set(completes) == {"série du baccalauréat", "environnement de travail"}


def test_fusion_ne_remplace_jamais_une_note_deja_connue():
    base = ProfilCandidat(resultats_scolaires={"mathematiques": 15.0})
    extrait = ProfilDeclareExtrait(
        resultats_scolaires=[
            NoteMatiereDeclaree(matiere="mathematiques", note=8.0),
            NoteMatiereDeclaree(matiere="physique", note=17.0),
        ]
    )

    profil, completes = fusionner_profils(base, extrait)

    assert profil.resultats_scolaires == {"mathematiques": 15.0, "physique": 17.0}
    assert completes == ["notes scolaires"]


def test_fusion_plafonne_la_taille_des_listes():
    base = ProfilCandidat(centres_interet=[f"interet{i}" for i in range(14)])
    extrait = ProfilDeclareExtrait(centres_interet=["nouveau_a", "nouveau_b", "nouveau_c"])

    profil, _ = fusionner_profils(base, extrait)

    assert len(profil.centres_interet) == 15  # MAX_ELEMENTS_PAR_LISTE
    assert "nouveau_a" in profil.centres_interet
    assert "nouveau_c" not in profil.centres_interet


def test_fusion_sans_rien_de_nouveau_ne_signale_aucun_champ():
    base = ProfilCandidat(matieres_preferees=["maths"])
    extrait = ProfilDeclareExtrait(matieres_preferees=["maths"])

    profil, completes = fusionner_profils(base, extrait)

    assert profil.matieres_preferees == ["maths"]
    assert completes == []


# --- extraire_profil_declare (LLM simulé) ----------------------------------


def test_message_vide_ne_declenche_aucun_appel(monkeypatch):
    appels = []
    monkeypatch.setattr(
        "src.extraction_profil.llm_call", lambda *a, **k: appels.append(1)
    )
    base = ProfilCandidat(matieres_preferees=["maths"])

    profil, completes = extraire_profil_declare("   ", [], base)

    assert appels == []
    assert profil is base
    assert completes == []


def test_extraction_fusionne_le_resultat_du_modele(monkeypatch):
    monkeypatch.setattr(
        "src.extraction_profil.llm_call",
        lambda *a, **k: ProfilDeclareExtrait(
            matieres_preferees=["maths", "informatique"], serie_bac="D"
        ),
    )
    base = ProfilCandidat(competences_declarees=["python"])

    profil, completes = extraire_profil_declare(
        "J'adore les maths et l'informatique, je suis en bac D", [], base
    )

    assert profil.matieres_preferees == ["maths", "informatique"]
    assert profil.competences_declarees == ["python"]  # préservé
    assert profil.serie_bac == "D"
    assert "matières préférées" in completes


def test_extraction_ne_regarde_que_les_messages_du_candidat(monkeypatch):
    recu = {}
    monkeypatch.setattr(
        "src.extraction_profil.llm_call",
        lambda systeme, utilisateur, **k: recu.setdefault("texte", utilisateur)
        or ProfilDeclareExtrait(),
    )
    historique = [
        TourConversation(question="Je fais du bac D", reponse="IGGLIA pourrait convenir."),
    ]

    extraire_profil_declare("Et les maths ?", historique, ProfilCandidat())

    assert "bac D" in recu["texte"]
    assert "Et les maths ?" in recu["texte"]
    # La réponse de l'assistant (qui nomme IGGLIA) n'est pas donnée à l'extracteur.
    assert "IGGLIA" not in recu["texte"]


def test_echec_llm_est_propage_pour_degradation(monkeypatch):
    def _tombe(*a, **k):
        raise LLMError("quota")

    monkeypatch.setattr("src.extraction_profil.llm_call", _tombe)

    with pytest.raises(LLMError):
        extraire_profil_declare("J'aime les maths", [], ProfilCandidat())
