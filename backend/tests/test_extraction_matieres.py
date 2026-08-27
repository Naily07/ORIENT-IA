"""Tests de l'extraction des matières depuis les calendriers ISPM (DATA-1).

Le script d'extraction est le **mécanisme reproductible de collecte** exigé
comme livrable (§3) : ses règles de nettoyage doivent être vérifiables, sinon
rien ne distingue une matière d'un fragment d'en-tête. Chaque cas ci-dessous
vient d'une sortie réelle de l'OCR sur l'archive.
"""

from scripts.extraire_matieres import (
    agreger,
    classe_imprimee,
    cle_canonique,
    construire_lexique,
    est_bruit,
    matieres_d_un_document,
    nettoyer_libelle,
    parcours_concernes,
    recoller_cellules,
    retablir_espaces,
    retirer_groupe_classe,
    segmenter,
)


def _bloc(texte: str, x: float, y: float, largeur: float = 100, hauteur: float = 20):
    return {
        "texte": texte,
        "boite": [[x, y], [x + largeur, y], [x + largeur, y + hauteur], [x, y + hauteur]],
    }


# --- Filtrage du bruit ---------------------------------------------------------


def test_l_entete_et_le_pied_de_page_sont_du_bruit():
    for texte in (
        "INSTITUTSUPERIEURPOLYTECHNIQUEDEMADAGASCAR",
        "AMBATOMARO-ANTSOBOLO-ANTANANARIVO",
        "I.S.P.M.",
        "ProfesseurRABOANARYJulienAmedee",
        "Le Recteur,",
        "Classe : IGGLIA 3",
    ):
        assert est_bruit(texte), texte


def test_les_jours_colles_a_leur_date_sont_du_bruit():
    """Le piège réel : « Jeudi05 » n'a pas de frontière de mot entre « i » et
    « 0 », donc un motif en `\\b` laissait passer toutes les dates comme
    matières."""
    for texte in ("Jeudi05", "LUNDI03.08.2026", "Mardi 17.03.2026", "Mercredi18:Expression"):
        assert est_bruit(texte), texte


def test_les_fragments_du_bandeau_sont_du_bruit():
    for texte in ("QUE", "TITUT", "S.P.M", "POL"):
        assert est_bruit(texte), texte


def test_une_vraie_matiere_n_est_pas_du_bruit():
    for texte in ("Intelligence Artificielle", "Cryptographie", "Analyse Harmonique"):
        assert not est_bruit(texte), texte


def test_les_sigles_courts_legitimes_survivent():
    """Filtrer tous les sigles de trois lettres supprimerait RDM (résistance
    des matériaux), SNI, PAO ou CAE, qui sont de vraies matières."""
    for sigle in ("RDM", "SNI", "PAO", "CAE", "GAFI"):
        assert not est_bruit(sigle), sigle


# --- Normalisation des libellés ------------------------------------------------


def test_les_espaces_perdus_sont_retablis():
    assert retablir_espaces("AnalyseMathématique") == "Analyse Mathématique"
    assert retablir_espaces("BiologieCellulaire") == "Biologie Cellulaire"


def test_un_libelle_correct_est_inchange():
    assert retablir_espaces("Analyse Harmonique") == "Analyse Harmonique"


def test_le_groupe_de_classe_est_retire():
    """« CYBERSECURITE(INFO5) » et « Cybersécurité » sont la même matière : le
    suffixe dit à quelle classe l'épreuve s'adresse."""
    assert retirer_groupe_classe("CYBERSECURITE(INFO5)") == "CYBERSECURITE"
    assert retirer_groupe_classe("ANGLAIS(BIO5-INFO5-GIC5)") == "ANGLAIS"
    assert retirer_groupe_classe("TP Informatique (IGGLIA1A)") == "TP Informatique"


def test_une_parenthese_qui_n_est_pas_un_groupe_est_conservee():
    assert retirer_groupe_classe("Anglais (renforcé)") == "Anglais (renforcé)"


def test_le_tout_majuscules_est_ramene_a_une_casse_comparable():
    assert nettoyer_libelle("CYBERSECURITE(INFO5)") == "Cybersecurite"


def test_les_sigles_restent_en_majuscules():
    for sigle in ("SNI", "CDS", "GAFI", "ODIA"):
        assert nettoyer_libelle(sigle) == sigle


# --- Clé canonique -------------------------------------------------------------


def test_les_variantes_d_un_meme_intitule_partagent_une_cle():
    """Sans cette clé, « Théorie des nombres », « Theorie des nombres » et
    « Theoriedesnombres » deviendraient trois matières distinctes."""
    cles = {
        cle_canonique("Théorie des nombres"),
        cle_canonique("Theorie des nombres"),
        cle_canonique("THEORIEDESNOMBRES"),
    }
    assert len(cles) == 1


def test_deux_matieres_differentes_ont_des_cles_differentes():
    assert cle_canonique("Analyse") != cle_canonique("Analyse Harmonique")


# --- Segmentation --------------------------------------------------------------


def test_un_libelle_colle_est_recoupe_avec_le_lexique_du_corpus():
    lexique = construire_lexique(["Analyse Harmonique", "Theorie des nombres"])
    assert segmenter("Analyseharmonique", lexique) == "Analyse harmonique"


def test_un_libelle_introuvable_dans_le_lexique_est_laisse_intact():
    """Mieux vaut un mot collé qu'une découpe inventée."""
    lexique = construire_lexique(["Analyse Harmonique"])
    assert segmenter("Xyzzyplughblorple", lexique) == "Xyzzyplughblorple"


def test_un_libelle_deja_espace_n_est_pas_touche():
    lexique = construire_lexique(["Analyse Harmonique"])
    assert segmenter("Analyse Harmonique", lexique) == "Analyse Harmonique"


# --- Recollage des cellules multi-lignes ---------------------------------------


def test_une_cellule_sur_deux_lignes_est_recollee():
    """« Droit administratif » sort en deux blocs superposés : les laisser
    séparés créerait deux matières fantômes, dont « administratif »."""
    blocs = [_bloc("Droit", x=10, y=100), _bloc("administratif", x=12, y=122)]
    assert recoller_cellules(blocs) == ["Droit administratif"]


def test_deux_colonnes_distinctes_ne_sont_pas_recollees():
    blocs = [_bloc("Codage", x=10, y=100), _bloc("Cryptographie", x=400, y=100)]
    assert sorted(recoller_cellules(blocs)) == ["Codage", "Cryptographie"]


def test_deux_lignes_eloignees_ne_sont_pas_recollees():
    blocs = [_bloc("Codage", x=10, y=100), _bloc("Linux", x=10, y=400)]
    assert sorted(recoller_cellules(blocs)) == ["Codage", "Linux"]


def test_aucun_bloc_ne_produit_aucune_cellule():
    assert recoller_cellules([]) == []


# --- Chaîne complète -----------------------------------------------------------


def test_un_document_realiste_ne_rend_que_ses_matieres():
    blocs = [
        _bloc("INSTITUTSUPERIEURPOLYTECHNIQUEDEMADAGASCAR", x=200, y=10, largeur=600),
        _bloc("Classe : IGGLIA 3", x=300, y=60),
        _bloc("Mardi04", x=10, y=120),
        _bloc("Gestion de Projets", x=10, y=160),
        _bloc("Mercredi05", x=300, y=120),
        _bloc("Intelligence Artificielle", x=300, y=160),
        _bloc("Le Recteur", x=600, y=500),
    ]
    assert sorted(matieres_d_un_document(blocs)) == [
        "Gestion de Projets",
        "Intelligence Artificielle",
    ]


def test_agreger_regroupe_les_variantes_et_conserve_le_niveau():
    documents = [
        {
            "parcours": "IGGLIA",
            "niveau": "L3",
            "semestre": "S1",
            "fichier": "L3/IGGLIA/S1.jpg",
            "blocs": [_bloc("Theorie des nombres", x=10, y=100)],
        },
        {
            "parcours": "IGGLIA",
            "niveau": "L3",
            "semestre": "S2",
            "fichier": "L3/IGGLIA/S2.jpg",
            "blocs": [_bloc("Théorie des nombres", x=10, y=100)],
        },
    ]
    catalogue, programme = agreger(documents)

    assert len(catalogue) == 1
    assert next(iter(catalogue.values()))["occurrences"] == 2
    assert [p["niveau"] for p in programme] == ["L3", "L3"]
    assert [p["semestre"] for p in programme] == ["S1", "S2"]


# --- Classe imprimée vs nom de dossier (découverte TEE/TOUR) -------------------


def test_la_classe_imprimee_prime_sur_le_nom_de_dossier():
    """L'archive range les calendriers de master de la mention Tourisme sous
    « TEH », alors que le document annonce « Classe : TOUR5 » — une classe qui
    réunit TEE et TEH. Sans cette lecture, TEE se retrouvait avec zéro matière
    alors que la moitié de ce programme le concerne (Écotourisme, Écologie
    marine, Droit de l'environnement)."""
    document = {
        "parcours": "TEH",
        "blocs": [_bloc("Classe : TOUR5", x=300, y=60), _bloc("Ecotourisme", x=10, y=200)],
    }
    assert classe_imprimee(document["blocs"]) == "TOUR"
    assert parcours_concernes(document) == ("TEE", "TEH")


def test_un_parcours_isole_reste_seul():
    document = {
        "parcours": "IGGLIA",
        "blocs": [_bloc("Classe : IGGLIA 3", x=300, y=60)],
    }
    assert parcours_concernes(document) == ("IGGLIA",)


def test_le_tronc_commun_bio_alimente_les_trois_parcours():
    document = {"parcours": "BIO", "blocs": [_bloc("Classe : BIO2", x=300, y=60)]}
    assert parcours_concernes(document) == ("AEE", "IAA", "PIP")


def test_sans_classe_imprimee_on_retombe_sur_le_dossier():
    document = {"parcours": "ISAIA", "blocs": [_bloc("Analyse", x=10, y=100)]}
    assert classe_imprimee(document["blocs"]) is None
    assert parcours_concernes(document) == ("ISAIA",)


# --- Parenthèses pleine chasse -------------------------------------------------


def test_les_parentheses_pleine_chasse_sont_traitees():
    """L'OCR rend parfois « （ » (U+FF08). Les motifs ASCII ne les voyaient pas,
    et « Marketing（TER5-DTJA5-TOUR5) » traversait toute la chaîne avec son
    groupe de classe accolé."""
    assert nettoyer_libelle("Marketing\uff08TER5-DTJA5-TOUR5)") == "Marketing"


def test_le_pied_de_page_est_filtre_meme_ampute():
    """« rofesseur RABOANARY… » : l'OCR perd parfois la capitale initiale, ce
    qui faisait passer la signature du Recteur pour une matière."""
    assert est_bruit("rofesseur RABOANARY Julien Amedee")
    assert est_bruit("Professeur RABOANARY Julien Amédée")
