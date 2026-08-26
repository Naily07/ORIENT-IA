"""Archétypes de profil par parcours — hypothèses documentées pour DATA-6.

**Ce que c'est** : pour chacun des 16 parcours réels de l'ISPM (collectés en
DATA-1, `backend/data/parcours.json`), un profil-type plausible (matières,
compétences, centres d'intérêt, préférences professionnelles, environnement
de travail recherché), dérivé à la main des descriptions officielles du
corpus (`backend/data/corpus.json`) — pas inventé au hasard.

**Pourquoi ça existe** : c'est l'hypothèse de génération exigée par le sujet
pour tout jeu de données synthétique (§5 : « documenter leur méthode de
génération, les hypothèses utilisées »). `donnees_synthetiques.py` échantillonne
et perturbe ces archétypes pour produire des profils d'entraînement.

**Limite assumée, à ne jamais masquer** : ces archétypes sont une
simplification manuelle, pas une vérité de terrain. Un modèle entraîné
uniquement dessus apprend à retrouver *ces hypothèses*, pas un lien réel
profil → parcours (risque explicitement nommé au §5 du sujet). C'est
exactement pourquoi le sujet exige de valider sur l'enquête réelle
(DATA-4/DATA-7, ML-7) avant de faire confiance aux résultats mesurés ici.
"""

ARCHETYPES: dict[str, dict] = {
    "IGGLIA": {
        "matieres": ["informatique", "gestion", "mathematiques"],
        "competences": ["programmation", "algorithmique", "gestion_de_projet"],
        "centres_interet": ["technologie", "logiciels", "entrepreneuriat"],
        "preferences_professionnelles": ["developpement_logiciel", "gestion_it"],
        "environnement": "bureau_informatique",
    },
    "ESIIA": {
        "matieres": ["electronique", "informatique", "mathematiques"],
        "competences": ["electronique", "programmation", "architecture_systemes"],
        "centres_interet": ["materiel_informatique", "robotique"],
        "preferences_professionnelles": ["ingenierie_electronique", "systemes_embarques"],
        "environnement": "laboratoire_atelier",
    },
    "IMTICIA": {
        "matieres": ["informatique", "arts_numeriques", "communication"],
        "competences": ["multimedia", "communication_numerique"],
        "centres_interet": ["creation_numerique", "reseaux_sociaux", "audiovisuel"],
        "preferences_professionnelles": ["production_multimedia", "communication_digitale"],
        "environnement": "studio_creatif",
    },
    "ISAIA": {
        "matieres": ["mathematiques", "informatique", "economie"],
        "competences": ["statistiques", "analyse_de_donnees"],
        "centres_interet": ["donnees", "finance"],
        "preferences_professionnelles": ["data_analyst", "informatique_decisionnelle"],
        "environnement": "bureau_analytique",
    },
    "EMII": {
        "matieres": ["physique", "mecanique", "informatique"],
        "competences": ["mecanique", "automatisme", "maintenance_industrielle"],
        "centres_interet": ["machines", "robotique_industrielle"],
        "preferences_professionnelles": ["technicien_industriel", "maintenance"],
        "environnement": "usine_atelier",
    },
    "ICMP": {
        "matieres": ["chimie", "physique", "geologie"],
        "competences": ["chimie_industrielle", "securite_industrielle"],
        "centres_interet": ["ressources_naturelles", "industrie_lourde"],
        "preferences_professionnelles": ["ingenieur_chimiste", "mines"],
        "environnement": "site_industriel_terrain",
    },
    "GCA": {
        "matieres": ["mathematiques", "physique", "dessin_technique"],
        "competences": ["dessin_technique", "gestion_chantier"],
        "centres_interet": ["construction", "urbanisme", "architecture"],
        "preferences_professionnelles": ["architecte", "ingenieur_btp"],
        "environnement": "chantier_bureau_etudes",
    },
    "CAA": {
        "matieres": ["gestion", "economie", "communication"],
        "competences": ["negociation", "marketing"],
        "centres_interet": ["commerce", "entrepreneuriat"],
        "preferences_professionnelles": ["commercial", "gestion_entreprise"],
        "environnement": "bureau_terrain_commercial",
    },
    "FIC": {
        "matieres": ["mathematiques", "comptabilite", "economie"],
        "competences": ["comptabilite", "analyse_financiere"],
        "centres_interet": ["finance", "chiffres"],
        "preferences_professionnelles": ["comptable", "analyste_financier"],
        "environnement": "bureau",
    },
    "DTJA": {
        "matieres": ["droit", "informatique"],
        "competences": ["techniques_juridiques", "redaction"],
        "centres_interet": ["droit_des_affaires", "justice"],
        "preferences_professionnelles": ["juriste_entreprise", "assistant_juridique"],
        "environnement": "bureau_cabinet",
    },
    "EMP": {
        "matieres": ["economie", "mathematiques", "gestion"],
        "competences": ["analyse_economique", "gestion_de_projet"],
        "centres_interet": ["economie", "politique_publique"],
        "preferences_professionnelles": ["charge_etudes_economiques", "chef_de_projet"],
        "environnement": "bureau",
    },
    "IAA": {
        "matieres": ["biologie", "chimie"],
        "competences": ["controle_qualite", "process_industriel"],
        "centres_interet": ["agroalimentaire", "nutrition"],
        "preferences_professionnelles": ["technicien_qualite", "production_agroalimentaire"],
        "environnement": "usine_agroalimentaire",
    },
    "PIP": {
        "matieres": ["biologie", "chimie"],
        "competences": ["recherche_pharmaceutique", "chimie_industrielle"],
        "centres_interet": ["sante", "medecine_naturelle", "recherche"],
        "preferences_professionnelles": ["recherche_pharmaceutique", "industrie_pharma"],
        "environnement": "laboratoire",
    },
    "AEE": {
        "matieres": ["biologie", "sciences_de_la_terre"],
        "competences": ["techniques_agricoles", "gestion_rurale"],
        "centres_interet": ["agriculture", "environnement_rural"],
        "preferences_professionnelles": ["technicien_agricole", "developpement_rural"],
        "environnement": "terrain_rural",
    },
    "TEE": {
        "matieres": ["geographie", "sciences_environnement", "langues"],
        "competences": ["guide_touristique", "sensibilisation_environnementale"],
        "centres_interet": ["nature", "ecotourisme", "voyage"],
        "preferences_professionnelles": ["guide_ecotouristique", "gestion_aires_protegees"],
        "environnement": "terrain_nature",
    },
    "TEH": {
        "matieres": ["langues", "histoire", "gestion_hoteliere"],
        "competences": ["accueil", "gestion_hoteliere"],
        "centres_interet": ["culture", "voyage", "hospitalite"],
        "preferences_professionnelles": ["hotellerie", "accueil_touristique"],
        "environnement": "hotel_contact_client",
    },
}


def _vocabulaire(champ: str) -> tuple[str, ...]:
    """Union triée des valeurs d'un champ à travers tous les archétypes.

    Dériver le vocabulaire des archétypes plutôt que de le retaper à la main
    garantit qu'ils restent en phase : ajouter une valeur à un archétype
    l'ajoute automatiquement au vocabulaire utilisé par `features.py`.
    """
    valeurs: set[str] = set()
    for archetype in ARCHETYPES.values():
        contenu = archetype[champ]
        valeurs.update(contenu if isinstance(contenu, list) else [contenu])
    return tuple(sorted(valeurs))


VOCAB_MATIERES = _vocabulaire("matieres")
VOCAB_COMPETENCES = _vocabulaire("competences")
VOCAB_CENTRES_INTERET = _vocabulaire("centres_interet")
VOCAB_PREFERENCES_PRO = _vocabulaire("preferences_professionnelles")
VOCAB_ENVIRONNEMENTS = _vocabulaire("environnement")

PARCOURS_CONNUS: tuple[str, ...] = tuple(sorted(ARCHETYPES))
