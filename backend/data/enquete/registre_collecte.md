# Registre de collecte de l'enquête (DATA-5)

Généré automatiquement par `scripts/preparer_jeu_test_reel.py` à partir de l'export anonymisé — ne pas éditer à la main, relancer le script.

## Population et période

- **Réponses reçues** : 86
- **Étudiants actuels** : 71
- **Professionnels diplômés** : 15
- **Période de collecte** : 2026-08-26 → 2026-08-27

## Volumes retenus pour l'évaluation ML (ML-7)

- **Parcours ou mention reconnu** : 79/86
  - dont étiquette au niveau **parcours** (précise) : 79
  - dont étiquette au niveau **mention** seulement (ambiguë entre plusieurs parcours, exclue des métriques par parcours) : 2
  - **non reconnu** (réponse libre non rattachable, écartée) : 5
- **Étiquette jugée fiable** (satisfaction/adéquation déclarée ≥ 3/5) : 62

## Texte de consentement recueilli

> J'accepte que mes réponses anonymisées soient utilisées dans le cadre d'un projet académique de l'ISPM (hackathon ORIENT'IA). Aucune information permettant de m'identifier ne sera collectée.

## Procédure d'anonymisation appliquée

- Colonne de consentement retirée avant tout traitement.
- Horodatage réduit à la date (heure/minute/seconde supprimées).
- `guardrails.masquer_objet` appliqué à chaque enregistrement produit (masque e-mails et motifs de secret résiduels dans le texte libre conservé).
- Réponse « Anonyme »/« Aucun »/équivalent sur le métier déclaré traitée comme une non-réponse, jamais comme un intitulé de poste.
- Aucun champ nominatif n'a été collecté par le formulaire source (consentement ci-dessus) ; le texte libre conservé dans le jeu de test final se limite aux matières préférées et à l'intitulé de poste déclaré.

## Biais et limites constatés

- Échantillon fortement concentré sur les mentions déjà identifiées : IGGLIA (27), ESIIA (14), ISAIA (11), IMTICIA (6), GCA (5), DTJA (3), EMII (3), TEH (2), CAA (2), PIP (2), MENTION-BIOTECH-AGRO (1), AEE (1), FIC (1), ICMP (1), IAA (1), MENTION-INFO-TELECOM (1).
- La question de niveau scolaire ne porte que sur un jugement combiné « maths/info » (échelle 1-5), jamais une note par matière : reporté à l'identique sur `mathematiques` et `informatique`, ce qui sous-estime la variance réelle entre ces deux matières.
- Aucune réponse ne renseigne `serie_bac`, `activites_projets`, `competences_declarees`, `centres_interet` ni `environnement_travail_recherche` : ces champs restent vides dans le jeu de test, jamais complétés par supposition.
- Échantillon de petite taille (moins de 100 réponses) : un recoupement entre `label_brut` (parcours + année) et le métier déclaré pourrait, en théorie, permettre à une personne connaissant la promotion concernée d'identifier un répondant. Ce risque de ré-identification par petits effectifs n'est pas éliminé par le masquage de motifs (e-mails/secrets) appliqué ici — à garder à l'esprit avant toute diffusion au-delà de l'équipe projet.
