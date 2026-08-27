# Limites, biais et risques

ORIENT'IA est un prototype d'aide à la décision, pas un dispositif d'admission ni un prédicteur de réussite. Les recommandations ne remplacent ni l'avis d'un conseiller pédagogique ni une décision officielle.

## Données et représentativité

- Le jeu d'entraînement principal est synthétique. Une excellente performance synthétique signifie surtout que le modèle retrouve les règles utilisées pour générer les profils.
- Les enquêtes sont petites, auto-sélectionnées, collectées sur une période courte et très concentrées sur quelques parcours informatiques. Les métriques réelles ont donc une forte incertitude.
- Plusieurs parcours ont peu ou pas de répondants ; une performance moyenne ne garantit aucune qualité pour ces classes.
- Les réponses sont auto-déclarées. Satisfaction, niveau et adéquation au métier ne sont pas des mesures indépendantes de réussite.
- Certaines matières proviennent de calendriers d'épreuves relayés par un groupe étudiant : ils ne constituent ni une maquette pédagogique exhaustive ni une source institutionnelle vérifiée.
- Des débouchés ont été générés automatiquement et sont isolés comme candidats à valider ; ils ne doivent pas être présentés comme officiels.

## Biais possibles

- Le vocabulaire et les archétypes choisis par l'équipe encodent une vision normative des parcours et peuvent renforcer des associations simplistes.
- Le déséquilibre des réponses réelles favorise les parcours surreprésentés et rend la calibration fragile hors de cet échantillon.
- Un profil moins détaillé reçoit mécaniquement une recommandation moins informative. Cela peut défavoriser les personnes moins à l'aise avec le questionnaire ou le français.
- L'absence volontaire de données sensibles réduit le risque de discrimination directe, mais des variables déclarées peuvent encore agir comme proxys indirects.
- Le texte produit par un LLM reste non déterministe : une source pertinente peut être omise ou une formulation trop affirmative peut apparaître malgré les contrôles.

## Risques techniques et opérationnels

- Dépendance à Gemini, au réseau et aux quotas : latence, indisponibilité ou dégradation vers une escalade prudente.
- L'index vectoriel peut être incomplet ou obsolète ; une empreinte déclenche sa reconstruction, sans garantir la qualité du corpus amont.
- Une injection de prompt ou un contenu source malveillant peut tenter de détourner l'agent. Des tests et garde-fous existent, sans constituer une preuve de sécurité absolue.
- Les règles d'admission peuvent changer. Toute décision officielle doit être vérifiée sur la source ISPM courante.
- Un artefact `joblib` dépend des versions Python/scikit-learn ; le script de réentraînement est la référence reproductible.

## Vie privée

- Aucun nom, e-mail, téléphone ni attribut sensible n'est nécessaire au modèle.
- Les exports publiés sont anonymisés et les motifs d'e-mail ou de secret sont masqués.
- Un petit effectif combiné à un parcours, une année et un métier rare conserve un risque de ré-identification par recoupement. Ne pas republier les données individuelles au-delà du cadre consenti sans réévaluation.
- Les journaux peuvent contenir du texte utilisateur : appliquer une politique de rétention, limiter l'accès et supprimer les traces avant toute diffusion publique.

## Mesures de réduction et décisions humaines

- Afficher l'avertissement obligatoire sur chaque recommandation.
- Montrer les sources, leur statut et l'incertitude ; ne jamais convertir un score en promesse de réussite.
- Demander davantage d'information lorsque moins de deux traits exploitables sont fournis.
- Escalader les profils ambigus, faibles en confiance ou concernés par une condition d'admission incertaine.
- Faire valider périodiquement corpus, règles et libellés par l'ISPM.
- Réentraîner et recalibrer sur un échantillon réel plus large, stratifié par parcours, puis publier les métriques avec intervalles de confiance et analyses par sous-groupe non sensible.

