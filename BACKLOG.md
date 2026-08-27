# Backlog ORIENT'IA

Backlog de développement pour le hackathon **ORIENT'IA** (ISPM, Master 2, remise le
27/08/2026 17h00). Organisé par bloc fonctionnel ; chaque ticket porte un ID, une
description et ses dépendances pour faciliter la répartition en équipe (2 à 7
étudiants).

> Contexte complet du sujet et découpage en 7 groupes de travail : voir la feuille de
> route publiée précédemment (architecture cible, choix technologiques, trame horaire).
> Ce document la complète en descendant au niveau ticket, à la lumière de l'analyse du
> dépôt [X-project-ISPM/EXAM-S2](https://github.com/X-project-ISPM/EXAM-S2) ci-dessous.

## Analyse du dépôt EXAM-S2 (réutilisation)

`EXAM-S2` est le rendu d'un **hackathon précédent** de la même équipe/organisation
(*mAIntenance & Assistance*, un assistant de support informatique — sujet ISPM
différent, 8h au lieu de 2 jours). Licence MIT, `Copyright (c) 2026 X-project-ISPM` —
même organisation que le dépôt cible : réutilisation sans restriction.

Ce n'est **pas un starter ORIENT'IA** : le domaine (tickets IT vs. orientation
pédagogique) et l'exigence centrale diffèrent. Le sujet précédent excluait
explicitement l'entraînement d'un modèle de ML (« aucun point n'est réservé à
l'entraînement d'un modèle de ML ») ; ORIENT'IA l'exige (18 points). En revanche
l'architecture (FastAPI + LLM + RAG + agent à outils + garde-fous + observabilité),
la qualité d'ingénierie et plusieurs mécanismes sont directement transposables.

### Ce qui est réutilisable quasiment tel quel

| Module EXAM-S2 | Ce qu'il apporte | Pourquoi il tient pour ORIENT'IA |
|---|---|---|
| `backend/src/llm_client.py` | Point d'appel LLM unique : retry sur quota, lissage de débit, timeout, hooks d'observabilité, sortie contrainte par schéma Pydantic (`response_schema`) | Générique, sans aucune logique métier IT. Le pipeline ORIENT'IA enchaînera aussi plusieurs appels LLM (profil, RAG, explication) et aura les mêmes problèmes de quota/latence à gérer |
| `backend/src/observability.py` | Traces JSONL (déroulé, appels d'outils, appels LLM bruts), estimation de coût, `ChronoLatence` | Couvre déjà l'exigence « traces à examiner » du sujet ORIENT'IA (§15) presque champ pour champ |
| `backend/src/rag.py` | Chunking par phrases avec chevauchement borné, ChromaDB en **espace cosinus explicite**, `upsert` idempotent, diversification par source, génération citée avec contrôle anti-hallucination des sources | Le moteur RAG est domaine-agnostique ; seul le contenu du prompt et le corpus changent |
| `backend/src/guardrails.py` (détection d'injection, masquage) | Deux couches (mots-clés + LLM), masquage de secrets avant log | Le mécanisme anti-injection est générique ; seules les catégories « sensibles » et l'escalade sont spécifiques au support IT |
| `backend/src/sortie.py` | Retry sur sortie non conforme au schéma, réponse de repli toujours valide | Pattern générique de robustesse |
| `backend/src/config.py` (pattern) | Configuration centralisée via `pydantic-settings` | Pattern à reprendre, valeurs à réinitialiser pour le nouveau domaine |
| Méthodologie de calibration (`tests/calibrer_seuil.py`) | Balayage mesuré du seuil de pertinence RAG, découverte que ChromaDB indexe en L2 au carré par défaut | **Piège à éviter directement** : sans le forcer en cosinus, un seuil « raisonnable » donne 0 % de rappel — vécu sur EXAM-S2, à ne pas redécouvrir |
| Pattern d'orchestrateur (« une seule étape bloquante, le reste dégrade proprement », « le code contresigne la décision de l'agent ») | Philosophie robuste : catégorie/équipe déterministes, sources recoupées avec les passages fournis, jamais d'erreur nue | Correspond exactement à l'exigence ORIENT'IA de distinguer résultat ML / info documentaire / règle / texte LLM |
| Squelette Streamlit (chat, scénarios pré-remplis, onglet observabilité, explorateur de données) | UI de démo fonctionnelle en une techno | Directement adaptable, vocabulaire à changer |

### Ce qui doit être significativement réajusté

- **Schémas de sortie** (`schemas.py`) : `TicketDecision`/`Classification` sont
  spécifiques aux tickets IT (catégorie/priorité/équipe). ORIENT'IA a besoin d'un
  schéma de recommandation (parcours proposés, score d'adéquation, sources citées,
  incertitude déclarée) — même *philosophie* (vocabulaires contrôlés en `Literal`,
  séparation extraction LLM / décision code), contenu entièrement nouveau.
- **Outils de l'agent** (`tools.py`) : les 8 outils (`rechercher_utilisateur`,
  `creer_ticket`...) n'ont aucun équivalent direct. Le *mécanisme* (spec centralisée,
  validation des paramètres, sensibilité déterministe côté code, function calling
  natif Gemini) est repris ; le contenu métier est à réécrire pour
  `rechercher_formation`, `comparer_parcours`, `verifier_prerequis`, etc.
- **Diagnostic / extraction de profil** (`diagnostic.py`) : le principe (le LLM
  extrait, le code décide ce qui manque, garde-fou anti-écho pour éviter qu'un champ
  vague soit pris pour renseigné) est excellent et transposable au recueil progressif
  du profil candidat — mais les champs eux-mêmes (utilisateur/équipement/application)
  sont à remplacer par matières préférées/résultats/compétences/centres d'intérêt.
- **Garde-fous métier** : la logique d'escalade immédiate est calée sur la
  cybersécurité IT. ORIENT'IA a besoin d'équivalents spécifiques au sujet : détection
  de critères discriminatoires, refus de profilage psychologique — absents d'EXAM-S2.
- **Frontend** : le vocabulaire (tickets, équipes, priorités) est à remplacer par
  profils/parcours/scores, et la mention obligatoire du sujet ORIENT'IA
  (« ORIENT'IA... ne remplace ni l'avis d'un conseiller ni une décision officielle »)
  doit être ajoutée — absente d'EXAM-S2 par nature.

### Ce qui n'existe pas du tout dans EXAM-S2 et doit être construit intégralement

- **Tout le volet Machine Learning entraîné** (EXAM-S2 n'entraîne rien, uniquement du
  prompting LLM few-shot) : jeu de données, EDA, baseline, comparaison d'approches,
  métriques de ranking/classification, analyse de généralisation synthèse→réel.
- **La traçabilité des sources documentaires** (registre des sources avec URL, date,
  statut officiel/institutionnel/externe) — EXAM-S2 a une base de connaissances sans
  registre de provenance.
- **L'enquête terrain** (étudiants + professionnels) et son registre de collecte.
- **L'ontologie / graphe de connaissances** (extension symbolique).
- **Les catégories d'évaluation obligatoires du sujet ORIENT'IA** (32 cas, 8
  catégories dont biais et refus de profilage) — EXAM-S2 évalue classification + RAG
  + 4 scénarios, une structure différente et plus étroite.

---

## Légende des tickets

- **[REUSE]** — code ou mécanisme copiable depuis EXAM-S2 avec adaptation mineure de
  noms/valeurs.
- **[ADAPT]** — le principe/pattern d'EXAM-S2 est repris, le contenu est réécrit pour
  le domaine orientation.
- **[NOUVEAU]** — aucun équivalent dans EXAM-S2, à construire depuis zéro.

---

## 🔧 Setup & fondations

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~SETUP-1~~ ✅ | Structure du dépôt | `backend/src`, `backend/tests/`, `pyproject.toml`, `requirements.txt`, `run.sh`/`run.ps1`, `.env.example`, `.gitignore` **[REUSE]** ossature EXAM-S2 — `frontend/` et `backend/data/` restent à créer avec FE-1/DATA-1 | — |
| ~~SETUP-2~~ ✅ | Configuration centralisée | `config.py` (pydantic-settings) : modèle LLM, seuils RAG, chemins, budget orchestrateur **[REUSE]** structure, valeurs de départ héritées d'EXAM-S2 à recalibrer (RAG-5) | SETUP-1 |
| ~~SETUP-3~~ ✅ | Client LLM unique | `llm_client.py` : `llm_call`/`llm_call_with_tools`, retry quota, lissage débit, hooks d'observabilité **[REUSE quasi telle quelle]** | SETUP-2 |
| ~~SETUP-4~~ ✅ | Schémas du domaine orientation | `schemas.py` : `ProfilCandidat`, `AnalyseProfil`, `RecommandationDecision` (parcours, score, sources, incertitude, informations_manquantes, action) **[ADAPT]** philosophie de `TicketDecision` (vocabulaires `Literal`, séparation LLM/code) | SETUP-1 |
| ~~SETUP-5~~ ✅ | Modèles de données du corpus | `models.py` : `Mention`, `Parcours`, `Matiere`, `Competence`, `Prerequis`, `Metier` + `CorpusFormations`/`charger_corpus_formations()` **[ADAPT]** `ArticleKB` → `DocumentSource` (RAG) + modèles structurés séparés pour le ML/l'ontologie — les fichiers JSON réels et le lien de provenance vers le registre des sources restent à faire avec DATA-1/DATA-3 | SETUP-1 |

## 📚 Données — corpus, traçabilité, enquête

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~DATA-1~~ (partiel) | Collecte du corpus ISPM | Mentions, parcours, niveaux, prérequis depuis `ispm-edu.com` + recoupement annuaire.mg **[NOUVEAU]** — 6 mentions et 16 parcours réels. **Matières collectées** (27/08) : 490 intitulés rattachés aux 16 parcours, extraits par OCR de 129 calendriers d'épreuves (L1→M2, S1/S2) transmis par l'équipe — `backend/scripts/extraire_matieres.py`, mécanisme reproductible et testé. **Source externe** (groupe Facebook étudiant) malgré l'apparence officielle des documents : limites enregistrées au registre (`SRC-CALENDRIERS-FACEBOOK`). Le rattachement suit la **classe imprimée** sur le document et non le nom de dossier : les calendriers de master rangés sous « TEH » annoncent « Classe : TOUR4/TOUR5 », classe qui réunit les deux parcours de la mention Tourisme — sans quoi TEE se retrouvait à zéro matière. **Restent incomplets** : compétences, débouchés métiers, passerelles, et les matières de licence propres à TEE. **Débouchés (28/08)** : 66 métiers extraits d'une source externe générée automatiquement (non ISPM, registre `SRC-GUIDE-FILIERES-GENERE`, statut **externe**) via `backend/scripts/preparer_candidats_metiers.py --ecrire-corpus` → fusionnés dans `backend/data/metiers.json` et `Parcours.debouches` des 16 parcours (décision produit du 28/08 : fusion telle quelle, sans relecture individuelle des 66 intitulés — même statut de provenance que les matières OCR, déjà en production). Restent incomplets : compétences, passerelles, matières de licence propres à TEE | — |
| ~~DATA-2~~ ✅ | Registre des sources | Structure (titre, URL, date de consultation, statut officiel/institutionnel/externe, données extraites, limites) — absent d'EXAM-S2 **[NOUVEAU]** — `backend/src/sources.py` + `backend/data/registre_sources.json` (7 entrées, dont une divergence de source explicitement documentée sur le sigle CAA, une expansion non confirmée pour AEE, et une source externe générée automatiquement pour les débouchés métiers) | DATA-1 |
| ~~DATA-3~~ (partiel) | Corpus structuré + provenance | Peupler `DocumentSource`/`Mention`/`Parcours` (SETUP-5) en liant chaque entrée à son entrée du registre (DATA-2) **[ADAPT]** ingestion RAG-1 d'EXAM-S2 — `source_id` posé sur tous les modèles, `verifier_provenance()` garantit qu'aucune référence n'est orpheline (testé sur les données réelles). **Incomplet** : compétences et passerelles pas encore peuplées (dépend de DATA-1) — matières et métiers/débouchés sont désormais tous les deux en production | SETUP-5, DATA-2 |
| ~~DATA-4~~ (partiel) | Questionnaire d'enquête | Rédaction et diffusion (étudiants + professionnels), lancé dès la première heure du hackathon **[NOUVEAU]** — **rédigé** : `backend/data/enquete/questionnaire.md` (version de référence, 21 questions, deux populations, consentement explicite, aucune donnée sensible) et `generer_google_form.gs`, script Apps Script qui génère le formulaire Google à l'identique depuis cette référence. **Diffusée** : 86 réponses reçues (voir DATA-5) | — |
| ~~DATA-5~~ ✅ | Registre de collecte de l'enquête | Populations visées, période, nombre de réponses reçues/retenues/écartées, texte de consentement, procédure d'anonymisation, biais constatés **[NOUVEAU]** — `backend/data/enquete/registre_collecte.md`, généré automatiquement par `backend/scripts/preparer_jeu_test_reel.py` : 86 réponses (71 étudiants, 15 professionnels), 79/86 rattachables à un parcours ou une mention connue, 62/86 à étiquette jugée fiable (satisfaction ≥ 3/5). Biais et limites (échantillon concentré sur quelques mentions, risque de ré-identification par petits effectifs) documentés dans le fichier généré, pas seulement déclarés ici | DATA-4 |
| ~~DATA-6~~ ✅ | Génération de profils synthétiques | Méthode, hypothèses, biais introduits, contrôles de cohérence documentés **[NOUVEAU]** — `backend/src/ml/donnees_synthetiques.py` + `archetypes.py` (16 archétypes ancrés sur les vraies descriptions de parcours, DATA-1) ; 800 profils dans `backend/data/ml/profils_synthetiques.json` | SETUP-4 |
| ~~DATA-7~~ (partiel) | Montage du jeu de données ML | Synthèse pour l'entraînement, enquête pour validation/test **[NOUVEAU]** — `backend/src/enquete_import.py` produit `reponses_orientia.json` (14 réponses exploitables, tous champs déclarés, jeu de test ML-7). **Reste** : élargir la collecte, 5 parcours sur 16 seulement étant représentés | DATA-5, DATA-6 |
| ~~DATA-8~~ ✅ | Anonymisation des réponses | Retrait des identifiants avant livraison **[ADAPT]** — aucune donnée identifiante n'existe à la source (le questionnaire n'en demandait aucune) ; à l'import, les horodatages sont écartés et les identifiants sont séquentiels, sans lien avec l'ordre de soumission | DATA-4 |

## 🕸️ Ontologie — IA symbolique

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~ONTO-1~~ ✅ | Schéma d'entités/relations | Étudiant, Formation, Mention, Parcours, Matière, Compétence, Prérequis, Métier, CentreIntérêt + relations `enseigne`/`développe`/`prépareA`/`nécessite`/`possède`/`préfère`/`estRequisePour` **[NOUVEAU]** — `backend/src/ontologie.py` (`SCHEMA_RELATIONS`, `relation_valide()`), purement déclaratif, ne dépend pas du corpus réel. Deux relations ajoutées au-delà des exemples du §12 (`appartientA`, `passerelleVers`) pour couvrir `Parcours.mention_id` et `Parcours.passerelles`, qui échappaient sinon au graphe **et** au contrôle de références orphelines d'ONTO-4 ; `Étudiant`/`CentreIntérêt` restent délibérément sans nœud (le graphe porte l'offre de formation, le profil vit dans `ProfilCandidat`), c'est documenté dans le module | DATA-1 |
| ~~ONTO-2~~ ✅ | Construction du graphe | Peuplement depuis le corpus structuré **[NOUVEAU]** — `backend/src/graphe.py` (`construire_graphe()`, NetworkX) ; une référence orpheline (id cité par un `Parcours` mais absent du corpus) n'aboutit jamais à une arête fantôme, elle est signalée par ONTO-4. Ajout mineur à `models.Competence` (`metiers_requis`) pour porter la relation `estRequisePour`, additif et rétrocompatible. Sur le corpus réel, le graphe compte désormais **583 nœuds et 1204 arêtes** (1106 `enseigne`, 66 `prepareA`, 16 `necessite`, 16 `appartientA`) depuis la collecte des matières (DATA-1, 27/08) et la fusion des débouchés (DATA-1, 28/08) ; compétences restent à collecter, `estRequisePour` toujours vide | ONTO-1, DATA-3 |
| ~~ONTO-3~~ ✅ | Outil `verifier_prerequis` | Requête de graphe déterministe (pas de LLM), exposée à l'agent comme outil **[NOUVEAU]** — `graphe.prerequis_du_parcours()`, branché dans `tools.verifier_prerequis` à la place du filtrage direct du corpus (comportement externe inchangé, tests existants toujours au vert) | ONTO-2 |
| ~~ONTO-4~~ ✅ | Outil `detecter_incoherences` | Ex. compétence requise sans prérequis satisfaisable, parcours sans débouché renseigné **[NOUVEAU]** — `graphe.detecter_incoherences()` + nouvel outil d'agent `tools.detecter_incoherences` (9e outil) : référence orpheline, parcours sans débouché, compétence requise inaccessible ou dont l'accès n'est pas vérifiable | ONTO-2 |
| ~~ONTO-5~~ ✅ | Raisonnement multiétape | Chemin Compétence → Parcours → Métier pour enrichir `expliquer_recommandation` **[NOUVEAU]** — `graphe.chemin_competence_parcours_metier()`, ajouté au champ `raisonnement_graphe` de `tools.expliquer_recommandation` ; dégrade en liste vide si le graphe n'est pas initialisé ou si le parcours n'a pas encore de compétence/débouché collecté, jamais une condition bloquante | ONTO-2, ML-8 |
| ~~ONTO-6~~ ✅ | Preuve d'apport de l'ontologie | Comparaison avec/sans graphe sur quelques cas — le sujet exige que l'apport soit démontré, pas seulement présent **[NOUVEAU]** — `backend/tests/eval_ontologie.py` → `backend/tests/eval_results_ontologie.json` (`cd backend && python -m tests.eval_ontologie`). Les deux mesures passent par les **outils réellement exposés à l'agent** (`src.tools`), pas par `src.graphe` en direct : une preuve qui court-circuite le chemin de production ne prouve rien. Sur le corpus réel ISPM, `detecter_incoherences` remonte désormais **0 constat** (16 « parcours sans débouché » avant la fusion des métiers, DATA-1/28-08 ; 0 contradiction réelle avant comme après) — la distinction `donnee_manquante`/contradiction est portée par l'outil lui-même, pas seulement par l'eval, et reste testée sur le corpus jouet (`test_graphe.py`) pour ne pas dépendre d'un état du corpus réel amené à changer. Le raisonnement multiétape (ONTO-5) est mesuré comme l'écart entre la sortie de `expliquer_recommandation` avant/après le graphe, sur un corpus jouet complet | ONTO-3, ONTO-4, EVAL-4 |

## 🤖 Machine Learning — absent d'EXAM-S2, à construire intégralement

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~ML-1~~ ✅ | Analyse exploratoire | Distributions, corrélations, déséquilibres sur synthèse + enquête **[NOUVEAU]** — `backend/src/ml/exploration.py` (calcul **testé**, `test_exploration.py`) + notebook livrable `backend/notebooks/01_analyse_exploratoire.ipynb`, exécuté avec ses sorties. Le calcul vit hors du notebook pour être rejouable en CI. Deux trous mesurés et nommés : `activites_projets` et `serie_bac` ne sont **jamais générés** (0 %), deux champs listés au §5 — le second rend les règles d'admission (ML-10) inertes sur ce jeu. Contrôle anti-fuite automatique (`pouvoir_discriminant_par_champ`) : 0 % de traits exclusifs à une classe, la fuite `environnement_travail_recherche` corrigée est désormais surveillée à chaque exécution. Volet enquête toujours bloqué sur DATA-7 | DATA-7 |
| ~~ML-2~~ ✅ | Nettoyage & split | Sélection de variables, stratégie train (synthèse) / test (enquête) **[NOUVEAU]** — `backend/src/ml/features.py` (vectorisation multi-hot sur vocabulaire contrôlé) + `entrainement.separer_train_test()` (split stratifié) | ML-1 |
| ~~ML-3~~ ✅ | Modèle de référence | Baseline simple (ex. régression logistique / kNN) pour classification de parcours **[NOUVEAU]** — régression logistique multinomiale, `backend/src/ml/entrainement.py` | ML-2 |
| ~~ML-4~~ ✅ | Second modèle comparé | Ex. LightGBM en ranking/score d'adéquation, ou clustering de profils similaires **[NOUVEAU]** — forêt aléatoire plutôt que LightGBM (pas de dépendance de boosting supplémentaire pour un jeu de cette taille), score d'adéquation = probabilité par classe | ML-2 |
| ~~ML-5~~ ✅ | Métriques adaptées | F1, matrice de confusion, ROC/PR-AUC, Top-k, MRR, NDCG, calibration **[ADAPT]** — `backend/src/ml/evaluation.py`. **MRR**, **NDCG@3**, rang médian, **PR-AUC macro**, **matrice de confusion complète**, et une calibration au sens propre (ECE, Brier, et surtout l'**écart signé** : l'ECE est une valeur absolue et ne dit pas dans quel sens le modèle se trompe — une première alerte affirmait « sur-confiance » sur un modèle en réalité sous-confiant). Deux stabilités distinctes : variance entre graines, et **stabilité des recommandations** au sens du §7, mesurée sur la sélection réellement présentée | ML-3, ML-4 |
| ~~ML-6~~ ✅ | Analyse des erreurs et biais | **[ADAPT méthodologie]** — trois défauts réels trouvés et corrigés, chacun mesuré avant et après. (1) Fuite `environnement_travail_recherche` : 100 % d'exactitude pour tout modèle, corrigée et désormais **surveillée automatiquement** (`exploration.pouvoir_discriminant_par_champ`, 0 % de traits exclusifs). (2) **Sous-confiance de 12 points** (exactitude 99,5 % annoncée 87,9 %) : calibration isotonique bornée → **ECE 0,120 → 0,033**, Brier 0,034 → 0,016, exactitude inchangée, garde-fou d'escalade préservé. (3) **Top-3 instable à 34 %** : cause trouvée — le rang 2 médian est à 2,0 % et le rang 3 à 1,1 %, c'était du bruit présenté comme une recommandation. Corrigé en ne proposant que les parcours atteignant 20 % du score de tête (seuil balayé, rappel 100 % à tous les seuils) → **stabilité 66 % (avant les deux correctifs) → 96 %**. Sur le seul modèle calibré, le top-3 fixe est à 86 % : les deux chiffres ne sont pas mesurés dans la même condition, et le rapport publie les deux | ML-5 |
| ~~ML-7~~ ✅ | Généralisation synthèse → réel | Mesurer la capacité du modèle entraîné sur profils synthétiques à généraliser aux réponses d'enquête réelles **[NOUVEAU]** — `eval_ml.py` mesure désormais `generalisation_reelle_ml_7` : le modèle de production (`ml.outils.analyser_profil`, sans substitution — ces 79 profils réels n'ont jamais été vus à l'entraînement) appliqué au jeu réel gelé (DATA-7). **Résultat mesuré, pas favorable** : top-1 à 1,3 % (contre 99,5 % sur synthèse), **56/79 profils jugés inexploitables** par le garde-fou d'exploitabilité (ML-9) — l'enquête ne recueille que 1 à 4 matières préférées et une note combinée maths/info, très en-deçà des 5 dimensions déclarées par un profil synthétique. Écart de généralisation réel et documenté (`backend/tests/eval_results_ml.json`), pas supposé | ML-5, DATA-7 |
| ~~ML-8~~ ✅ | Empaquetage en outils appelables | `analyser_profil()`, `classer_parcours()`, `calculer_adequation()`, `identifier_points_forts()` **[ADAPT]** principe « le modèle ne doit jamais rester isolé dans un notebook » — `backend/src/ml/outils.py`, les 4 signatures exactes citées par le sujet, justifications par classe via les coefficients de la régression logistique | ML-4 |
| ~~ML-9~~ ✅ | Vocabulaire d'entrée ouvert + refus d'affirmer sur un profil non exploité | **Défaut trouvé à l'auto-évaluation du bloc ML** : le vocabulaire fermé ignorait *silencieusement* tout terme hors liste — un candidat déclarant « maths », « info », « Python » produisait un vecteur entièrement nul, et le système émettait malgré tout un score d'apparence normale. Corrigé par `src/ml/vocabulaire.py` : normalisation → alias curés → repli sémantique (embeddings ONNX déjà présents pour le RAG, seuil 0,50 mesuré). `features.analyser_couverture()` remonte les termes non reconnus, et `outils.analyser_profil()` met la confiance à 0 sur un profil non exploitable au lieu de présenter un classement trompeur | ML-2, ML-8 |
| ~~ML-10~~ ✅ | Modèle hybride : apprentissage statistique + règles d'admission | Approche explicitement valorisée par le sujet (§6, « modèle hybride combinant apprentissage statistique et règles »). **Défaut mesuré qu'elle corrige** : le modèle ne voit jamais `serie_bac` (absente de l'espace de features), il classait donc en tête des parcours inaccessibles au candidat — un profil « Bac A, intéressé par l'informatique » obtenait ses **4** premières recommandations parmi des parcours exigeant un Bac C/D/S. `src/ml/hybride.py` rétrograde les parcours inadmissibles derrière les accessibles, en s'appuyant sur les prérequis du graphe (ONTO-3) et la règle partagée `src/admission.py` (une seule implémentation avec `verifier_prerequis`, pour qu'elles ne divergent pas). Rétrograde sans masquer, ne touche jamais aux scores, et ne fait rien quand la série n'est pas déclarée. Vérifié bout-en-bout contre l'API réelle | ML-8, ONTO-3 |

## 📖 RAG — recherche documentaire (fortement réutilisable)

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~RAG-1~~ ✅ | Moteur RAG | Chunking par phrases, chevauchement borné, ChromaDB **en espace cosinus explicite**, `upsert` idempotent **[REUSE quasi intégral]** de `rag.py`, adapté au modèle générique `DocumentSource` (SETUP-5) en attendant `ArticleFormation` | DATA-3, SETUP-3 |
| ~~RAG-2~~ ✅ | Prompt de génération citée | Interdiction d'inventer une formation/règle d'admission, citations obligatoires, drapeau `incertain` **[ADAPT]** `PROMPT_RAG` d'EXAM-S2, réécrit pour le domaine orientation | RAG-1 |
| ~~RAG-3~~ ✅ | Contrôle anti-hallucination des sources | Retirer toute source citée absente des passages fournis **[REUSE tel quel]** | RAG-2 |
| ~~RAG-4~~ ✅ | Recherche hybride | BM25 lexical sans dépendance supplémentaire + recherche vectorielle Chroma, fusionnés par **Reciprocal Rank Fusion** (les scores bruts BM25/cosinus sont incomparables). Une porte lexicale exige deux termes significatifs ou un sigle exact afin de préserver le silence hors corpus ; normalisation des accents et pluriels simples. Le graphe n'est volontairement pas injecté dans le retrieval tant que compétences/débouchés sont incomplets : ses requêtes déterministes restent disponibles via les outils de l'agent. Mesure avant/après sur le jeu RAG-6 figé : rappel **0,75 → 1,00**, précision **0,48 → 0,67**, silence hors corpus maintenu à **1,00** | RAG-1, ONTO-2 |
| ~~RAG-5~~ ✅ | Calibration du seuil et de k | Balayage mesuré (pas de valeur supposée) **[REUSE méthodologie]** — `backend/tests/calibrer_seuil_rag.py` → `eval_results_rag_calibration.json` (`cd backend && python -m tests.calibrer_seuil_rag`), 12 questions à source connue + 4 hors corpus, 30 couples (seuil, k). **Les valeurs héritées (0,75 / k=8) étaient franchement mauvaises ici** : rappel 1,00 mais **silence nul** sur les questions hors corpus (le RAG répondait toujours, ce qui invite le modèle à broder, §16) et précision 0,17. Retenu **seuil 0,56 / k=5** : rappel 0,75, précision 0,48, **silence 1,00**. Le silence prime — ne rien trouver est un comportement exigé au §9, répondre hors sujet est le mode d'échec dangereux | RAG-1 |
| ~~RAG-6~~ ✅ | Jeu de test RAG | 16 cas figés avant RAG-4 dans `backend/tests/eval_rag.json` : 12 questions avec source attendue et 4 volontairement hors corpus. `backend/tests/eval_rag.py` produit deux résultats séparés et reproductibles (`eval_results_rag_vectoriel.json`, `eval_results_rag_hybride.json`) sans appel LLM. Les citations inventées restent couvertes indépendamment par `test_rag.py` | RAG-2 |

## 🛠️ Agent & outils

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~AGT-1~~ ✅ | Boucle agent bornée | Function calling natif, limite d'itérations, séparation stricte modèle/code **[REUSE quasi tel quel]** `agent.py` — schéma de sortie (`RecommandationDecision`) et prompt système réécrits pour le domaine orientation ; vérifié à la fois sans réseau (LLM simulé) et contre l'API Gemini réelle (`pytest -m reseau`, `test_agent_reel_recommande_un_parcours_coherent`) | SETUP-4, SETUP-3 |
| ~~AGT-2~~ ✅ | Spécification des outils du domaine | `rechercher_formation`, `comparer_parcours`, `analyser_profil_ml`, `calculer_score_adequation`, `verifier_prerequis`, `rechercher_competences`, `identifier_debouches`, `expliquer_recommandation` **[ADAPT]** structure `OUTILS`/`TOOL_REGISTRY` — les 8 outils cités par le sujet, plus **`detecter_incoherences`** (9e, apport ONTO-4), `OUTILS` et `TOOL_REGISTRY` vérifiés cohérents par test. `verifier_prerequis` interroge désormais **le graphe** (ONTO-3), pas les données structurées. `identifier_debouches` dispose désormais de 66 métiers (DATA-1, 28/08, source externe) ; `rechercher_competences` reste data-limité : les compétences ne sont toujours pas collectées | DATA-3, ML-8, ONTO-3 |
| ~~AGT-3~~ ✅ | Validation & exécution des outils | `valider_parametres()`/`executer_outil()` **[REUSE tel quel]** | AGT-2 |
| AGT-4 (partiel) | Politique de refus / incertitude / renvoi | Remplace le concept « action sensible → validation humaine » d'EXAM-S2 par « confiance insuffisante → escalade vers un conseiller pédagogique » **[ADAPT concept]** — amorcé directement dans `agent.py` (seuil de confiance force `escalade_conseiller`) ; la version complète « le code contresigne la décision » (recoupement systématique avec les règles pédagogiques, gestion de `renvoi_administration`) attend l'orchestrateur | AGT-1, ORCH-2 |
| ~~AGT-5~~ ✅ | Séparation des sources dans la réponse | Prompt système imposant de distinguer résultat ML / info documentaire / règle pédagogique / texte généré — exigence explicite du sujet ORIENT'IA, absente d'EXAM-S2 **[NOUVEAU]** | AGT-1 |
| AGT-6 | Traçabilité des réponses issues des outils structurés | **Défaut trouvé à l'évaluation post-fusion** (voir `backend/tests/eval_analyse.md`, cas EVAL-17) : `agent._appliquer_controles_deterministes()` ne conserve que les sources présentes dans le contexte RAG, filtre écrit quand le RAG était le seul chemin vers une information. Depuis le bloc Ontologie, une réponse fondée sur les outils structurés (`identifier_debouches`, `verifier_prerequis` sur graphe) ne peut plus citer sa source, pourtant disponible (`Parcours.source_id`, rattaché au registre DATA-2). **Correctif** : faire remonter `source_id` dans les retours d'outils (`tools._fiche_parcours` et apparentés), puis élargir l'ensemble des sources autorisées à celles réellement retournées par les outils appelés | AGT-5, ONTO-3 |
| ~~AGT-7~~ ✅ | Cohérence entre la prose de l'agent et le classement du modèle | **Défaut trouvé en vérifiant ML-9, reproduit 2 fois sur 2** : sur un profil informatique, `parcours_recommandes[0]` vaut IGGLIA (0,54) mais l'explication en prose annonce « le modèle recommande en priorité ESIIA » (0,11). L'agent narre à partir des passages RAG plutôt que du classement de l'outil. Le barème note explicitement « cohérence entre le modèle ML et la réponse finale ». **Corrigé** : `agent._verifier_coherence_prose_classement()` annexe une note `[Contrôle automatique]` rappelant le classement réel quand la prose **omet** le parcours le mieux classé tout en en citant d'autres. Le critère est l'omission, pas le rang de citation — « contrairement à ESIIA, IGGLIA convient mieux » est légitime. Frontières de mot obligatoires (`EMP` correspondrait sinon à « emploi »), vérifié sans faux positif sur tout le corpus réel. Confirmé contre l'API réelle | AGT-5, ML-8 |

## 🔗 Orchestrateur

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~ORCH-1~~ ✅ | Pipeline complet | Garde-fous → RAG → agent → sortie structurée **[REUSE]** squelette `orchestrator.py` : « une seule étape bloquante (les garde-fous), le reste dégrade proprement » — vérifié par appel réel à l'API (`curl` sur `/orientation/traiter`), pas seulement en mocké | AGT-1, RAG-2, ML-8, SEC-1 |
| ~~ORCH-2~~ ✅ | Règles métier déterministes | Sources recoupées avec les passages réellement fournis, confiance plafonnée en cas de dégradation **[REUSE]** `_appliquer_plafond_de_confiance` — le recoupement des sources et le seuil de confiance vivent déjà dans `agent.py` (AGT-1), l'orchestrateur ajoute le plafonnement en cas de dégradation *pipeline* (RAG en échec, budget dépassé) | ORCH-1 |
| ~~ORCH-3~~ ✅ | Gestion des échecs | Timeout LLM, sortie non conforme, budget de temps global, dégradation propre (jamais d'erreur nue) **[REUSE]** budget orchestrateur — `_decision_repli()` (équivalent `reponse_erreur_controlee`) et `_etape_optionnelle()` écrits dans `orchestrator.py`. **Défaut trouvé en audit et corrigé** : l'orchestrateur ne rattrapait que `LLMError` autour de l'agent, si bien qu'une réponse finale au JSON valide mais non conforme au schéma faisait remonter une `ValidationError` jusqu'à FastAPI — un HTTP 500 nu, alors que ce module, `api.traiter()` et ce ticket promettaient tous trois l'inverse. La promesse « ne lève jamais » ne tolère pas une liste d'exceptions connues : c'est la sortie d'un modèle de langage. `sortie.generer_avec_retry()` était crédité ici sans jamais avoir été branché — il reste disponible mais n'est pas dans le chemin | ORCH-1 |
| ~~ORCH-4~~ ✅ | Endpoints FastAPI | `POST /orientation/traiter`, `GET /observabilite/traces`, `GET /health` **[ADAPT]** `api.py` — pas d'équivalent `/tickets/valider` : `tools.OUTILS_SENSIBLES` est vide (aucun outil sensible dans le périmètre actuel) | ORCH-1 |
| ~~ORCH-5~~ ✅ | Le budget de temps ne borne pas la durée totale | **Défaut observé en testant l'interface dans un navigateur réel** : une demande a mis 208 s côté serveur, dépassant le délai du client (180 s) — l'utilisateur voyait une erreur réseau alors que le backend avait correctement dégradé et produit une réponse valide. Cause : `_budget_epuise()` était vérifié **avant** de lancer une étape, sans interrompre une étape déjà partie. L'écart était plus large qu'estimé au départ : un appel LLM peut atteindre **138 s** à lui seul (4 tentatives × 30 s + 3 reprises de 6 s), donc **un seul** appel dépassait déjà le budget de 120 s, et le pire cas de la boucle agent était de **5 × 138 s = 690 s**. **Corrigé** : `traiter_demande()` installe une échéance isolée par requête (`limiter_temps_llm`, `ContextVar` — correct même quand FastAPI sert plusieurs requêtes dans des threads distincts) ; `llm_client` borne avec le temps restant le verrou de débit, l'attente de lissage, le timeout HTTP de chaque tentative et les pauses de reprise. `BudgetTempsDepasse` hérite de `LLMError`, donc la dégradation existante s'applique sans changement | ORCH-3 |

## 🛡️ Sécurité & garde-fous

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~SEC-1~~ ✅ | Détection d'injection à deux couches | Mots-clés + vérification LLM indépendante **[REUSE quasi tel quel]** `guardrails.py` — mécanisme générique, escalade métier spécifique volontairement non reprise (dépend du schéma de décision, SETUP-4) | SETUP-3 |
| ~~SEC-2~~ ✅ | Masquage des données sensibles | Avant toute écriture de log **[REUSE]** `masquer_objet`/`masquer_donnees_sensibles`, branché dans `observability._ecrire_jsonl()` | OBS-1 |
| ~~SEC-3~~ ✅ | Détecteur de critères discriminatoires | Contrôle déterministe : aucun facteur sensible (genre, origine, âge…) ne doit influencer la recommandation, même déclaré **[NOUVEAU]**, exigence explicite du sujet — défense structurelle en premier lieu (le vocabulaire ML ne contient aucune de ces dimensions, testé), `securite.detecter_criteres_discriminatoires()` en filet de sécurité sur le texte généré | ML-8, ORCH-2 |
| ~~SEC-4~~ ✅ | Refus de profilage psychologique | Contrôle de prompt + contrôle déterministe : n'utiliser que les préférences déclarées explicitement, jamais une inférence sur le style d'écriture **[NOUVEAU]**, spécifique à ORIENT'IA — consigne explicite dans `PROMPT_SYSTEME_AGENT`, confirmée en réel sur l'exemple adverse du sujet lui-même (« Analyse ma personnalité... ») ; `securite.detecter_profilage_psychologique()` en filet de sécurité | AGT-1 |
| ~~SEC-5~~ ✅ | Mention obligatoire dans l'interface | « ORIENT'IA... ne remplace ni l'avis d'un conseiller pédagogique ni une décision officielle d'admission » **[NOUVEAU]** — `src.config.MENTION_OBLIGATOIRE` reste la source unique, **lue en direct** par le frontend. Défaut corrigé après audit : les deux emplacements étaient conditionnés à une réponse de `GET /health`, donc la mention **disparaissait quand l'API était injoignable** — précisément l'état affiché par la barre latérale. Elle est maintenant inconditionnelle, avec un repli local explicitement signalé si `src` est hors d'atteinte. Non-régression : `test_frontend.py` | FE-1 |
| ~~SEC-6~~ ✅ | Jeu de test sécurité/biais | 3 cas prompt injection, 2 cas biais, 2 cas provenance/refus de profilage **[ADAPT]** méthodologie EXAM-S2 (attaques reformulées + cas légitimes de contrôle) — couvert de bout en bout dans `test_orchestrator.py` (cas biais/profilage) et `test_guardrails.py`/`test_orchestrator.py` (injection) ; le jeu de 32 cas complet exigé par le sujet reste EVAL-1 | SEC-1, SEC-3, SEC-4 |

## 📊 Observabilité

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~OBS-1~~ ✅ | Traces JSONL | Question, profil construit, passages récupérés + scores, outils appelés, I/O ML, réponse finale, latence, erreurs **[REUSE]** `observability.py`, `log_trace()` généralisé (description/contexte/décision + champs libres) pour ne pas dépendre d'un schéma métier pas encore écrit | ORCH-1 |
| ~~OBS-2~~ ✅ | Log des appels LLM bruts | Prompts/réponses complets, par étape **[REUSE tel quel]** `log_llm_call`/hooks | SETUP-3 |
| ~~OBS-3~~ ✅ | Estimation de coût | **[REUSE tel quel]** `estimer_cout` | OBS-2 |
| ~~OBS-4~~ ✅ | Endpoint de lecture des traces | Pour le dashboard **[REUSE tel quel]** `GET /observabilite/traces` — exposé dans `backend/src/api.py`, avec limite validée de 1 à 500, réponse vide contrôlée et tests HTTP. Les fonctions de lecture sont fournies par `observability.py`. | OBS-1, ORCH-4 |

## ✅ Évaluation

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~EVAL-1~~ ✅ | Jeu de 32 cas de test | 8 catégories imposées (factuel, comparaisons, profils→ML, multi-sources, info absente, profils ambigus, sécurité/injection, biais, provenance/refus profilage) **[NOUVEAU contenu]**, mécanique de restitution **[REUSE]** `eval_system.py` → `eval_results.json` — `backend/tests/eval_dataset.json`, minimums par catégorie vérifiés par un test | ORCH-4 |
| ~~EVAL-2~~ ✅ | Évaluation ML | Métriques de ML-5 exécutées sur le jeu de test **[ADAPT]** `evaluer_classification`, généralisé au ranking/score — les métriques quantitatives (F1, top-3, calibration, stabilité) restent celles d'`eval_ml.py` sur le split synthétique (les 32 cas n'ont pas d'étiquette « bon parcours », ce sont des cas comportementaux) ; les cas `profils_ml` d'EVAL-1 vérifient en plus, en conditions réelles, que le modèle est effectivement consulté et fonde la décision | ML-5, EVAL-1 |
| ~~EVAL-3~~ ✅ | Évaluation RAG | Rappel des sources, précision des sources récupérées et détection hors-corpus sur le jeu dédié RAG-6 ; comparaison vectoriel/hybride enregistrée. Le chemin génératif reste vérifié par les catégories factuelles/comparaisons d'EVAL-1 et le garde-fou déterministe anti-citation inventée par `test_rag.py` | RAG-6, EVAL-1 |
| ~~EVAL-4~~ ✅ | Évaluation système complet | Cohérence ML/réponse finale, latence, robustesse, sécurité **[ADAPT]** `evaluer_scenarios_obligatoires` étendu aux 8 catégories du sujet — `eval_system.py`, 30/32 (93,75 %), latence moyenne 9,1 s, les 2 échecs restants root-causés à une instabilité réseau réelle et reproduite (pas un défaut de code), voir `backend/tests/eval_analyse.md` | EVAL-2, EVAL-3, SEC-6 |
| ~~EVAL-5~~ ✅ | Script d'évaluation unique | Un seul run produit `eval_results.json`, livrable distinct exigé par le sujet **[REUSE pattern]** — `python -m backend.tests.eval_system` | EVAL-2, EVAL-3, EVAL-4 |
| ~~EVAL-6~~ ✅ | Analyse des erreurs et limites | Section rapport dédiée **[REUSE méthode de rédaction]** — `backend/tests/eval_analyse.md` : 3 défauts de code trouvés et corrigés (action « information » manquante, boucle sur `expliquer_recommandation`, garde-fou ML étendu aux escalades), 2 défauts de jeu de test corrigés, limites honnêtes documentées (pas de mémoire de conversation, ML validé seulement en synthétique, dépendance réseau) | EVAL-5 |

## 🖥️ Frontend

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~FE-1~~ ✅ | Interface, front-office et back-office | **[REUSE]** puis restructuré après audit — `frontend/` éclaté en `noyau.py` (chemin d'import, client API, mention réglementaire), `front_office.py` (1 page candidat) et `back_office.py` (5 vues admin), routés par une **seule** `st.navigation` en sections. Conditionner la liste des pages à un sélecteur cassait les liens profonds (« page does not seem to exist » sur `/page_mesures`, vérifié au navigateur) : le contrôle d'accès vit donc dans les pages (`exiger_acces_admin`, code facultatif `ORIENTIA_ADMIN_CODE`, ouvert et annoncé comme tel sinon). Client API avec `raise_for_status` — sans lui un 500 était lu comme une liste de traces et produisait un `AttributeError` brut à l'écran | ORCH-4 |
| ~~FE-2~~ ✅ | Collecte progressive du profil | Formulaire 6 champs, tous facultatifs — un profil incomplet est un cas normal que le système traite en posant des questions **[NOUVEAU]**. Les termes ne sont pas normalisés côté interface : `src.ml.vocabulaire` les résout et **signale ce qu'il n'a pas reconnu**, nettoyer ici masquerait ce diagnostic. 7 scénarios de démonstration, dont trois qui exercent les garde-fous (hors-vocabulaire, admissibilité, injection) | SETUP-4 |
| ~~FE-3~~ ✅ | Affichage de la recommandation | Parcours, score, sources, incertitude **[ADAPT]** — vocabulaire interne traduit en langage candidat (`escalade_conseiller` → « à voir avec un conseiller pédagogique »). **Un score déclaré non informatif n'est jamais affiché en gros chiffre** : le mettre en forme comme un score fiable serait un mensonge par mise en page. Badge « admissibilité à vérifier » sur un parcours rétrogradé par les règles d'admission (ML-10). Marqueurs lus depuis le backend (`AVERTISSEMENT_NON_EXPLOITABLE`, `MARQUEUR_REGLE_ADMISSION`), pas devinés | FE-1, ORCH-1 |
| ~~FE-4~~ ✅ | Explorateur de corpus et de graphe | Visualisation des relations **[NOUVEAU]** — graphe DOT filtrable par type d'entité, étiquettes échappées (un guillemet dans un nom cassait tout le graphe). L'onglet dégradait auparavant en « Graphe indisponible » sans diagnostic parce que `src` n'était pas importable depuis la racine : corrigé dans `noyau.py`, et toute erreur restante est désormais affichée avec son type | ONTO-2 |
| ~~FE-5~~ ✅ | Migration frontend vers Next.js — front-office chatbot | `frontend-next/` **[NOUVEAU]**, `frontend/` (Streamlit) conservé tel quel, les deux coexistent. Espace candidat (`app/chat/`) reconstruit en vraie conversation (historique + profil accumulés côté client dans `sessionStorage`, renvoyés à chaque tour au même `POST /orientation/traiter` **toujours stateless côté serveur** — pas de mémoire de conversation ajoutée, cf. limite déjà documentée dans `eval_analyse.md`). Port fidèle de `front_office.py` : mention obligatoire affichée inconditionnellement, score jamais affiché si `AVERTISSEMENT_NON_EXPLOITABLE` détecté, badge « Admissibilité à vérifier » si `MARQUEUR_REGLE_ADMISSION` détecté (ces deux marqueurs sont désormais aussi exposés par `GET /health`, pour un frontend qui ne peut pas les lire par import Python), 7 scénarios de démo, profil édité explicitement (jamais déduit du texte libre du chat — SEC-4) | FE-1, FE-2, FE-3 |
| ~~FE-6~~ ✅ | Migration frontend vers Next.js — backoffice admin | `app/admin/` **[NOUVEAU]**, visuellement et structurellement séparé du chat candidat. Auth par cookie signé (`jose`, HttpOnly) plutôt que `st.session_state` local à l'onglet — même secret `ORIENTIA_ADMIN_CODE`, même repli « accès ouvert, annoncé comme tel » si absent. 5 vues portées depuis `back_office.py`, dont un tableau de bord avec **évolution des stats dans le temps** (volume, latence, confiance, répartition des actions par jour/heure — nouveau, absent du backoffice Streamlit) et un graphe de connaissances interactif (`@xyflow/react` + `dagre`, remplace le rendu DOT/graphviz qui n'a pas d'équivalent React). Nouveaux endpoints backend en lecture seule (`backend/src/admin_api.py`, `/admin/*`) exposant en HTTP ce que `back_office.py` calculait par import Python direct — **ces endpoints restent ouverts, sans authentification côté API** (l'auth vit uniquement dans Next.js, cohérent avec le fait que le prototype ne manipule aucune donnée personnelle) ; point de vigilance si l'API est un jour exposée au-delà de `localhost` | FE-1, FE-4, OBS-4 |

## 📄 Livrables & documentation

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| DOC-1 | README | Architecture, choix techniques, instructions de lancement **[REUSE]** plan de structure d'EXAM-S2 | Tout |
| DOC-2 | Rapport technique synthétique | Couvrant les axes notés du sujet **[REUSE]** structure de `rapport-technique.md` | EVAL-6 |
| DOC-3 | Note de limites, biais et risques | **[REUSE]** section « Limites connues » d'EXAM-S2 comme gabarit | EVAL-6, ML-7 |
| DOC-4 | Checklist de remise | Les 14 livrables du sujet ORIENT'IA **[ADAPT]** checklist EXAM-S2 | Tout |
| DOC-5 | Vidéo de démo (3–5 min) | Système en fonctionnement, pas de diapositives commentées **[NOUVEAU]** | FE-3 |
| DOC-6 | Schéma d'architecture | Raffiner le schéma déjà esquissé dans la feuille de route avec les modules réellement codés | ORCH-1 |

---

## Ordre de priorité si le temps manque

Le sujet pèse le plus lourd sur ML (18 pts), évaluation bout-en-bout (14 pts), RAG
(12 pts) et agent/outils (12 pts) — 56 des 100 points. Ne jamais sacrifier ces blocs
au profit de l'ontologie ou du style du frontend.

**Ne jamais sacrifier** : ML-5/ML-7 (métriques + généralisation, sinon le cœur noté à
18 pts est vide), RAG-3 (anti-hallucination des sources), SEC-3/SEC-4 (biais et
profilage — exigences non négociables du sujet), EVAL-1/EVAL-5 (résultats mesurés
exigés comme livrable), DATA-4 (l'enquête ne peut pas être rattrapée après le jour 1).

**Sacrifiable en premier** : ONTO-4/ONTO-5 (raffinements de l'ontologie), les
extensions ontologiques supplémentaires du retrieval hybride, FE-4 (explorateur
de graphe), DOC-6 au-delà d'une version minimale.
