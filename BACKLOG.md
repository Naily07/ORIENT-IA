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
| ~~DATA-1~~ (partiel) | Collecte du corpus ISPM | Mentions, parcours, niveaux, prérequis d'admission, depuis le site officiel `ispm-edu.com` (accueil, filières, présentation, inscription) et une source tierce de recoupement (annuaire.mg) **[NOUVEAU]** — 6 mentions et 16 parcours réels chargés dans `backend/data/`. **Incomplet** : matières précises, compétences détaillées, débouchés métiers nommés et passerelles entre formations n'ont pas de source fiable identifiée en ligne — à compléter via brochures ou contact direct ISPM | — |
| ~~DATA-2~~ ✅ | Registre des sources | Structure (titre, URL, date de consultation, statut officiel/institutionnel/externe, données extraites, limites) — absent d'EXAM-S2 **[NOUVEAU]** — `backend/src/sources.py` + `backend/data/registre_sources.json` (5 entrées, dont une divergence de source explicitement documentée sur le sigle CAA et une expansion non confirmée pour AEE) | DATA-1 |
| ~~DATA-3~~ (partiel) | Corpus structuré + provenance | Peupler `DocumentSource`/`Mention`/`Parcours` (SETUP-5) en liant chaque entrée à son entrée du registre (DATA-2) **[ADAPT]** ingestion RAG-1 d'EXAM-S2 — `source_id` posé sur tous les modèles, `verifier_provenance()` garantit qu'aucune référence n'est orpheline (testé sur les données réelles). **Incomplet** : matières/compétences/métiers/passerelles pas encore peuplés (dépend de DATA-1) | SETUP-5, DATA-2 |
| DATA-4 | Questionnaire d'enquête | Rédaction et diffusion (étudiants + professionnels), lancé dès l'heure 1 du hackathon **[NOUVEAU]** | — |
| DATA-5 | Registre de collecte de l'enquête | Populations visées, période, nombre de réponses reçues/retenues/écartées, texte de consentement, procédure d'anonymisation, biais constatés **[NOUVEAU]** | DATA-4 |
| ~~DATA-6~~ ✅ | Génération de profils synthétiques | Méthode, hypothèses, biais introduits, contrôles de cohérence documentés **[NOUVEAU]** — `backend/src/ml/donnees_synthetiques.py` + `archetypes.py` (16 archétypes ancrés sur les vraies descriptions de parcours, DATA-1) ; 800 profils dans `backend/data/ml/profils_synthetiques.json` | SETUP-4 |
| DATA-7 | Montage du jeu de données ML | Synthèse pour l'entraînement, réponses d'enquête gelées en fin de J1 pour validation/test **[NOUVEAU]**, recommandation explicite du sujet — la partie synthèse existe (DATA-6) ; **bloqué** sur la partie enquête réelle (DATA-4/DATA-5), qui ne peut être menée que pendant le hackathon | DATA-5, DATA-6 |
| DATA-8 | Anonymisation des réponses | Retrait des identifiants directs avant livraison du jeu de données **[ADAPT]** `masquer_donnees_sensibles`/`masquer_objet` de `guardrails.py` comme base | DATA-4 |

## 🕸️ Ontologie — IA symbolique

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~ONTO-1~~ ✅ | Schéma d'entités/relations | Étudiant, Formation, Mention, Parcours, Matière, Compétence, Prérequis, Métier, CentreIntérêt + relations `enseigne`/`développe`/`prépareA`/`nécessite`/`possède`/`préfère`/`estRequisePour` **[NOUVEAU]** — `backend/src/ontologie.py` (`SCHEMA_RELATIONS`, `relation_valide()`), purement déclaratif, ne dépend pas du corpus réel. Deux relations ajoutées au-delà des exemples du §12 (`appartientA`, `passerelleVers`) pour couvrir `Parcours.mention_id` et `Parcours.passerelles`, qui échappaient sinon au graphe **et** au contrôle de références orphelines d'ONTO-4 ; `Étudiant`/`CentreIntérêt` restent délibérément sans nœud (le graphe porte l'offre de formation, le profil vit dans `ProfilCandidat`), c'est documenté dans le module | DATA-1 |
| ~~ONTO-2~~ ✅ | Construction du graphe | Peuplement depuis le corpus structuré **[NOUVEAU]** — `backend/src/graphe.py` (`construire_graphe()`, NetworkX) ; une référence orpheline (id cité par un `Parcours` mais absent du corpus) n'aboutit jamais à une arête fantôme, elle est signalée par ONTO-4. Ajout mineur à `models.Competence` (`metiers_requis`) pour porter la relation `estRequisePour`, additif et rétrocompatible. Sur le corpus réel (16 parcours, DATA-1) seules les arêtes `necessite` existent : matières/compétences/débouchés restent à collecter | ONTO-1, DATA-3 |
| ~~ONTO-3~~ ✅ | Outil `verifier_prerequis` | Requête de graphe déterministe (pas de LLM), exposée à l'agent comme outil **[NOUVEAU]** — `graphe.prerequis_du_parcours()`, branché dans `tools.verifier_prerequis` à la place du filtrage direct du corpus (comportement externe inchangé, tests existants toujours au vert) | ONTO-2 |
| ~~ONTO-4~~ ✅ | Outil `detecter_incoherences` | Ex. compétence requise sans prérequis satisfaisable, parcours sans débouché renseigné **[NOUVEAU]** — `graphe.detecter_incoherences()` + nouvel outil d'agent `tools.detecter_incoherences` (9e outil) : référence orpheline, parcours sans débouché, compétence requise inaccessible ou dont l'accès n'est pas vérifiable | ONTO-2 |
| ~~ONTO-5~~ ✅ | Raisonnement multiétape | Chemin Compétence → Parcours → Métier pour enrichir `expliquer_recommandation` **[NOUVEAU]** — `graphe.chemin_competence_parcours_metier()`, ajouté au champ `raisonnement_graphe` de `tools.expliquer_recommandation` ; dégrade en liste vide si le graphe n'est pas initialisé ou si le parcours n'a pas encore de compétence/débouché collecté, jamais une condition bloquante | ONTO-2, ML-8 |
| ~~ONTO-6~~ ✅ | Preuve d'apport de l'ontologie | Comparaison avec/sans graphe sur quelques cas — le sujet exige que l'apport soit démontré, pas seulement présent **[NOUVEAU]** — `backend/tests/eval_ontologie.py` → `backend/tests/eval_results_ontologie.json` (`cd backend && python -m tests.eval_ontologie`). Les deux mesures passent par les **outils réellement exposés à l'agent** (`src.tools`), pas par `src.graphe` en direct : une preuve qui court-circuite le chemin de production ne prouve rien. Sur le corpus réel ISPM, `detecter_incoherences` remonte 16 constats, tous marqués `donnee_manquante` (débouchés non collectés, DATA-1) et **0 contradiction réelle** — la distinction est portée par l'outil lui-même, pas seulement par l'eval. Le raisonnement multiétape (ONTO-5) est mesuré comme l'écart entre la sortie de `expliquer_recommandation` avant/après le graphe, sur un corpus jouet complet | ONTO-3, ONTO-4, EVAL-4 |

## 🤖 Machine Learning — absent d'EXAM-S2, à construire intégralement

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~ML-1~~ (partiel) | Analyse exploratoire | Distributions, corrélations, déséquilibres sur synthèse + enquête **[NOUVEAU]** — fait sur la synthèse (800 profils, 16 classes équilibrées par construction) ; volet enquête bloqué sur DATA-4/DATA-7 | DATA-7 |
| ~~ML-2~~ ✅ | Nettoyage & split | Sélection de variables, stratégie train (synthèse) / test (enquête) **[NOUVEAU]** — `backend/src/ml/features.py` (vectorisation multi-hot sur vocabulaire contrôlé) + `entrainement.separer_train_test()` (split stratifié) | ML-1 |
| ~~ML-3~~ ✅ | Modèle de référence | Baseline simple (ex. régression logistique / kNN) pour classification de parcours **[NOUVEAU]** — régression logistique multinomiale, `backend/src/ml/entrainement.py` | ML-2 |
| ~~ML-4~~ ✅ | Second modèle comparé | Ex. LightGBM en ranking/score d'adéquation, ou clustering de profils similaires **[NOUVEAU]** — forêt aléatoire plutôt que LightGBM (pas de dépendance de boosting supplémentaire pour un jeu de cette taille), score d'adéquation = probabilité par classe | ML-2 |
| ~~ML-5~~ ✅ | Métriques adaptées | F1, matrice de confusion, ROC/PR-AUC, Top-k, MRR, NDCG, calibration **[ADAPT]** méthodologie « recall ET précision par catégorie » d'`evaluer_classification` (EXAM-S2) — `backend/src/ml/evaluation.py` : F1 par classe et macro, top-3 accuracy, calibration (confiance correcte vs erronée) et stabilité (écart-type sur 5 seeds) | ML-3, ML-4 |
| ~~ML-6~~ ✅ | Analyse des erreurs et biais | Revue systématique des cas d'échec, dans l'esprit des revues de code d'EXAM-S2 (nommer les défauts plutôt que les masquer) **[ADAPT méthodologie]** — un défaut réel trouvé et corrigé pendant le calibrage : `environnement_travail_recherche` déterministe par archétype donnait 100 % d'exactitude à tout modèle, indépendamment du bruit ajouté ailleurs (fuite documentée dans `donnees_synthetiques.py`) ; corrigé, l'exactitude retombe à ~86-99 % selon le modèle | ML-5 |
| ML-7 | Généralisation synthèse → réel | Mesurer la capacité du modèle entraîné sur profils synthétiques à généraliser aux réponses d'enquête réelles **[NOUVEAU]**, aucun équivalent EXAM-S2 — **bloqué** : nécessite l'enquête réelle (DATA-4/DATA-5), qui ne peut être menée que pendant le hackathon lui-même (populations réelles, lancement heure 1) | ML-5, DATA-7 |
| ~~ML-8~~ ✅ | Empaquetage en outils appelables | `analyser_profil()`, `classer_parcours()`, `calculer_adequation()`, `identifier_points_forts()` **[ADAPT]** principe « le modèle ne doit jamais rester isolé dans un notebook » — `backend/src/ml/outils.py`, les 4 signatures exactes citées par le sujet, justifications par classe via les coefficients de la régression logistique | ML-4 |
| ~~ML-9~~ ✅ | Vocabulaire d'entrée ouvert + refus d'affirmer sur un profil non exploité | **Défaut trouvé à l'auto-évaluation du bloc ML** : le vocabulaire fermé ignorait *silencieusement* tout terme hors liste — un candidat déclarant « maths », « info », « Python » produisait un vecteur entièrement nul, et le système émettait malgré tout un score d'apparence normale. Corrigé par `src/ml/vocabulaire.py` : normalisation → alias curés → repli sémantique (embeddings ONNX déjà présents pour le RAG, seuil 0,50 mesuré). `features.analyser_couverture()` remonte les termes non reconnus, et `outils.analyser_profil()` met la confiance à 0 sur un profil non exploitable au lieu de présenter un classement trompeur | ML-2, ML-8 |

## 📖 RAG — recherche documentaire (fortement réutilisable)

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~RAG-1~~ ✅ | Moteur RAG | Chunking par phrases, chevauchement borné, ChromaDB **en espace cosinus explicite**, `upsert` idempotent **[REUSE quasi intégral]** de `rag.py`, adapté au modèle générique `DocumentSource` (SETUP-5) en attendant `ArticleFormation` | DATA-3, SETUP-3 |
| ~~RAG-2~~ ✅ | Prompt de génération citée | Interdiction d'inventer une formation/règle d'admission, citations obligatoires, drapeau `incertain` **[ADAPT]** `PROMPT_RAG` d'EXAM-S2, réécrit pour le domaine orientation | RAG-1 |
| ~~RAG-3~~ ✅ | Contrôle anti-hallucination des sources | Retirer toute source citée absente des passages fournis **[REUSE tel quel]** | RAG-2 |
| RAG-4 | Recherche hybride | Ajouter BM25 lexical et/ou requête ontologie (ONTO-2) en complément du vectoriel — EXAM-S2 n'a que le vectoriel, le sujet ORIENT'IA valorise l'hybridation **[NOUVEAU + REUSE base vectorielle]** | RAG-1, ONTO-2 |
| RAG-5 | Calibration du seuil et de k | Balayage mesuré (pas de valeur supposée) **[REUSE méthodologie]** `calibrer_seuil.py`, refaire la mesure sur le nouveau corpus — ne pas réutiliser les valeurs numériques d'EXAM-S2, propres au corpus IT | RAG-1 |
| RAG-6 | Jeu de test RAG | Questions avec source attendue + questions volontairement hors corpus **[ADAPT]** format `eval_rag.json` — amorcé : tests unitaires du chunking, du garde-fou anti-hallucination et de la recherche sur un mini-corpus jouet (`backend/tests/test_rag.py`) ; le vrai jeu de test sur le corpus ISPM reste à faire une fois DATA-3 disponible | RAG-2 |

## 🛠️ Agent & outils

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~AGT-1~~ ✅ | Boucle agent bornée | Function calling natif, limite d'itérations, séparation stricte modèle/code **[REUSE quasi tel quel]** `agent.py` — schéma de sortie (`RecommandationDecision`) et prompt système réécrits pour le domaine orientation ; vérifié à la fois sans réseau (LLM simulé) et contre l'API Gemini réelle (`pytest -m reseau`, `test_agent_reel_recommande_un_parcours_coherent`) | SETUP-4, SETUP-3 |
| ~~AGT-2~~ ✅ | Spécification des outils du domaine | `rechercher_formation`, `comparer_parcours`, `analyser_profil_ml`, `calculer_score_adequation`, `verifier_prerequis`, `rechercher_competences`, `identifier_debouches`, `expliquer_recommandation` **[ADAPT]** structure `OUTILS`/`TOOL_REGISTRY` — `verifier_prerequis` construit sur les données structurées (Mention/Parcours/Prerequis) plutôt que sur le graphe ONTO-2 (pas encore construit) ; `rechercher_competences`/`identifier_debouches` fonctionnels mais data-limités tant que DATA-1 (compétences/débouchés) n'est pas complété | DATA-3, ML-8, ONTO-3 |
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
| ~~ORCH-3~~ ✅ | Gestion des échecs | Timeout LLM, sortie non conforme, budget de temps global, dégradation propre (jamais d'erreur nue) **[REUSE]** `sortie.py` + budget orchestrateur — `_decision_repli()` (équivalent `reponse_erreur_controlee`) et `_etape_optionnelle()` avec budget de temps écrits dans `orchestrator.py` | ORCH-1 |
| ~~ORCH-4~~ ✅ | Endpoints FastAPI | `POST /orientation/traiter`, `GET /observabilite/traces`, `GET /health` **[ADAPT]** `api.py` — pas d'équivalent `/tickets/valider` : `tools.OUTILS_SENSIBLES` est vide (aucun outil sensible dans le périmètre actuel) | ORCH-1 |

## 🛡️ Sécurité & garde-fous

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| ~~SEC-1~~ ✅ | Détection d'injection à deux couches | Mots-clés + vérification LLM indépendante **[REUSE quasi tel quel]** `guardrails.py` — mécanisme générique, escalade métier spécifique volontairement non reprise (dépend du schéma de décision, SETUP-4) | SETUP-3 |
| ~~SEC-2~~ ✅ | Masquage des données sensibles | Avant toute écriture de log **[REUSE]** `masquer_objet`/`masquer_donnees_sensibles`, branché dans `observability._ecrire_jsonl()` | OBS-1 |
| ~~SEC-3~~ ✅ | Détecteur de critères discriminatoires | Contrôle déterministe : aucun facteur sensible (genre, origine, âge…) ne doit influencer la recommandation, même déclaré **[NOUVEAU]**, exigence explicite du sujet — défense structurelle en premier lieu (le vocabulaire ML ne contient aucune de ces dimensions, testé), `securite.detecter_criteres_discriminatoires()` en filet de sécurité sur le texte généré | ML-8, ORCH-2 |
| ~~SEC-4~~ ✅ | Refus de profilage psychologique | Contrôle de prompt + contrôle déterministe : n'utiliser que les préférences déclarées explicitement, jamais une inférence sur le style d'écriture **[NOUVEAU]**, spécifique à ORIENT'IA — consigne explicite dans `PROMPT_SYSTEME_AGENT`, confirmée en réel sur l'exemple adverse du sujet lui-même (« Analyse ma personnalité... ») ; `securite.detecter_profilage_psychologique()` en filet de sécurité | AGT-1 |
| SEC-5 (partiel) | Mention obligatoire dans l'interface | « ORIENT'IA... ne remplace ni l'avis d'un conseiller pédagogique ni une décision officielle d'admission » **[NOUVEAU]** — texte figé dans `config.MENTION_OBLIGATOIRE`, exposé par `GET /health` ; l'affichage réel dans l'interface attend FE-1 | FE-1 |
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
| EVAL-3 (partiel) | Évaluation RAG | Rappel@k, précision des citations, détection hors-corpus **[REUSE tel quel]** `evaluer_rag` — les catégories factuelles/comparaisons d'EVAL-1 démontrent le rappel et la précision des citations en conditions réelles (9/9), mais un jeu RAG dédié avec cas hors-corpus reste à construire (RAG-6) | RAG-6, EVAL-1 |
| ~~EVAL-4~~ ✅ | Évaluation système complet | Cohérence ML/réponse finale, latence, robustesse, sécurité **[ADAPT]** `evaluer_scenarios_obligatoires` étendu aux 8 catégories du sujet — `eval_system.py`, 30/32 (93,75 %), latence moyenne 9,7 s, les 2 échecs restants root-causés à une instabilité réseau réelle et reproduite (pas un défaut de code), voir `backend/tests/eval_analyse.md` | EVAL-2, EVAL-3, SEC-6 |
| ~~EVAL-5~~ ✅ | Script d'évaluation unique | Un seul run produit `eval_results.json`, livrable distinct exigé par le sujet **[REUSE pattern]** — `python -m backend.tests.eval_system` | EVAL-2, EVAL-3, EVAL-4 |
| ~~EVAL-6~~ ✅ | Analyse des erreurs et limites | Section rapport dédiée **[REUSE méthode de rédaction]** — `backend/tests/eval_analyse.md` : 3 défauts de code trouvés et corrigés (action « information » manquante, boucle sur `expliquer_recommandation`, garde-fou ML étendu aux escalades), 2 défauts de jeu de test corrigés, limites honnêtes documentées (pas de mémoire de conversation, ML validé seulement en synthétique, dépendance réseau) | EVAL-5 |

## 🖥️ Frontend

| ID | Tâche | Description | Dépendances |
|---|---|---|---|
| FE-1 | Squelette Streamlit | Chat, scénarios pré-remplis, onglet observabilité, explorateur de données **[REUSE]** `frontend/app.py` comme base, vocabulaire ticket → profil/recommandation | ORCH-4 |
| FE-2 | Collecte progressive du profil | Formulaire ou chat multi-tours pour construire le profil candidat **[NOUVEAU]**, spécifique à l'exigence « recueillir progressivement le profil » | SETUP-4 |
| FE-3 | Affichage de la recommandation | Parcours proposés, score, sources citées, incertitude déclarée **[ADAPT]** carte de décision d'EXAM-S2 | FE-1, ORCH-1 |
| FE-4 | Onglet explorateur de graphe | Visualisation des relations Compétence/Parcours/Métier, si ontologie construite **[NOUVEAU]** | ONTO-2 |

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

**Sacrifiable en premier** : ONTO-4/ONTO-5 (raffinements de l'ontologie), RAG-4
(hybridation lexicale — le vectoriel seul suffit à un score correct), FE-4
(explorateur de graphe), DOC-6 au-delà d'une version minimale.
