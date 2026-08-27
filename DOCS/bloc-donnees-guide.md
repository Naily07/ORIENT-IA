# Bloc Données — guide de délégation (DATA-1 à DATA-8)

Ce document est destiné à un développeur qui reprend le bloc **Données** du
projet ORIENT'IA sans avoir suivi le reste du travail. Il rassemble ce qui
est déjà fait, ce qui reste à faire, la structure de fichiers à respecter,
les dépendances techniques, et les exigences non négociables du sujet.

Source de vérité pour le suivi d'avancement : [`BACKLOG.md`](../BACKLOG.md)
à la racine du dépôt (tickets `DATA-*`). Ce guide en est la version détaillée
pour ce bloc précis — en cas de divergence, `BACKLOG.md` fait foi (c'est lui
que l'équipe tient à jour au fil des PR).

## 1. Contexte en une minute

ORIENT'IA est un assistant d'orientation pédagogique pour l'ISPM (hackathon
ISPM, Master 2). Le sujet exige que le système combine un corpus documentaire
(déjà largement fait), un modèle de Machine Learning (fait, sur données
synthétiques), et un agent conversationnel (pas encore fait). Le bloc
**Données** couvre l'acquisition et la traçabilité des deux matières
premières du système : le corpus pédagogique et le jeu de données
d'entraînement du modèle.

## 2. Ce qui est déjà fait — ne pas refaire

| Ticket | État | Où |
|---|---|---|
| DATA-1 | Partiel | `backend/data/{mentions,parcours,prerequis,corpus}.json` — 6 mentions et 16 parcours réels de l'ISPM (IGGLIA, ESIIA, ISAIA, CAA, PIP, TEH...), niveaux de diplôme, prérequis d'admission, collectés depuis le site officiel `ispm-edu.com` |
| DATA-2 | ✅ Fait | `backend/src/sources.py` + `backend/data/registre_sources.json` — registre de traçabilité des sources |
| DATA-3 | Partiel | `source_id` posé sur tous les modèles structurés, `verifier_provenance()` opérationnel |
| DATA-6 | ✅ Fait | `backend/src/ml/{archetypes,donnees_synthetiques}.py` — 800 profils synthétiques dans `backend/data/ml/profils_synthetiques.json` |

Le corpus réel a été collecté par recherche web sur le site officiel de
l'ISPM (`ispm-edu.com/{filieres,presentation,inscription}.php`) et recoupé
avec une source tierce (`annuaire.mg`). Deux incertitudes ont été
volontairement documentées plutôt que tranchées à l'aveugle : le
développement exact du sigle **CAA** diverge selon les sources, et le sigle
**AEE** n'a pas d'expansion confirmée — voir les `limites` de
`SRC-ANNUAIRE-MG` dans `registre_sources.json`.

## 3. Ce qu'il reste à faire — le travail délégué

### 3.1 Compléter DATA-1 / DATA-3 : matières, compétences, débouchés, passerelles

Aucune source fiable en ligne n'a été trouvée pour :
- les matières précises enseignées par parcours,
- les compétences détaillées visées,
- les débouchés métiers nommés,
- les passerelles entre formations.

**Ne pas les inventer.** Le sujet est explicite : une information non
vérifiée ne doit jamais être présentée comme officielle. Pistes à explorer :
brochures papier de l'ISPM (à scanner/photographier si besoin), maquettes
pédagogiques, ou contact direct avec l'administration de l'ISPM. Une fois
les données obtenues, peupler `backend/data/matieres.json`,
`backend/data/competences.json`, `backend/data/metiers.json` (le loader
`charger_corpus_formations()` de `backend/src/models.py` les attend déjà,
mais tolère leur absence) et ajouter les entrées de source correspondantes
dans `registre_sources.json`.

### 3.2 DATA-4 : questionnaire d'enquête — **rédigé, reste à diffuser**

Le questionnaire est écrit et couvre les **deux populations distinctes**
exigées par le sujet :

- [`backend/data/enquete/questionnaire.md`](../backend/data/enquete/questionnaire.md)
  — version de référence (21 questions, consentement explicite, aucune donnée
  sensible, chaque question annotée du champ `ProfilCandidat` qu'elle
  alimente). C'est le livrable exigé au §5 (« le questionnaire lui-même, dans
  la version effectivement diffusée »).
- [`backend/data/enquete/generer_google_form.gs`](../backend/data/enquete/generer_google_form.gs)
  — script Apps Script qui génère le formulaire Google à l'identique depuis
  cette référence, avec l'aiguillage étudiants / professionnels. Mode d'emploi
  en tête du fichier (~1 minute).

**Ce qu'il reste à faire, le jour J :**

1. Exécuter le script (script.google.com → coller → Exécuter).
2. Vérifier que « Collecter les adresses e-mail » est bien **désactivé**
   (anonymat) et lier une feuille de réponses pour l'export CSV.
3. Diffuser aux deux populations, en notant le mode de diffusion propre à
   chacune (à reporter dans DATA-5).
4. **Geler la collecte en fin de première journée.**

**Contrainte de calendrier** : le sujet exige que l'enquête soit lancée dès
la première heure du hackathon et gelée à la fin de la première journée —
c'est le seul ticket de tout le backlog dont l'échéance ne se rattrape pas.
Le questionnaire étant désormais prêt, cette étape se réduit à quelques
minutes le matin même.

### 3.3 DATA-5 : registre de collecte de l'enquête

Documenter la collecte, avec au minimum les champs listés au §4. Suivre le
même pattern que le registre des sources (DATA-2) : un module
`backend/src/enquete.py` avec une classe Pydantic et un loader tolérant à
l'absence de fichier, sur le modèle de `backend/src/sources.py` :

```python
# backend/src/sources.py — à prendre comme modèle
class EntreeRegistreSource(BaseModel):
    id: str
    titre: str
    url: str
    date_consultation: date
    statut: StatutSource
    donnees_extraites: list[str] = Field(default_factory=list)
    limites: list[str] = Field(default_factory=list)

def charger_registre_sources(nom_fichier: str = "registre_sources.json") -> list[EntreeRegistreSource]:
    ...
```

### 3.4 DATA-8 : anonymisation des réponses

Avant toute livraison du jeu de données d'enquête, retirer les identifiants
directs (nom, email, numéro de téléphone...). Point de départ : les
fonctions déjà écrites dans `backend/src/guardrails.py` —
`masquer_donnees_sensibles()` (texte libre) et `masquer_objet()` (structures
JSON récursives) savent déjà masquer emails et motifs de type
secret/identifiant. Elles ont été conçues pour les logs d'observabilité, pas
spécifiquement pour l'anonymisation RGPD-like d'une enquête : à revoir et
étendre au besoin (ex. noms propres), pas à réutiliser en aveugle.

### 3.5 DATA-7 (volet enquête) : fusion avec le jeu synthétique

Une fois les réponses d'enquête anonymisées disponibles, les convertir au
format compatible avec `src.schemas.ProfilCandidat` (le même schéma que les
profils synthétiques, voir `backend/src/ml/donnees_synthetiques.py` pour le
format exact d'un enregistrement `{"profil": ..., "parcours_id": ...}`) et
les assembler en jeu de **validation/test**, séparé du jeu synthétique qui
reste réservé à l'entraînement (montage recommandé explicitement par le
sujet — voir §4). Le ticket **ML-7** (mesurer la généralisation
synthèse → réel) ne pourra être fait qu'une fois ce travail terminé.

## 4. Exigences du sujet à respecter à la lettre

Reformulées depuis le sujet original (`Sujet Clinique - OrientIA.pdf`, §5) —
en cas de doute, se référer au PDF directement.

**Les deux populations à enquêter** :
- **Étudiants** : profil déclaré au moment de l'inscription, parcours
  effectivement choisi, satisfaction constatée après coup.
- **Professionnels déjà établis** : profil tel qu'il était avant les études,
  parcours suivi, métier réellement exercé aujourd'hui, et jugement
  rétrospectif sur l'adéquation entre les deux. Cette population est la plus
  précieuse : elle seule montre le point d'arrivée réel, là où un étudiant ne
  renseigne qu'un choix dont l'issue reste inconnue.

**Montage recommandé** : données synthétiques pour l'entraînement (déjà
fait, DATA-6), réponses d'enquête pour la validation et le test uniquement.
Ne pas mélanger les deux dans le même sous-ensemble.

**Trois limites à nommer explicitement dans le registre de collecte, jamais
à masquer** :
1. **Le volume** — quelques centaines de réponses au mieux ; les intervalles
   de confiance seront larges, à annoncer comme tels.
2. **L'auto-sélection** — les répondants sur-représenteront probablement
   certains parcours et certains profils.
3. **La nature de l'étiquette** — chez un étudiant, le parcours *choisi*
   n'est pas forcément le parcours qui *convenait* ; un modèle entraîné sur
   des choix passés en reproduit les biais (y compris de genre ou de filière
   d'origine). Le jugement rétrospectif des professionnels corrige en partie
   ce défaut, au prix d'un biais de reconstruction (la mémoire d'un choix
   ancien se reconstruit, elle ne se rejoue pas).

**Champs minimum du registre de collecte (DATA-5)** :
- le questionnaire lui-même, dans la version effectivement diffusée ;
- les populations visées et le mode de diffusion propre à chacune ;
- la période de collecte, le nombre de réponses reçues, retenues et
  écartées ;
- la répartition des réponses entre étudiants et professionnels ;
- le texte de consentement présenté aux répondants ;
- la procédure d'anonymisation appliquée ;
- les traitements postérieurs (nettoyage, exclusions, recodages) et leur
  justification ;
- les biais d'échantillonnage constatés.

**Règle non négociable** (citée telle quelle dans le sujet) : *« Aucune
donnée personnelle sensible ne devra être collectée. Les réponses livrées
devront être anonymisées et le consentement des répondants explicite. Un jeu
de données dont la provenance ne peut être retracée ne sera pas
recevable. »*

## 5. Dépendances techniques (setup)

Identique au reste du projet — voir la racine du dépôt pour le détail complet.

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"    # Linux/macOS : .venv/bin/pip
cp .env.example .env                      # GEMINI_API_KEY, optionnel pour ce bloc
```

- **Python 3.11+**
- Les dépendances du bloc Données sont déjà toutes dans `pyproject.toml` /
  `requirements.txt` (`pydantic`, `pydantic-settings`) — rien de nouveau à
  ajouter pour DATA-4/5/8.
- `GEMINI_API_KEY` n'est **pas nécessaire** pour ce bloc précis (aucune tâche
  ci-dessus n'appelle le LLM), mais est nécessaire pour lancer la suite de
  tests complète du projet sans exclusion.
- **Pas de scraping obligatoire** — la collecte manuelle est recommandée pour
  ce qui reste du corpus (voir la discussion dans l'historique du projet :
  un scraper ne dispense pas de relire chaque page pour remplir le registre
  de traçabilité, donc le gain de temps est faible).

## 6. Structure de fichiers à respecter

```
backend/
├── data/
│   ├── registre_sources.json      # existant — ajouter toute nouvelle source ISPM ici
│   ├── mentions.json              # existant
│   ├── parcours.json              # existant
│   ├── prerequis.json             # existant
│   ├── matieres.json              # à créer (DATA-1)
│   ├── competences.json           # à créer (DATA-1)
│   ├── metiers.json               # à créer (DATA-1)
│   ├── corpus.json                # existant (RAG) — ajouter des articles si utile
│   ├── enquete/                   # à créer
│   │   ├── questionnaire.md       # le questionnaire tel que diffusé (DATA-4)
│   │   ├── reponses_orientia.json     # livrable — jamais de données brutes identifiantes ici
│   │   └── registre_collecte.json     # DATA-5
│   └── ml/
│       └── profils_synthetiques.json  # existant, ne pas modifier ici (DATA-7 fusionne ailleurs)
├── src/
│   ├── models.py       # Mention/Parcours/Matiere/Competence/Prerequis/Metier déjà définis
│   ├── sources.py       # pattern à reproduire pour enquete.py
│   ├── enquete.py        # à créer (DATA-5) : RegistreCollecte + loader
│   ├── guardrails.py      # masquer_donnees_sensibles/masquer_objet, point de départ DATA-8
│   └── schemas.py          # ProfilCandidat — format cible pour les réponses d'enquête (DATA-7)
└── tests/
    ├── test_sources.py    # pattern à suivre pour test_enquete.py
    └── test_enquete.py     # à créer
```

## 7. Definition of done

- [ ] Matières/compétences/débouchés/passerelles collectées avec une source
      identifiée pour chacune (ou explicitement laissées de côté avec la
      raison documentée)
- [ ] Questionnaire rédigé, couvrant les deux populations (étudiants et
      professionnels), avec un texte de consentement explicite
- [ ] Questionnaire réellement diffusé, collecte gelée en fin de première
      journée du hackathon
- [ ] Réponses anonymisées avant toute livraison
- [ ] `backend/src/enquete.py` + `backend/data/enquete/registre_collecte.json`
      couvrant tous les champs du §4
- [ ] Fusion avec le jeu synthétique en jeu de validation/test distinct
      (DATA-7)
- [ ] Tests écrits (`test_enquete.py`), suivant le pattern de
      `test_sources.py` (y compris un test de non-régression qui vérifie
      qu'aucune référence de source n'est orpheline, comme
      `test_le_corpus_reel_ne_contient_aucune_source_orpheline`)
- [ ] `BACKLOG.md` mis à jour (cocher DATA-1/DATA-3 complets si applicable,
      DATA-4, DATA-5, DATA-8 ; compléter DATA-7)
- [ ] `pytest` et `ruff check .` passent sans erreur depuis la racine du dépôt

## 8. Workflow Git (identique au reste du projet)

```bash
git checkout develop && git pull
git checkout -b feature/<nom-descriptif>
# ... travail, commits réguliers ...
pytest -q && ruff check .   # doivent passer avant toute PR
git push -u origin feature/<nom-descriptif>
gh pr create --base develop --head feature/<nom-descriptif> --title "..." --body "..."
```

Ne pas ajouter de ligne `Co-Authored-By` dans les commits ou les PR
(convention de ce projet). Ouvrir une PR vers `develop`, jamais directement
vers `main`.
