# Questionnaire d'enquête ORIENT'IA (DATA-4)

Version de référence du questionnaire, telle que destinée à être diffusée.
C'est le livrable exigé par le sujet (§5, « le questionnaire lui-même, dans la
version effectivement diffusée ») ; le formulaire Google est généré à partir
d'ici par [`generer_google_form.gs`](generer_google_form.gs), pour que les deux
ne puissent pas diverger.

## Ce que cette enquête doit produire, et pourquoi

Le sujet demande deux populations, et explique pourquoi la seconde est la plus
précieuse : un étudiant ne renseigne qu'un choix dont l'issue reste inconnue,
là où un professionnel montre le **point d'arrivée réel**.

Contrainte propre à notre système : les réponses doivent pouvoir alimenter
`src.schemas.ProfilCandidat` pour servir de jeu de **validation/test** face au
modèle entraîné sur données synthétiques (DATA-7, ML-7). Chaque question
ci-dessous porte donc le champ qu'elle alimente.

**Choix de conception : cases à cocher issues du vocabulaire contrôlé *plus* un
champ libre.** Les cases donnent des données propres et un questionnaire
répondable en 5 minutes ; le champ libre teste en conditions réelles la
résolution de vocabulaire ouvert (ML-9), qui n'a jusqu'ici été éprouvée que sur
des termes que nous avons choisis nous-mêmes. Ne proposer que des cases
biaiserait le test de généralisation vers notre propre vocabulaire.

## Données volontairement NON collectées

Aucune question sur le genre, l'âge, l'origine, la religion, la situation de
famille ou l'état de santé. Deux raisons, et la seconde suffirait :

1. Le sujet l'interdit explicitement (§5 : « aucune donnée personnelle sensible
   ne devra être collectée »).
2. Le modèle ne peut de toute façon pas s'en servir — ces dimensions n'existent
   pas dans son espace de features, par construction (SEC-3).

Aucun nom, e-mail ni numéro de téléphone non plus : les réponses sont anonymes
à la source, ce qui rend l'anonymisation (DATA-8) triviale plutôt que
rattrapée après coup.

---

## Section 0 — Présentation et consentement

**Titre** : ORIENT'IA — Enquête sur les parcours de formation (ISPM)

**Description affichée :**

> Cette enquête alimente **ORIENT'IA**, un projet étudiant de l'Institut
> Supérieur Polytechnique de Madagascar : un assistant d'aide à l'orientation
> qui recommande des parcours à partir d'un profil déclaré.
>
> Vos réponses servent à **vérifier** si les recommandations du système
> correspondent à des parcours réels — aujourd'hui, il n'a été testé que sur des
> profils générés artificiellement.
>
> **Durée : environ 5 minutes.**
>
> **Anonymat** : aucune donnée permettant de vous identifier n'est demandée (ni
> nom, ni adresse e-mail, ni téléphone), et aucune donnée personnelle sensible
> (genre, âge, origine, santé). Les réponses sont utilisées uniquement dans le
> cadre de ce projet pédagogique, agrégées, et publiées sous une forme qui ne
> permet pas de remonter à une personne.
>
> Vous pouvez arrêter à tout moment en fermant la page : rien n'est enregistré
> tant que vous ne validez pas.

**Q0. Consentement** *(obligatoire, case à cocher)*
- [ ] J'ai lu ce qui précède et j'accepte que mes réponses anonymes soient
      utilisées dans le cadre de ce projet.

---

## Section 1 — Aiguillage

**Q1. Vous êtes actuellement :** *(obligatoire, choix unique — détermine la suite)*
- Étudiant(e), en cours d'études → *section 2*
- Professionnel(le) en activité, études terminées → *section 3*

---

## Section 2 — Étudiants

> Répondez en vous replaçant **au moment où vous avez choisi votre formation**.

**Q2. Série de votre baccalauréat** *(choix unique)* → `serie_bac`
A · A2 · C · D · S · Technique industrielle · Technique agricole ·
Technique génie civil · Autre

**Q3. Quelle formation suivez-vous ?** *(liste déroulante)* → **étiquette**

| Sigle | Intitulé | Mention |
|---|---|---|
| IGGLIA | Informatique de Gestion, Génie Logiciel et Intelligence Artificielle | Informatique et Télécommunications |
| ESIIA | Électronique, Systèmes Informatiques et Intelligence Artificielle | Informatique et Télécommunications |
| IMTICIA | Informatique, Multimédia, TIC et Intelligence Artificielle | Informatique et Télécommunications |
| ISAIA | Informatique, Statistique Appliquée et Intelligence Artificielle | Informatique et Télécommunications |
| EMII | Électromécanique et Techniques Industrielles Informatisées | Génie Industriel |
| ICMP | Industries Chimiques, Minières et Pétrolières | Génie Industriel |
| GCA | Génie Civil et Architecture | Génie Civil et Architecture |
| CAA | Commerce et Administration des Affaires | Droit et Techniques des Affaires |
| FIC | Finance et Comptabilité des Entreprises | Droit et Techniques des Affaires |
| DTJA | Droit et Techniques Juridiques des Affaires | Droit et Techniques des Affaires |
| EMP | Économie et Management de Projet | Droit et Techniques des Affaires |
| IAA | Industries Agro-Alimentaires | Biotechnologie et Agronomie |
| PIP | Pharmacologie et Industries Pharmaceutiques | Biotechnologie et Agronomie |
| AEE | AEE (Agriculture / développement rural) | Biotechnologie et Agronomie |
| TEE | Tourisme de l'Environnement | Tourisme |
| TEH | Tourisme et Hôtellerie | Tourisme |

*(+ « Une formation hors ISPM » — réponse conservée mais exclue du jeu
d'évaluation, et comptabilisée dans le registre de collecte comme écartée.)*

**Q4. Matières que vous préfériez au lycée** *(cases à cocher, plusieurs
réponses)* → `matieres_preferees`
Mathématiques · Physique · Chimie · Biologie · Sciences de la Terre ·
Informatique · Électronique · Mécanique · Économie · Gestion · Comptabilité ·
Droit · Langues · Histoire · Géographie · Communication · Arts / dessin

**Q5. Autres matières qui vous plaisaient, non listées ci-dessus** *(texte
libre, facultatif)* → `matieres_preferees` (test de ML-9)

**Q6. Compétences que vous aviez déjà** *(cases à cocher)* → `competences_declarees`
Programmation · Algorithmique · Statistiques · Analyse de données ·
Dessin technique · Électronique · Mécanique · Comptabilité · Négociation ·
Rédaction · Accueil / relation client · Techniques agricoles · Aucune en particulier

**Q7. Ce qui vous intéressait** *(cases à cocher)* → `centres_interet`
Technologie · Logiciels · Matériel informatique · Robotique · Données ·
Construction · Urbanisme · Machines · Industrie · Ressources naturelles ·
Agriculture · Nature / environnement · Santé · Recherche · Commerce ·
Entrepreneuriat · Finance · Droit / justice · Culture · Voyage · Hôtellerie

**Q8. Environnement de travail que vous imaginiez** *(choix unique)* →
`environnement_travail_recherche`
Bureau · Laboratoire · Atelier ou usine · Chantier · Terrain / extérieur ·
Contact direct avec des clients · Sans préférence

**Q9. Aujourd'hui, êtes-vous satisfait(e) de ce choix ?** *(échelle 1 à 5)*
1 = pas du tout · 5 = tout à fait

**Q10. Avec le recul, referiez-vous le même choix ?** *(choix unique)*
Oui · Non · Je ne sais pas

**Q11. Si non ou si vous hésitez : quelle formation aurait mieux convenu ?**
*(liste déroulante, facultatif — même liste qu'en Q3)*

---

## Section 3 — Professionnels

> Répondez en vous replaçant **avant vos études**, puis sur votre situation
> actuelle. C'est cette population qui montre le point d'arrivée réel — un
> étudiant ne peut pas encore le connaître.

**Q12. Série de votre baccalauréat** *(choix unique)* → `serie_bac`
*(mêmes options qu'en Q2)*

**Q13. Quelle formation avez-vous suivie ?** *(liste déroulante)* → **étiquette**
*(même liste qu'en Q3)*

**Q14. Quel métier exercez-vous aujourd'hui ?** *(texte libre)*
→ alimentera `Metier` du corpus structuré (DATA-1, débouchés non encore
collectés) autant que l'évaluation

**Q15 à Q18** — profil **avant les études** : matières préférées, autres
matières (texte libre), compétences, centres d'intérêt, environnement de
travail souhaité *(identiques à Q4–Q8)*

**Q19. Votre formation correspond-elle au métier que vous exercez ?**
*(échelle 1 à 5)* — 1 = pas du tout · 5 = tout à fait

**Q20. Avec le recul, quelle formation aurait le mieux correspondu à votre
profil de l'époque ?** *(liste déroulante — même liste qu'en Q3)*

> C'est la question la plus précieuse de l'enquête : elle fournit une étiquette
> corrigée par l'expérience, là où le parcours *choisi* n'est pas
> nécessairement le parcours qui *convenait*.

**Q21. Un commentaire à ajouter ?** *(texte libre, facultatif)*

---

## Limites à reporter dans le registre de collecte (DATA-5)

Les trois limites que le sujet demande de nommer, à instancier avec les
chiffres réels une fois la collecte close :

1. **Le volume** — quelques centaines de réponses au mieux ; les intervalles de
   confiance seront larges et doivent être annoncés comme tels.
2. **L'auto-sélection** — la diffusion par les réseaux de l'équipe
   sur-représentera certains parcours (vraisemblablement l'informatique) et
   certains profils.
3. **La nature de l'étiquette** — chez un étudiant, le parcours *choisi* n'est
   pas le parcours qui *convenait* ; un modèle entraîné sur des choix passés en
   reproduit les biais. Q10/Q11 et Q20 servent précisément à mesurer cet écart.

Une quatrième limite, propre à la population professionnelle : **le biais de
reconstruction**. La formation date de plusieurs années, l'offre a changé, et
le souvenir des motivations d'alors se reconstruit — Q15 à Q18 demandent de se
replacer dans le passé, ce qui n'est pas une mesure directe.
