/**
 * Génère le Google Form de l'enquête ORIENT'IA (DATA-4).
 *
 * POURQUOI UN SCRIPT plutôt qu'un formulaire construit à la main : le
 * questionnaire est un livrable versionné (`questionnaire.md`). Le
 * reconstruire à la souris à chaque ajustement le ferait diverger de sa
 * version de référence ; ici, le formulaire se régénère à l'identique.
 *
 * MODE D'EMPLOI (environ 1 minute)
 *  1. Ouvrir https://script.google.com/ → « Nouveau projet »
 *  2. Coller ce fichier en entier (remplacer le contenu par défaut)
 *  3. Cliquer « Exécuter ». `genererFormulaire` est la seule fonction de
 *     premier niveau du fichier, donc la seule que le sélecteur propose : il
 *     n'y a rien d'autre à choisir.
 *  4. Autoriser l'accès quand Google le demande (le script ne fait que créer
 *     un formulaire dans votre Drive)
 *  5. L'URL publique du formulaire s'affiche dans le journal d'exécution
 *     (Ctrl+Entrée pour l'ouvrir)
 *
 * APRÈS GÉNÉRATION, deux réglages à vérifier dans l'interface Google Forms :
 *  - « Collecter les adresses e-mail » doit rester DÉSACTIVÉ (anonymat, §5) ;
 *  - lier une feuille de réponses (Réponses → icône Sheets) pour l'export CSV
 *    qui alimentera DATA-7.
 */

// --- Vocabulaires partagés ---------------------------------------------------
// Alignés sur `backend/data/parcours.json` et sur le vocabulaire contrôlé de
// `src/ml/archetypes.py`. Toute modification ici doit être répercutée dans
// `questionnaire.md`, qui reste la version de référence.

var SERIES_BAC = [
  'A', 'A2', 'C', 'D', 'S',
  'Technique industrielle', 'Technique agricole', 'Technique génie civil',
  'Autre'
];

var PARCOURS = [
  'IGGLIA — Informatique de Gestion, Génie Logiciel et Intelligence Artificielle',
  'ESIIA — Électronique, Systèmes Informatiques et Intelligence Artificielle',
  'IMTICIA — Informatique, Multimédia, TIC et Intelligence Artificielle',
  'ISAIA — Informatique, Statistique Appliquée et Intelligence Artificielle',
  'EMII — Électromécanique et Techniques Industrielles Informatisées',
  'ICMP — Industries Chimiques, Minières et Pétrolières',
  'GCA — Génie Civil et Architecture',
  'CAA — Commerce et Administration des Affaires',
  'FIC — Finance et Comptabilité des Entreprises',
  'DTJA — Droit et Techniques Juridiques des Affaires',
  'EMP — Économie et Management de Projet',
  'IAA — Industries Agro-Alimentaires',
  'PIP — Pharmacologie et Industries Pharmaceutiques',
  'AEE — Agriculture / développement rural',
  'TEE — Tourisme de l\'Environnement',
  'TEH — Tourisme et Hôtellerie',
  'Une formation hors ISPM'
];

var MATIERES = [
  'Mathématiques', 'Physique', 'Chimie', 'Biologie', 'Sciences de la Terre',
  'Informatique', 'Électronique', 'Mécanique', 'Économie', 'Gestion',
  'Comptabilité', 'Droit', 'Langues', 'Histoire', 'Géographie',
  'Communication', 'Arts / dessin'
];

var COMPETENCES = [
  'Programmation', 'Algorithmique', 'Statistiques', 'Analyse de données',
  'Dessin technique', 'Électronique', 'Mécanique', 'Comptabilité',
  'Négociation', 'Rédaction', 'Accueil / relation client',
  'Techniques agricoles', 'Aucune en particulier'
];

var INTERETS = [
  'Technologie', 'Logiciels', 'Matériel informatique', 'Robotique', 'Données',
  'Construction', 'Urbanisme', 'Machines', 'Industrie',
  'Ressources naturelles', 'Agriculture', 'Nature / environnement', 'Santé',
  'Recherche', 'Commerce', 'Entrepreneuriat', 'Finance', 'Droit / justice',
  'Culture', 'Voyage', 'Hôtellerie'
];

var ENVIRONNEMENTS = [
  'Bureau', 'Laboratoire', 'Atelier ou usine', 'Chantier',
  'Terrain / extérieur', 'Contact direct avec des clients', 'Sans préférence'
];

var DESCRIPTION = [
  'Cette enquête alimente ORIENT\'IA, un projet étudiant de l\'Institut Supérieur',
  'Polytechnique de Madagascar : un assistant d\'aide à l\'orientation qui recommande',
  'des parcours à partir d\'un profil déclaré.',
  '',
  'Vos réponses servent à vérifier si ses recommandations correspondent à des',
  'parcours réels — aujourd\'hui, il n\'a été testé que sur des profils générés',
  'artificiellement.',
  '',
  'Durée : environ 5 minutes.',
  '',
  'ANONYMAT — aucune donnée permettant de vous identifier n\'est demandée (ni nom,',
  'ni adresse e-mail, ni téléphone), et aucune donnée personnelle sensible (genre,',
  'âge, origine, santé). Les réponses sont utilisées uniquement dans le cadre de ce',
  'projet pédagogique, agrégées, et publiées sous une forme qui ne permet pas de',
  'remonter à une personne.',
  '',
  'Vous pouvez arrêter à tout moment en fermant la page : rien n\'est enregistré',
  'tant que vous ne validez pas.'
].join('\n');

// --- Construction du formulaire ----------------------------------------------

function genererFormulaire() {
  // Helpers imbriqués volontairement : une fonction définie au niveau du
  // fichier apparaît dans le sélecteur d'Apps Script, et l'éditeur mémorise la
  // dernière fonction choisie dans le projet. Un helper lancé par mégarde
  // échouait alors sur un `form` absent — et réordonner le fichier ne
  // corrigeait pas une sélection déjà enregistrée. Imbriqués ici, ils
  // n'apparaissent plus du tout : `genererFormulaire` est le seul point
  // d'entrée possible.

  function ajouterSerieBac(form) {
    form.addMultipleChoiceItem()
      .setTitle('Série de votre baccalauréat')
      .setChoiceValues(SERIES_BAC)
      .setRequired(true);
  }

  function ajouterQuestionsProfil(form, moment) {
    form.addCheckboxItem()
      .setTitle('Matières que vous préfériez ' + moment)
      .setHelpText('Plusieurs réponses possibles.')
      .setChoiceValues(MATIERES);

    form.addTextItem()
      .setTitle('Autres matières qui vous plaisaient, non listées ci-dessus')
      .setHelpText('Facultatif — écrivez librement, séparé par des virgules.')
      .setRequired(false);

    form.addCheckboxItem()
      .setTitle('Compétences que vous aviez déjà ' + moment)
      .setChoiceValues(COMPETENCES);

    form.addCheckboxItem()
      .setTitle('Ce qui vous intéressait ' + moment)
      .setChoiceValues(INTERETS);

    form.addMultipleChoiceItem()
      .setTitle('Environnement de travail que vous imaginiez')
      .setChoiceValues(ENVIRONNEMENTS);
  }

  var form = FormApp.create('ORIENT\'IA — Enquête sur les parcours de formation (ISPM)');
  form.setDescription(DESCRIPTION);
  form.setProgressBar(true);
  // Anonymat (§5) : ne jamais associer une réponse à un compte Google.
  form.setCollectEmail(false);
  form.setLimitOneResponsePerUser(false);

  // Section 0 — consentement
  form.addCheckboxItem()
    .setTitle('Consentement')
    .setChoiceValues([
      'J\'ai lu ce qui précède et j\'accepte que mes réponses anonymes soient utilisées dans le cadre de ce projet.'
    ])
    .setRequired(true);

  // Section 1 — aiguillage. Les pages sont créées AVANT d'être référencées :
  // `createChoice(texte, page)` exige que la page existe déjà.
  var routage = form.addMultipleChoiceItem().setTitle('Vous êtes actuellement :').setRequired(true);

  var pageEtudiants = form.addPageBreakItem()
    .setTitle('Votre parcours (étudiant·e)')
    .setHelpText('Répondez en vous replaçant au moment où vous avez choisi votre formation.');

  ajouterSerieBac(form);

  form.addListItem()
    .setTitle('Quelle formation suivez-vous ?')
    .setChoiceValues(PARCOURS)
    .setRequired(true);

  ajouterQuestionsProfil(form, 'au lycée');

  form.addScaleItem()
    .setTitle('Aujourd\'hui, êtes-vous satisfait·e de ce choix ?')
    .setBounds(1, 5)
    .setLabels('Pas du tout', 'Tout à fait')
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('Avec le recul, referiez-vous le même choix ?')
    .setChoiceValues(['Oui', 'Non', 'Je ne sais pas'])
    .setRequired(true);

  form.addListItem()
    .setTitle('Si non, ou si vous hésitez : quelle formation aurait mieux convenu ?')
    .setHelpText('Facultatif.')
    .setChoiceValues(PARCOURS)
    .setRequired(false);

  var pagePros = form.addPageBreakItem()
    .setTitle('Votre parcours (professionnel·le)')
    .setHelpText(
      'Répondez en vous replaçant avant vos études, puis sur votre situation '
      + 'actuelle. C\'est cette population qui montre le point d\'arrivée réel.'
    );

  // Fin du parcours étudiant : soumettre au lieu d'enchaîner sur la section
  // professionnelle.
  //
  // `setGoToPage` se pose sur le saut de page qui SUIT la section à terminer,
  // pas sur celui qui l'ouvre : la documentation Apps Script précise qu'il
  // règle « la page vers laquelle naviguer après avoir terminé la page qui
  // précède ce saut de page ». Posé sur `pageEtudiants`, il s'appliquait donc
  // à la page de consentement, et les étudiants enchaînaient sur la section
  // professionnelle — 12 des 15 premières réponses ont rempli les deux
  // sections avant que ce défaut ne soit repéré dans l'export.
  pagePros.setGoToPage(FormApp.PageNavigationType.SUBMIT);

  ajouterSerieBac(form);

  form.addListItem()
    .setTitle('Quelle formation avez-vous suivie ?')
    .setChoiceValues(PARCOURS)
    .setRequired(true);

  form.addTextItem()
    .setTitle('Quel métier exercez-vous aujourd\'hui ?')
    .setRequired(true);

  ajouterQuestionsProfil(form, 'avant vos études');

  form.addScaleItem()
    .setTitle('Votre formation correspond-elle au métier que vous exercez ?')
    .setBounds(1, 5)
    .setLabels('Pas du tout', 'Tout à fait')
    .setRequired(true);

  form.addListItem()
    .setTitle('Avec le recul, quelle formation aurait le mieux correspondu à votre profil de l\'époque ?')
    .setHelpText(
      'La question la plus utile de cette enquête : le parcours choisi n\'est pas '
      + 'toujours celui qui convenait.'
    )
    .setChoiceValues(PARCOURS)
    .setRequired(true);

  form.addParagraphTextItem()
    .setTitle('Un commentaire à ajouter ?')
    .setRequired(false);

  routage.setChoices([
    routage.createChoice('Étudiant·e, en cours d\'études', pageEtudiants),
    routage.createChoice('Professionnel·le en activité, études terminées', pagePros)
  ]);

  Logger.log('Formulaire créé.');
  Logger.log('À diffuser  : ' + form.getPublishedUrl());
  Logger.log('À modifier  : ' + form.getEditUrl());
  Logger.log('');
  Logger.log('Pensez à lier une feuille de réponses (Réponses → icône Sheets)');
  Logger.log('pour disposer de l\'export CSV qui alimentera DATA-7.');
}
