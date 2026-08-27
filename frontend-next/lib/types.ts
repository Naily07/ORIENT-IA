/**
 * Miroir TypeScript des schémas Pydantic du backend.
 *
 * Garder ce fichier synchrone avec `backend/src/schemas.py` et
 * `backend/src/admin_api.py` — aucune logique ici, uniquement des formes de
 * données. Un champ ajouté/renommé côté backend doit être répercuté ici.
 */

// --- backend/src/schemas.py ------------------------------------------------

export const ACTIONS_CONNUES = [
  "information",
  "recommandation",
  "demande_information",
  "escalade_conseiller",
  "renvoi_administration",
] as const;

export type Action = (typeof ACTIONS_CONNUES)[number];

export interface ProfilCandidat {
  matieres_preferees: string[];
  resultats_scolaires: Record<string, number>;
  competences_declarees: string[];
  centres_interet: string[];
  activites_projets: string[];
  preferences_professionnelles: string[];
  environnement_travail_recherche: string | null;
  serie_bac: string | null;
  informations_manquantes: string[];
}

export function profilVide(): ProfilCandidat {
  return {
    matieres_preferees: [],
    resultats_scolaires: {},
    competences_declarees: [],
    centres_interet: [],
    activites_projets: [],
    preferences_professionnelles: [],
    environnement_travail_recherche: null,
    serie_bac: null,
    informations_manquantes: [],
  };
}

export interface RecommandationParcours {
  parcours: string;
  score_adequation: number;
  justification: string;
}

export interface RecommandationDecision {
  resume: string;
  /**
   * Réponse rédigée pour l'utilisateur, en langage courant. C'est ce que le
   * chat affiche en premier ; `resume`/`explication`/`sources` restent la
   * version tracée pour le jury.
   */
  reponse: string;
  parcours_recommandes: RecommandationParcours[];
  confiance: number;
  informations_manquantes: string[];
  explication: string;
  sources: string[];
  outils_utilises: string[];
  action: Action;
  incertitude_declaree: boolean;
}

/**
 * Un échange déjà joué, rejoué à chaque requête (`backend/src/schemas.py`).
 *
 * Le pipeline backend reste sans état : c'est le client qui porte la
 * conversation. Sans cet aller-retour, une question de suivi (« et les
 * matières de cette filière ? ») arrivait seule et restait insoluble.
 */
export interface TourHistorique {
  question: string;
  reponse: string;
}

/** Doit rester aligné sur `MAX_TOURS_HISTORIQUE` (backend/src/schemas.py) : le
 *  backend rejette au-delà. */
export const MAX_TOURS_HISTORIQUE = 6;

export interface OrientationInput {
  message: string;
  profil: ProfilCandidat;
  historique?: TourHistorique[];
}

export interface OrientationReponse {
  trace_id: string;
  decision: RecommandationDecision;
}

// --- backend/src/api.py : GET /health ---------------------------------------

export interface SanteReponse {
  status: string;
  modele: string;
  cle_llm_configuree: boolean;
  corpus: {
    mentions: number;
    parcours: number;
  };
  // SEC-5 : source unique du texte réglementaire, à afficher tel quel.
  mention_obligatoire: string;
  // Lus par les cartes de parcours pour masquer un score creux / signaler une
  // admissibilité à vérifier — voir lib/markers.ts.
  marqueur_regle_admission: string;
  avertissement_non_exploitable: string;
}

// --- backend/src/observability.py : GET /observabilite/traces --------------

export interface Trace {
  horodatage: string;
  trace_id: string;
  description: string | null;
  nb_documents_contexte: number;
  decision: Partial<RecommandationDecision> | null;
  latence_ms: number;
  // `log_trace` accepte des champs libres en plus de ceux ci-dessus
  // (`**champs_supplementaires`) — non typés ici, non utilisés par le
  // frontend au-delà de l'expansion JSON brute.
  [champLibre: string]: unknown;
}

// --- backend/src/admin_api.py ------------------------------------------------

export interface ConfigurationCalibree {
  rag_seuil_pertinence: number;
  rag_k: number;
  agent_max_iterations: number;
  orchestrateur_seuil_confiance: number;
}

export interface EtatAvancementDonnees {
  matieres: number;
  competences: number;
  metiers: number;
  prerequis: number;
}

export interface TableauDeBordReponse {
  configuration: ConfigurationCalibree;
  etat_avancement_donnees: EtatAvancementDonnees;
}

export type Intervalle = "heure" | "jour";

export interface SeauTendance {
  periode: string;
  volume: number;
  latence_moyenne_ms: number;
  confiance_moyenne: number | null;
  repartition_actions: Record<string, number>;
}

export interface TendancesReponse {
  intervalle: Intervalle;
  seaux: SeauTendance[];
}

export type StatutSource = "officiel" | "institutionnel" | "externe";

export interface EntreeRegistreSource {
  id: string;
  titre: string;
  url: string;
  date_consultation: string;
  statut: StatutSource;
  donnees_extraites: string[];
  limites: string[];
}

export interface QualiteDonneesReponse {
  incoherences: Record<string, unknown>[];
  donnees_manquantes: Record<string, unknown>[];
  contradictions: Record<string, unknown>[];
  registre_sources: EntreeRegistreSource[];
  references_orphelines: string[];
}

export interface Mention {
  id: string;
  nom: string;
  niveau: string;
  diplome: string | null;
  source_id: string | null;
}

export interface Matiere {
  id: string;
  nom: string;
  source_id: string | null;
}

export interface Competence {
  id: string;
  nom: string;
  metiers_requis: string[];
  source_id: string | null;
}

export interface Prerequis {
  id: string;
  description: string;
  source_id: string | null;
}

export interface Metier {
  id: string;
  nom: string;
  secteur: string | null;
  source_id: string | null;
}

export interface Parcours {
  id: string;
  nom: string;
  mention_id: string;
  matieres: string[];
  competences: string[];
  prerequis: string[];
  debouches: string[];
  passerelles: string[];
  source_id: string | null;
}

export interface CorpusFormations {
  mentions: Mention[];
  parcours: Parcours[];
  matieres: Matiere[];
  competences: Competence[];
  prerequis: Prerequis[];
  metiers: Metier[];
}

export type TypeEntiteGraphe =
  | "Parcours"
  | "Mention"
  | "Prerequis"
  | "Competence"
  | "Metier"
  | "Matiere";

export interface NoeudGraphe {
  id: string;
  type: TypeEntiteGraphe;
  nom: string;
  couleur: string;
}

export interface RelationGraphe {
  source: string;
  cible: string;
  relation: string;
}

export interface GrapheReponse {
  noeuds: NoeudGraphe[];
  relations: RelationGraphe[];
}

export interface ArtefactMesure {
  disponible: boolean;
  donnees: Record<string, unknown> | unknown[] | null;
  commande: string | null;
}

export interface MesuresReponse {
  ml: ArtefactMesure;
  rag: ArtefactMesure;
  systeme: ArtefactMesure;
}
