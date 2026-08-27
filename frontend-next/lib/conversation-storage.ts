/**
 * Persistance côté client de la conversation (historique + profil accumulé).
 *
 * Le backend est stateless (`POST /orientation/traiter` ne garde aucune
 * mémoire entre deux appels — limitation documentée, voir
 * `backend/tests/eval_analyse.md`) : c'est ce module qui simule une
 * conversation en gardant l'historique et en renvoyant le profil complet à
 * chaque tour. `sessionStorage` plutôt que `localStorage` : la conversation
 * ne doit pas survivre à la fermeture de l'onglet, cohérent avec le choix
 * déjà fait côté auth admin (session par onglet).
 */
import type { OrientationReponse, ProfilCandidat } from "@/lib/types";
import { profilVide } from "@/lib/types";

export interface TourConversation {
  id: string;
  message: string;
  reponse: OrientationReponse | null;
  erreur: string | null;
}

export interface EtatConversation {
  messages: TourConversation[];
  profil: ProfilCandidat;
}

const CLE_STOCKAGE = "orientia_conversation";

export function etatInitial(): EtatConversation {
  return { messages: [], profil: profilVide() };
}

export function chargerConversation(): EtatConversation | null {
  if (typeof window === "undefined") return null;
  try {
    const brut = window.sessionStorage.getItem(CLE_STOCKAGE);
    return brut ? (JSON.parse(brut) as EtatConversation) : null;
  } catch {
    return null;
  }
}

export function sauvegarderConversation(etat: EtatConversation): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(CLE_STOCKAGE, JSON.stringify(etat));
  } catch {
    // Quota dépassé ou stockage indisponible (navigation privée) : la
    // conversation reste utilisable pour la session en cours, seule la
    // persistance entre rechargements est perdue.
  }
}

export function effacerConversation(): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(CLE_STOCKAGE);
  } catch {
    // idem
  }
}
