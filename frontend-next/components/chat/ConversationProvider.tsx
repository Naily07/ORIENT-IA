"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  chargerConversation,
  effacerConversation,
  etatInitial,
  sauvegarderConversation,
  type EtatConversation,
  type TourConversation,
} from "@/lib/conversation-storage";
import {
  MAX_TOURS_HISTORIQUE,
  type OrientationReponse,
  type ProfilCandidat,
  type TourHistorique,
} from "@/lib/types";

/**
 * Condense les tours déjà joués pour l'agent.
 *
 * On rejoue `decision.reponse` — le texte réellement lu par l'utilisateur —
 * complété des parcours nommés. Une question de suivi (« et les matières de
 * cette filière ? ») n'est résoluble que si le tour précédent dit encore de
 * quelle filière il s'agit.
 *
 * Les tours en erreur sont écartés : rejouer une question restée sans réponse
 * ferait croire à l'agent qu'il y a déjà répondu.
 */
function construireHistorique(messages: TourConversation[]): TourHistorique[] {
  return messages
    .filter((tour) => tour.reponse !== null)
    .slice(-MAX_TOURS_HISTORIQUE)
    .map((tour) => {
      const decision = tour.reponse!.decision;
      const parcours = decision.parcours_recommandes.map((p) => p.parcours).join(", ");
      const morceaux = [decision.reponse || decision.explication || decision.resume];
      if (parcours) morceaux.push(`Parcours cités : ${parcours}.`);
      return { question: tour.message, reponse: morceaux.join(" ") };
    });
}

interface ConversationContextValue {
  messages: TourConversation[];
  profil: ProfilCandidat;
  brouillon: string;
  enCours: boolean;
  definirBrouillon: (texte: string) => void;
  envoyerMessage: (message: string) => Promise<void>;
  mettreAJourProfil: (profil: ProfilCandidat) => void;
  reinitialiser: () => void;
}

const ConversationContext = createContext<ConversationContextValue | null>(null);

/**
 * Le backend est stateless : cette conversation n'existe que côté client
 * (historique + profil accumulé, persistés en sessionStorage). Chaque tour
 * renvoie le profil complet à `POST /orientation/traiter` via
 * `app/api/orientation/route.ts` — voir `lib/conversation-storage.ts`.
 */
export function ConversationProvider({ children }: { children: ReactNode }) {
  const [etat, setEtat] = useState<EtatConversation>(etatInitial);
  const [brouillon, setBrouillon] = useState("");
  const [enCours, setEnCours] = useState(false);
  const hydrate = useRef(false);

  useEffect(() => {
    // Hydratation depuis sessionStorage (browser-only) après le montage :
    // nécessaire pour que le rendu serveur (sans accès à sessionStorage) et
    // le premier rendu client coïncident — pas remplaçable par un calcul
    // pendant le rendu sans provoquer un mismatch d'hydratation.
    const sauvegarde = chargerConversation();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (sauvegarde) setEtat(sauvegarde);
    hydrate.current = true;
  }, []);

  useEffect(() => {
    // Ignore le premier rendu (avant hydratation depuis sessionStorage) pour
    // ne pas écraser une conversation déjà persistée avec l'état vide initial.
    if (!hydrate.current) return;
    sauvegarderConversation(etat);
  }, [etat]);

  const envoyerMessage = useCallback(
    async (message: string) => {
      const id = crypto.randomUUID();
      const profilCourant = etat.profil;
      const historique = construireHistorique(etat.messages);

      setEtat((precedent) => ({
        ...precedent,
        messages: [...precedent.messages, { id, message, reponse: null, erreur: null }],
      }));
      setEnCours(true);

      try {
        const reponseHttp = await fetch("/api/orientation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, profil: profilCourant, historique }),
        });
        const corps = await reponseHttp.json().catch(() => null);
        if (!reponseHttp.ok) {
          throw new Error(corps?.erreur ?? `${reponseHttp.status} ${reponseHttp.statusText}`);
        }
        const reponse = corps as OrientationReponse;
        setEtat((precedent) => ({
          ...precedent,
          messages: precedent.messages.map((tour) =>
            tour.id === id ? { ...tour, reponse } : tour,
          ),
        }));
      } catch (erreur) {
        const texte = erreur instanceof Error ? erreur.message : String(erreur);
        setEtat((precedent) => ({
          ...precedent,
          messages: precedent.messages.map((tour) =>
            tour.id === id ? { ...tour, erreur: texte } : tour,
          ),
        }));
      } finally {
        setEnCours(false);
      }
    },
    // `etat.messages` est indispensable ici : sans lui, la fermeture capturerait
    // la liste du premier rendu et l'historique envoyé resterait vide.
    [etat.profil, etat.messages],
  );

  const mettreAJourProfil = useCallback((profil: ProfilCandidat) => {
    setEtat((precedent) => ({ ...precedent, profil }));
  }, []);

  const reinitialiser = useCallback(() => {
    effacerConversation();
    setEtat(etatInitial());
    setBrouillon("");
  }, []);

  const valeur = useMemo<ConversationContextValue>(
    () => ({
      messages: etat.messages,
      profil: etat.profil,
      brouillon,
      enCours,
      definirBrouillon: setBrouillon,
      envoyerMessage,
      mettreAJourProfil,
      reinitialiser,
    }),
    [etat.messages, etat.profil, brouillon, enCours, envoyerMessage, mettreAJourProfil, reinitialiser],
  );

  return <ConversationContext.Provider value={valeur}>{children}</ConversationContext.Provider>;
}

export function useConversation(): ConversationContextValue {
  const contexte = useContext(ConversationContext);
  if (!contexte) {
    throw new Error("useConversation doit être utilisé sous ConversationProvider");
  }
  return contexte;
}
