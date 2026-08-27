/**
 * Miroir du dict `ACTIONS` de `front_office.py` : traduit le vocabulaire
 * technique (`schemas.Action`) en ce qu'un candidat comprend. La clé
 * technique reste visible dans le JSON brut, pour le jury.
 */
import type { Action } from "@/lib/types";

export type TonaliteAction = "info" | "succes" | "attention" | "violet";

export interface LibelleAction {
  libelle: string;
  icone: string; // nom d'icône lucide-react
  tonalite: TonaliteAction;
}

export const LIBELLES_ACTIONS: Record<Action, LibelleAction> = {
  information: { libelle: "Réponse à votre question", icone: "BookOpen", tonalite: "info" },
  recommandation: { libelle: "Parcours suggérés pour vous", icone: "Target", tonalite: "succes" },
  demande_information: {
    libelle: "Il me manque des informations",
    icone: "HelpCircle",
    tonalite: "attention",
  },
  escalade_conseiller: {
    libelle: "À voir avec un conseiller pédagogique",
    icone: "GraduationCap",
    tonalite: "attention",
  },
  renvoi_administration: {
    libelle: "À voir avec l'administration de l'ISPM",
    icone: "Landmark",
    tonalite: "violet",
  },
};

export function libelleAction(action: string): LibelleAction {
  return (
    LIBELLES_ACTIONS[action as Action] ?? { libelle: action, icone: "Circle", tonalite: "info" }
  );
}
