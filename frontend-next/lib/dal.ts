/**
 * Data Access Layer — vérification réelle de la session admin, appelée dans
 * `app/admin/(protected)/layout.tsx`. `proxy.ts` ne fait qu'une vérification
 * optimiste (lecture du cookie avant tout accès disque) ; cette fonction est
 * la défense en profondeur recommandée par la doc Next.js pour l'auth.
 */
import "server-only";

import { cookies } from "next/headers";
import { cache } from "react";

import { jetonSessionValide, NOM_COOKIE } from "@/lib/session";

/** Même court-circuit que `noyau.exiger_acces_admin()` : sans
 * `ORIENTIA_ADMIN_CODE`, l'accès admin est ouvert (et annoncé comme tel dans
 * l'UI — voir `components/admin/Sidebar.tsx`), pas un défaut silencieux. */
export const accesAdminAutorise = cache(async (): Promise<boolean> => {
  if (!process.env.ORIENTIA_ADMIN_CODE) return true;
  const jeton = (await cookies()).get(NOM_COOKIE)?.value;
  return jetonSessionValide(jeton);
});
