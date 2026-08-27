/**
 * Session admin — cookie signé (JWT HS256 via `jose`), pattern recommandé par
 * la doc officielle Next.js pour l'auth stateless (voir
 * `node_modules/next/dist/docs/01-app/02-guides/authentication.md`).
 *
 * Miroir fonctionnel de `noyau.exiger_acces_admin()` : même secret partagé
 * (`ORIENTIA_ADMIN_CODE`), même volonté de rester simple (comparaison directe
 * en clair, pas de table d'utilisateurs) — le prototype ne manipule aucune
 * donnée personnelle. Seule différence : un cookie `httpOnly` plutôt qu'un
 * `st.session_state` local à l'onglet Streamlit, donc une session qui
 * survient à une navigation entre pages admin.
 */
import "server-only";

import { jwtVerify, SignJWT } from "jose";

export const NOM_COOKIE = "orientia_admin_session";
const DUREE_SESSION = "12h";
const DUREE_SESSION_S = 60 * 60 * 12;

class ConfigurationSessionManquante extends Error {
  constructor() {
    super(
      "SESSION_SECRET est absent ou trop court (32 caractères minimum) : impossible de signer " +
        "une session admin. Générer une valeur avec `openssl rand -base64 32` et la placer dans " +
        "le `.env` racine.",
    );
  }
}

function cleSecrete(): Uint8Array {
  const secret = process.env.SESSION_SECRET ?? "";
  if (secret.length < 32) throw new ConfigurationSessionManquante();
  return new TextEncoder().encode(secret);
}

export async function creerJetonSession(): Promise<string> {
  return new SignJWT({ role: "admin" })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(DUREE_SESSION)
    .sign(cleSecrete());
}

export async function jetonSessionValide(jeton: string | undefined): Promise<boolean> {
  if (!jeton) return false;
  try {
    await jwtVerify(jeton, cleSecrete(), { algorithms: ["HS256"] });
    return true;
  } catch {
    return false;
  }
}

export const OPTIONS_COOKIE_SESSION = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  path: "/",
  maxAge: DUREE_SESSION_S,
};
