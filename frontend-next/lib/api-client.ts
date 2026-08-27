/**
 * Client HTTP vers le backend FastAPI — miroir de `frontend/noyau.py`
 * (`_appeler`/`api_get`/`api_post`/`ApiIndisponible`).
 *
 * `server-only` : ce module n'est jamais bundlé côté navigateur. Next.js agit
 * en BFF (Route Handlers / Server Components appellent l'API directement,
 * server-to-server) — le navigateur du candidat ne parle jamais à FastAPI, ce
 * qui évite tout souci de CORS et garde `API_URL` hors du bundle client.
 */
import "server-only";

import type { SanteReponse } from "@/lib/types";

const API_URL = process.env.API_URL ?? "http://localhost:8000";

// Le pipeline enchaîne plusieurs appels LLM lissés — même valeur que
// `DELAI_TRAITEMENT_S` côté Streamlit.
const DELAI_TRAITEMENT_MS = 180_000;
const DELAI_LECTURE_MS = 15_000;

export class ApiIndisponible extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiIndisponible";
  }
}

async function appeler<T>(
  methode: "GET" | "POST",
  chemin: string,
  options: { params?: Record<string, string | number | string[] | undefined>; corps?: unknown; timeoutMs: number },
): Promise<T> {
  const url = new URL(chemin, API_URL);
  for (const [cle, valeur] of Object.entries(options.params ?? {})) {
    if (valeur === undefined) continue;
    if (Array.isArray(valeur)) {
      for (const element of valeur) url.searchParams.append(cle, element);
    } else {
      url.searchParams.set(cle, String(valeur));
    }
  }

  const controleur = new AbortController();
  const declencheurTimeout = setTimeout(() => controleur.abort(), options.timeoutMs);

  let reponse: Response;
  try {
    reponse = await fetch(url, {
      method: methode,
      headers: options.corps !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: options.corps !== undefined ? JSON.stringify(options.corps) : undefined,
      signal: controleur.signal,
      cache: "no-store",
    });
  } catch (erreur) {
    const message = erreur instanceof Error ? erreur.message : String(erreur);
    throw new ApiIndisponible(message);
  } finally {
    clearTimeout(declencheurTimeout);
  }

  if (!reponse.ok) {
    const detail = await reponse.text().catch(() => "");
    throw new ApiIndisponible(`${reponse.status} ${reponse.statusText}${detail ? ` — ${detail}` : ""}`);
  }

  try {
    return (await reponse.json()) as T;
  } catch (erreur) {
    const message = erreur instanceof Error ? erreur.message : String(erreur);
    throw new ApiIndisponible(`réponse illisible : ${message}`);
  }
}

export function apiGet<T>(
  chemin: string,
  params?: Record<string, string | number | string[] | undefined>,
): Promise<T> {
  return appeler<T>("GET", chemin, { params, timeoutMs: DELAI_LECTURE_MS });
}

export function apiPost<T>(chemin: string, corps: unknown, timeoutMs = DELAI_TRAITEMENT_MS): Promise<T> {
  return appeler<T>("POST", chemin, { corps, timeoutMs });
}

/** `GET /health`, utilisé aussi bien par le chat (mention réglementaire) que
 * par le tableau de bord admin — pas un endpoint `/admin/*`, donc pas dans
 * `lib/admin-api.ts`. */
export function getSante(): Promise<SanteReponse> {
  return apiGet<SanteReponse>("/health");
}
