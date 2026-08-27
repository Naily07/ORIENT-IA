import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { jetonSessionValide, NOM_COOKIE } from "@/lib/session";

/**
 * Vérification optimiste avant tout accès à `/admin/*` — Next.js 16 a
 * renommé middleware en Proxy (même mécanique, nouveau nom). La vérification
 * réelle vit dans `lib/dal.ts` (défense en profondeur, comme recommandé par
 * la doc Next.js) ; ce fichier ne fait qu'éviter un aller-retour de rendu
 * inutile pour un visiteur non authentifié.
 */
export async function proxy(request: NextRequest) {
  const chemin = request.nextUrl.pathname;

  if (!chemin.startsWith("/admin") || chemin === "/admin/login") {
    return NextResponse.next();
  }

  // Même court-circuit que `noyau.exiger_acces_admin()` : accès ouvert sans
  // ORIENTIA_ADMIN_CODE, annoncé comme tel dans l'UI plutôt que caché.
  if (!process.env.ORIENTIA_ADMIN_CODE) {
    return NextResponse.next();
  }

  const jeton = request.cookies.get(NOM_COOKIE)?.value;
  if (!(await jetonSessionValide(jeton))) {
    return NextResponse.redirect(new URL("/admin/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/admin/:path*"],
};
