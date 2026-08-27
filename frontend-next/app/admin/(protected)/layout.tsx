import { redirect } from "next/navigation";
import type { ReactNode } from "react";

import { Sidebar } from "@/components/admin/Sidebar";
import { accesAdminAutorise } from "@/lib/dal";

/**
 * Défense en profondeur : `proxy.ts` a déjà redirigé un visiteur non
 * authentifié avant même d'atteindre ce layout (vérification optimiste sur
 * le cookie), mais cette vérification côté serveur reste la source de
 * vérité — voir `lib/dal.ts`. `app/admin/login/` est un segment frère, donc
 * hors de ce groupe de routes protégé.
 */
export default async function LayoutAdminProtege({ children }: { children: ReactNode }) {
  const autorise = await accesAdminAutorise();
  if (!autorise) redirect("/admin/login");

  return (
    <div className="flex min-h-dvh">
      <Sidebar accesOuvert={!process.env.ORIENTIA_ADMIN_CODE} />
      <main className="flex-1 overflow-y-auto p-6">{children}</main>
    </div>
  );
}
