"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { z } from "zod";

import { creerJetonSession, NOM_COOKIE, OPTIONS_COOKIE_SESSION } from "@/lib/session";

const SchemaConnexion = z.object({
  code: z.string().min(1, "Le code d'accès est requis."),
});

export interface EtatConnexion {
  erreur?: string;
}

export async function connexionAdmin(
  _etatPrecedent: EtatConnexion,
  formData: FormData,
): Promise<EtatConnexion> {
  const analyse = SchemaConnexion.safeParse({ code: formData.get("code") });
  if (!analyse.success) {
    return { erreur: analyse.error.issues[0]?.message ?? "Code invalide." };
  }

  const codeAttendu = process.env.ORIENTIA_ADMIN_CODE ?? "";
  if (!codeAttendu || analyse.data.code !== codeAttendu) {
    return { erreur: "Code incorrect." };
  }

  let jeton: string;
  try {
    jeton = await creerJetonSession();
  } catch (erreur) {
    const message = erreur instanceof Error ? erreur.message : "Erreur de configuration.";
    return { erreur: message };
  }

  (await cookies()).set(NOM_COOKIE, jeton, OPTIONS_COOKIE_SESSION);
  redirect("/admin/tableau-de-bord");
}

export async function deconnexionAdmin(): Promise<void> {
  (await cookies()).delete(NOM_COOKIE);
  redirect("/admin/login");
}
