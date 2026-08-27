import { NextResponse } from "next/server";

import { ApiIndisponible, apiPost } from "@/lib/api-client";
import type { OrientationInput, OrientationReponse } from "@/lib/types";

/**
 * Seul point de passage du chat vers `POST /orientation/traiter` (BFF) : le
 * navigateur du candidat ne parle jamais directement à FastAPI, `API_URL`
 * reste un secret serveur. Le pipeline backend reste stateless — ce handler
 * ne fait que relayer `{message, profil}` tel quel, voir
 * `components/chat/ConversationProvider.tsx` pour l'accumulation côté client.
 */
export async function POST(request: Request) {
  let corps: OrientationInput;
  try {
    corps = (await request.json()) as OrientationInput;
  } catch {
    return NextResponse.json({ erreur: "Corps JSON invalide." }, { status: 400 });
  }

  if (!corps.message || !corps.message.trim()) {
    return NextResponse.json({ erreur: "Merci d'écrire votre question." }, { status: 400 });
  }

  try {
    const reponse = await apiPost<OrientationReponse>("/orientation/traiter", corps);
    return NextResponse.json(reponse);
  } catch (erreur) {
    const message = erreur instanceof ApiIndisponible ? erreur.message : "Erreur inattendue.";
    return NextResponse.json({ erreur: message }, { status: 502 });
  }
}
