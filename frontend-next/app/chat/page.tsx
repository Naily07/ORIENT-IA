import { GraduationCap } from "lucide-react";

import { Composer } from "@/components/chat/Composer";
import { ConversationProvider } from "@/components/chat/ConversationProvider";
import { MentionBanner } from "@/components/chat/MentionBanner";
import { MessageList } from "@/components/chat/MessageList";
import { NouvelleConversationButton } from "@/components/chat/NouvelleConversationButton";
import { ProfilPanel } from "@/components/chat/ProfilPanel";
import { ScenarioMenu } from "@/components/chat/ScenarioMenu";
import { getSante } from "@/lib/api-client";
import { MARQUEURS_REPLI, type Marqueurs } from "@/lib/markers";
import { MENTION_OBLIGATOIRE_REPLI } from "@/lib/mention";

async function chargerEnTete(): Promise<{ mention: string; marqueurs: Marqueurs }> {
  try {
    const sante = await getSante();
    return {
      mention: sante.mention_obligatoire,
      marqueurs: {
        avertissementNonExploitable: sante.avertissement_non_exploitable,
        marqueurRegleAdmission: sante.marqueur_regle_admission,
      },
    };
  } catch {
    // API injoignable : la mention reste affichée (jamais conditionnée à un
    // /health réussi) avec un repli local — voir lib/mention.ts, lib/markers.ts.
    return { mention: MENTION_OBLIGATOIRE_REPLI, marqueurs: MARQUEURS_REPLI };
  }
}

export default async function PageChat() {
  const { mention, marqueurs } = await chargerEnTete();

  return (
    <div className="mx-auto flex h-dvh max-w-4xl flex-col px-4">
      <header className="flex items-center justify-between gap-3 py-4">
        <h1 className="flex items-center gap-2 text-lg font-semibold">
          <GraduationCap className="size-5" aria-hidden="true" />
          Trouver ma formation à l&apos;ISPM
        </h1>
      </header>

      <ConversationProvider>
        <MentionBanner mention={mention} />

        <div className="flex items-center justify-between gap-3 py-3">
          <ScenarioMenu />
          <NouvelleConversationButton />
        </div>

        <div className="grid min-h-0 flex-1 gap-4 pb-4 sm:grid-cols-[1fr_18rem]">
          <div className="flex min-h-0 flex-col rounded-2xl border border-neutral-200 dark:border-neutral-800">
            <MessageList marqueurs={marqueurs} />
            <Composer />
          </div>
          <aside className="hidden overflow-y-auto sm:block">
            <ProfilPanel />
          </aside>
        </div>
      </ConversationProvider>
    </div>
  );
}
