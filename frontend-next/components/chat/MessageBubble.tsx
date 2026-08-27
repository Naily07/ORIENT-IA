import { AlertCircle, Loader2 } from "lucide-react";

import { DecisionCard } from "@/components/chat/DecisionCard";
import type { TourConversation } from "@/lib/conversation-storage";
import type { Marqueurs } from "@/lib/markers";

export function MessageBubble({
  tour,
  marqueurs,
}: {
  tour: TourConversation;
  marqueurs: Marqueurs;
}) {
  return (
    <div className="motion-safe:animate-fade-in-up space-y-3">
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-tr-sm bg-neutral-900 px-4 py-2.5 text-sm text-white dark:bg-white dark:text-neutral-900">
        {tour.message}
      </div>
      <div className="mr-auto max-w-[92%] rounded-2xl rounded-tl-sm border border-neutral-200 bg-white px-4 py-3 dark:border-neutral-800 dark:bg-neutral-950">
        {tour.reponse ? (
          <DecisionCard reponse={tour.reponse} marqueurs={marqueurs} />
        ) : tour.erreur ? (
          <div className="flex items-start gap-2 text-sm text-red-700 dark:text-red-400">
            <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            <p>La demande n&apos;a pas pu être traitée : {tour.erreur}</p>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-neutral-500">
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
            Analyse de votre profil, du corpus et du modèle…
          </div>
        )}
      </div>
    </div>
  );
}
