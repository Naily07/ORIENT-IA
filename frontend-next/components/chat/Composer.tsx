"use client";

import { SendHorizontal } from "lucide-react";
import { type FormEvent, type KeyboardEvent } from "react";

import { useConversation } from "@/components/chat/ConversationProvider";

export function Composer() {
  const { brouillon, definirBrouillon, envoyerMessage, enCours } = useConversation();

  async function soumettre(evenement: FormEvent) {
    evenement.preventDefault();
    const texte = brouillon.trim();
    if (!texte || enCours) return;
    definirBrouillon("");
    await envoyerMessage(texte);
  }

  function gererTouche(evenement: KeyboardEvent<HTMLTextAreaElement>) {
    if (evenement.key === "Enter" && !evenement.shiftKey) {
      evenement.preventDefault();
      void soumettre(evenement);
    }
  }

  return (
    <form
      onSubmit={soumettre}
      className="flex items-end gap-2 border-t border-neutral-200 p-3 dark:border-neutral-800"
    >
      <textarea
        value={brouillon}
        onChange={(evenement) => definirBrouillon(evenement.target.value)}
        onKeyDown={gererTouche}
        placeholder="Quel parcours correspond à mon profil ?"
        rows={1}
        className="max-h-40 flex-1 resize-none rounded-xl border border-neutral-300 bg-white px-3 py-2 text-sm focus:ring-2 focus:ring-neutral-900 focus:outline-none dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-white"
      />
      <button
        type="submit"
        disabled={!brouillon.trim() || enCours}
        className="flex size-10 shrink-0 items-center justify-center rounded-full bg-neutral-900 text-white transition-transform duration-150 ease-out-strong active:scale-[0.93] disabled:opacity-40 disabled:active:scale-100 dark:bg-white dark:text-neutral-900"
        aria-label="Envoyer"
      >
        <SendHorizontal className="size-4" aria-hidden="true" />
      </button>
    </form>
  );
}
