"use client";

import { RotateCcw } from "lucide-react";

import { useConversation } from "@/components/chat/ConversationProvider";

export function NouvelleConversationButton() {
  const { messages, reinitialiser } = useConversation();

  if (messages.length === 0) return null;

  return (
    <button
      type="button"
      onClick={reinitialiser}
      className="inline-flex items-center gap-1.5 rounded-full border border-neutral-300 px-3 py-1.5 text-xs font-medium text-neutral-600 transition-transform duration-150 ease-out-strong active:scale-[0.96] dark:border-neutral-700 dark:text-neutral-300"
    >
      <RotateCcw className="size-3.5" aria-hidden="true" />
      Nouvelle conversation
    </button>
  );
}
