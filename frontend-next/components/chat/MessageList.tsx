"use client";

import { useEffect, useRef } from "react";

import { useConversation } from "@/components/chat/ConversationProvider";
import { MessageBubble } from "@/components/chat/MessageBubble";
import type { Marqueurs } from "@/lib/markers";

export function MessageList({ marqueurs }: { marqueurs: Marqueurs }) {
  const { messages } = useConversation();
  const finRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    finRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center px-6 text-center text-sm text-neutral-500">
        <p>
          Décrivez ce qui vous intéresse et posez votre question.
          <br />
          L&apos;assistant s&apos;appuie sur les documents de l&apos;ISPM et sur un modèle
          entraîné, et vous dira toujours sur quoi il s&apos;appuie — et ce qu&apos;il ignore.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-6 overflow-y-auto px-1 py-4">
      {messages.map((tour) => (
        <MessageBubble key={tour.id} tour={tour} marqueurs={marqueurs} />
      ))}
      <div ref={finRef} />
    </div>
  );
}
