"use client";

import { ChevronDown, Sparkles } from "lucide-react";
import { useState } from "react";

import { useConversation } from "@/components/chat/ConversationProvider";
import { profilDuScenario, SCENARIOS_DEMO, type ScenarioDemo } from "@/lib/scenarios";

/** Menu « Essayer un exemple » — port du sélecteur de scénarios de
 * `front_office.page()`, en menu plutôt qu'en formulaire pré-rempli. */
export function ScenarioMenu() {
  const [ouvert, setOuvert] = useState(false);
  const { definirBrouillon, mettreAJourProfil } = useConversation();

  function choisir(scenario: ScenarioDemo) {
    definirBrouillon(scenario.message);
    mettreAJourProfil(profilDuScenario(scenario));
    setOuvert(false);
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOuvert((valeur) => !valeur)}
        className="inline-flex items-center gap-1.5 rounded-full border border-neutral-300 px-3 py-1.5 text-xs font-medium text-neutral-600 transition-transform duration-150 ease-out-strong active:scale-[0.96] dark:border-neutral-700 dark:text-neutral-300"
      >
        <Sparkles className="size-3.5" aria-hidden="true" />
        Essayer un exemple
        <ChevronDown className="size-3" aria-hidden="true" />
      </button>
      {ouvert && (
        <>
          <button
            type="button"
            aria-label="Fermer le menu"
            onClick={() => setOuvert(false)}
            className="fixed inset-0 z-10 cursor-default"
          />
          <div
            style={{ transformOrigin: "top left" }}
            className="motion-safe:animate-scale-in absolute z-20 mt-2 w-80 rounded-xl border border-neutral-200 bg-white p-1 shadow-lg dark:border-neutral-800 dark:bg-neutral-900"
          >
            {SCENARIOS_DEMO.map((scenario) => (
              <button
                key={scenario.titre}
                type="button"
                onClick={() => choisir(scenario)}
                className="block w-full rounded-lg px-3 py-2 text-left text-xs transition-colors duration-100 hover:bg-neutral-100 dark:hover:bg-neutral-800"
              >
                {scenario.titre}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
