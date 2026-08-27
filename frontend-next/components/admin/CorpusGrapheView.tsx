"use client";

import { useState } from "react";

import { CorpusGraph } from "@/components/admin/CorpusGraph";
import { DataTable } from "@/components/admin/DataTable";
import type { CorpusFormations, GrapheReponse } from "@/lib/types";

const ONGLETS = ["Corpus structuré", "Graphe (ontologie)"] as const;

export function CorpusGrapheView({
  corpus,
  graphe,
}: {
  corpus: CorpusFormations;
  graphe: GrapheReponse;
}) {
  const [onglet, setOnglet] = useState<(typeof ONGLETS)[number]>(ONGLETS[0]);

  // Les entités du corpus (Mention, Parcours...) n'ont pas de signature
  // d'index : DataTable les affiche génériquement, d'où ce passage explicite
  // par `unknown` plutôt qu'un typage structurel qui n'apporterait rien ici.
  const categories: { titre: string; donnees: Record<string, unknown>[] }[] = [
    { titre: "Mentions", donnees: corpus.mentions as unknown as Record<string, unknown>[] },
    { titre: "Parcours", donnees: corpus.parcours as unknown as Record<string, unknown>[] },
    {
      titre: "Prérequis d'admission",
      donnees: corpus.prerequis as unknown as Record<string, unknown>[],
    },
    { titre: "Matières", donnees: corpus.matieres as unknown as Record<string, unknown>[] },
    { titre: "Compétences", donnees: corpus.competences as unknown as Record<string, unknown>[] },
    { titre: "Métiers", donnees: corpus.metiers as unknown as Record<string, unknown>[] },
  ];

  return (
    <div>
      <div className="mb-4 flex gap-1 border-b border-neutral-200 dark:border-neutral-800">
        {ONGLETS.map((valeur) => (
          <button
            key={valeur}
            type="button"
            onClick={() => setOnglet(valeur)}
            className={`px-3 py-2 text-sm transition-colors duration-150 ${
              onglet === valeur
                ? "border-b-2 border-neutral-900 font-medium dark:border-white"
                : "text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100"
            }`}
          >
            {valeur}
          </button>
        ))}
      </div>

      {onglet === "Corpus structuré" ? (
        <div className="space-y-6">
          {categories.map(({ titre, donnees }) => (
            <div key={titre}>
              <p className="mb-1 text-sm font-medium">
                {titre} — {donnees.length}
              </p>
              {donnees.length > 0 ? (
                <DataTable colonnes={Object.keys(donnees[0])} lignes={donnees} />
              ) : (
                <p className="text-xs text-neutral-500">
                  Pas encore collecté (BACKLOG.md, DATA-1).
                </p>
              )}
            </div>
          ))}
        </div>
      ) : (
        <CorpusGraph graphe={graphe} />
      )}
    </div>
  );
}
