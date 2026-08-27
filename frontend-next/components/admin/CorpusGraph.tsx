"use client";

import "@xyflow/react/dist/style.css";

import { Background, Controls, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { useMemo, useState } from "react";

import { calculerLayout, HAUTEUR_NOEUD, LARGEUR_NOEUD } from "@/lib/graph-layout";
import type { GrapheReponse, TypeEntiteGraphe } from "@/lib/types";

const TOUS_LES_TYPES: TypeEntiteGraphe[] = [
  "Parcours",
  "Mention",
  "Prerequis",
  "Competence",
  "Metier",
  "Matiere",
];

/** Port de `back_office._graphe_dot` / `page_corpus_graphe` (onglet Graphe) —
 * rendu interactif (`@xyflow/react` + `dagre`) plutôt que DOT/graphviz, qui
 * n'a pas de renderer React. */
export function CorpusGraph({ graphe }: { graphe: GrapheReponse }) {
  const [typesRetenus, setTypesRetenus] = useState<Set<TypeEntiteGraphe>>(
    new Set<TypeEntiteGraphe>(["Parcours", "Mention", "Prerequis"]),
  );

  const noeudsFiltres = useMemo(
    () => graphe.noeuds.filter((n) => typesRetenus.has(n.type)),
    [graphe.noeuds, typesRetenus],
  );
  const idsGardes = useMemo(() => new Set(noeudsFiltres.map((n) => n.id)), [noeudsFiltres]);
  const relationsFiltrees = useMemo(
    () => graphe.relations.filter((r) => idsGardes.has(r.source) && idsGardes.has(r.cible)),
    [graphe.relations, idsGardes],
  );
  const positions = useMemo(
    () => calculerLayout(noeudsFiltres, relationsFiltrees),
    [noeudsFiltres, relationsFiltrees],
  );

  const nodes: Node[] = noeudsFiltres.map((n) => ({
    id: n.id,
    position: positions[n.id] ?? { x: 0, y: 0 },
    data: { label: n.nom.length > 38 ? `${n.nom.slice(0, 35)}…` : n.nom },
    style: {
      border: `2px solid ${n.couleur}`,
      borderRadius: 8,
      padding: 8,
      width: LARGEUR_NOEUD,
      height: HAUTEUR_NOEUD,
      fontSize: 12,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      background: "var(--background)",
      color: "var(--foreground)",
    },
  }));

  const edges: Edge[] = relationsFiltrees.map((r, index) => ({
    id: `${r.source}->${r.cible}-${index}`,
    source: r.source,
    target: r.cible,
    label: r.relation,
    style: { stroke: "#9ca3af" },
    labelStyle: { fontSize: 10 },
  }));

  function basculerType(type: TypeEntiteGraphe) {
    setTypesRetenus((precedent) => {
      const suivant = new Set(precedent);
      if (suivant.has(type)) suivant.delete(type);
      else suivant.add(type);
      return suivant;
    });
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-neutral-500">
        Seules les relations réellement présentes dans le corpus sont tracées. Elles
        apparaîtront au fur et à mesure que matières, compétences et débouchés seront collectés
        (DATA-1).
      </p>
      <div className="flex flex-wrap gap-1.5">
        {TOUS_LES_TYPES.map((type) => (
          <button
            key={type}
            type="button"
            onClick={() => basculerType(type)}
            className={`rounded-full border px-2.5 py-1 text-xs transition-[transform,background-color,color] duration-150 ease-out-strong active:scale-[0.95] ${
              typesRetenus.has(type)
                ? "border-neutral-900 bg-neutral-900 text-white dark:border-white dark:bg-white dark:text-neutral-900"
                : "border-neutral-300 text-neutral-600 dark:border-neutral-700 dark:text-neutral-300"
            }`}
          >
            {type}
          </button>
        ))}
      </div>
      {nodes.length === 0 ? (
        <p className="text-sm text-neutral-500">Sélectionnez au moins un type d&apos;entité.</p>
      ) : (
        <div className="h-[520px] w-full overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-800">
          {/* Pas de `proOptions={{ hideAttribution: true }}` : masquer l'attribution React
              Flow sans licence Pro est contraire à leurs conditions d'utilisation. */}
          <ReactFlow nodes={nodes} edges={edges} fitView>
            <Background />
            <Controls />
          </ReactFlow>
        </div>
      )}
    </div>
  );
}
