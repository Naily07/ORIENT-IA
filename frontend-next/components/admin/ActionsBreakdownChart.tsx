"use client";

import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { SeauTendance } from "@/lib/types";

// Mêmes teintes que `_COULEURS_NOEUDS` du backoffice Streamlit pour les
// entités du graphe (`admin_api._COULEURS_NOEUDS`), réutilisées ici pour une
// cohérence de palette entre les vues admin — les deux jeux de clés ne se
// recoupent pas, donc pas de confusion possible.
const COULEURS_ACTIONS: Record<string, string> = {
  information: "#2c6fa8",
  recommandation: "#1f6f5c",
  demande_information: "#c98a1f",
  escalade_conseiller: "#a85a25",
  renvoi_administration: "#7a2c5c",
};

export function ActionsBreakdownChart({ seaux }: { seaux: SeauTendance[] }) {
  const actions = Array.from(new Set(seaux.flatMap((s) => Object.keys(s.repartition_actions))));
  const donnees = seaux.map((s) => ({ periode: s.periode, ...s.repartition_actions }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={donnees} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-neutral-200 dark:stroke-neutral-800" />
          <XAxis dataKey="periode" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {actions.map((action) => (
            <Bar key={action} dataKey={action} stackId="actions" fill={COULEURS_ACTIONS[action] ?? "#888888"} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
