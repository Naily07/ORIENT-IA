"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SeauTendance } from "@/lib/types";

/** Volume et latence moyenne dans le temps — dérivé de
 * `GET /admin/observabilite/tendances`, le cœur du point « évolution des
 * stats » du tableau de bord. */
export function TrendChart({ seaux }: { seaux: SeauTendance[] }) {
  const donnees = seaux.map((s) => ({
    periode: s.periode,
    volume: s.volume,
    latence: s.latence_moyenne_ms,
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={donnees} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-neutral-200 dark:stroke-neutral-800" />
          <XAxis dataKey="periode" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="volume" tick={{ fontSize: 11 }} allowDecimals={false} />
          <YAxis yAxisId="latence" orientation="right" tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line
            yAxisId="volume"
            type="monotone"
            dataKey="volume"
            name="Volume"
            stroke="#1f6f5c"
            strokeWidth={2}
            dot={false}
          />
          <Line
            yAxisId="latence"
            type="monotone"
            dataKey="latence"
            name="Latence moyenne (ms)"
            stroke="#5b4b8a"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
