"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MetricTile } from "@/components/admin/MetricTile";
import type { Trace } from "@/lib/types";

function actionDe(trace: Trace): string {
  return (trace.decision as { action?: string } | null)?.action ?? "?";
}

function outilsDe(trace: Trace): string[] {
  return (trace.decision as { outils_utilises?: string[] } | null)?.outils_utilises ?? [];
}

/** Port de `back_office.page_observabilite` : métriques, répartition des
 * actions/outils, latence par requête, filtre, détail brut par trace. */
export function ObservabiliteView({ traces }: { traces: Trace[] }) {
  const [filtre, setFiltre] = useState<Set<string>>(new Set());

  const latences = traces.map((t) => (typeof t.latence_ms === "number" ? t.latence_ms : 0));

  const actions = useMemo(() => {
    const compteur = new Map<string, number>();
    for (const trace of traces) {
      const action = actionDe(trace);
      compteur.set(action, (compteur.get(action) ?? 0) + 1);
    }
    return compteur;
  }, [traces]);

  const outils = useMemo(() => {
    const compteur = new Map<string, number>();
    for (const trace of traces) {
      for (const outil of outilsDe(trace)) compteur.set(outil, (compteur.get(outil) ?? 0) + 1);
    }
    return compteur;
  }, [traces]);

  if (traces.length === 0) {
    return (
      <p className="text-sm text-neutral-500">
        Aucune trace enregistrée — posez d&apos;abord une question dans le front-office.
      </p>
    );
  }

  const donneesActions = Array.from(actions.entries()).map(([action, nombre]) => ({
    action,
    nombre,
  }));
  const donneesOutils = Array.from(outils.entries()).map(([outil, nombre]) => ({ outil, nombre }));
  const donneesLatence = [...traces]
    .reverse()
    .map((t, index) => ({ index, latence: typeof t.latence_ms === "number" ? t.latence_ms : 0 }));

  const tracesFiltrees =
    filtre.size === 0 ? traces : traces.filter((t) => filtre.has(actionDe(t)));

  function basculer(action: string) {
    setFiltre((precedent) => {
      const suivant = new Set(precedent);
      if (suivant.has(action)) suivant.delete(action);
      else suivant.add(action);
      return suivant;
    });
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label="Traces" value={traces.length} />
        <MetricTile
          label="Latence moyenne"
          value={`${Math.round(latences.reduce((a, b) => a + b, 0) / latences.length)} ms`}
        />
        <MetricTile label="Latence max" value={`${Math.max(...latences)} ms`} />
        <MetricTile label="Actions distinctes" value={actions.size} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-xs text-neutral-500">Répartition des actions</p>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={donneesActions} layout="vertical" margin={{ left: 16 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  className="stroke-neutral-200 dark:stroke-neutral-800"
                />
                <XAxis type="number" tick={{ fontSize: 11 }} allowDecimals={false} />
                <YAxis type="category" dataKey="action" tick={{ fontSize: 11 }} width={150} />
                <Tooltip />
                <Bar dataKey="nombre" fill="#1f6f5c" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="mb-1 text-xs text-neutral-500">Outils appelés</p>
          {donneesOutils.length === 0 ? (
            <p className="text-xs text-neutral-500">Aucun outil appelé sur cet échantillon.</p>
          ) : (
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={donneesOutils} layout="vertical" margin={{ left: 16 }}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    className="stroke-neutral-200 dark:stroke-neutral-800"
                  />
                  <XAxis type="number" tick={{ fontSize: 11 }} allowDecimals={false} />
                  <YAxis type="category" dataKey="outil" tick={{ fontSize: 11 }} width={150} />
                  <Tooltip />
                  <Bar dataKey="nombre" fill="#2c6fa8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      <div>
        <p className="mb-1 text-xs text-neutral-500">Latence par requête (ms)</p>
        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={donneesLatence}>
              <CartesianGrid
                strokeDasharray="3 3"
                className="stroke-neutral-200 dark:stroke-neutral-800"
              />
              <XAxis dataKey="index" tick={{ fontSize: 11 }} hide />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="latence" stroke="#5b4b8a" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs text-neutral-500">Filtrer par action</p>
        <div className="flex flex-wrap gap-1.5">
          {Array.from(actions.keys())
            .sort()
            .map((action) => (
              <button
                key={action}
                type="button"
                onClick={() => basculer(action)}
                className={`rounded-full border px-2.5 py-1 text-xs transition-[transform,background-color,color] duration-150 ease-out-strong active:scale-[0.95] ${
                  filtre.has(action)
                    ? "border-neutral-900 bg-neutral-900 text-white dark:border-white dark:bg-white dark:text-neutral-900"
                    : "border-neutral-300 text-neutral-600 dark:border-neutral-700 dark:text-neutral-300"
                }`}
              >
                {action}
              </button>
            ))}
        </div>
      </div>

      <div className="space-y-2">
        <h2 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
          Détail des traces
        </h2>
        {tracesFiltrees.map((trace, index) => (
          <details
            key={`${trace.trace_id}-${index}`}
            className="rounded-lg border border-neutral-200 px-3 py-2 text-sm dark:border-neutral-800"
          >
            <summary className="cursor-pointer">
              {String(trace.horodatage).slice(0, 19)} · {actionDe(trace)} ·{" "}
              {String(trace.latence_ms)} ms
            </summary>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-neutral-100 p-3 text-xs dark:bg-neutral-900">
              {JSON.stringify(trace, null, 2)}
            </pre>
          </details>
        ))}
      </div>
    </div>
  );
}
