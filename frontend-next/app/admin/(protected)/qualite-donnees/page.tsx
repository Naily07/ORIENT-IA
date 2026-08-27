import { AlertTriangle, CheckCircle2, Info } from "lucide-react";

import { DataTable } from "@/components/admin/DataTable";
import { ErreurChargement } from "@/components/admin/ErreurChargement";
import { MetricTile } from "@/components/admin/MetricTile";
import { getQualiteDonnees } from "@/lib/admin-api";
import type { QualiteDonneesReponse } from "@/lib/types";

function colonnesDe(lignes: Record<string, unknown>[]): string[] {
  const colonnes = new Set<string>();
  for (const ligne of lignes) {
    for (const cle of Object.keys(ligne)) colonnes.add(cle);
  }
  return Array.from(colonnes);
}

// Le chargement (et son échec possible) reste isolé dans ce try/catch, sans
// JSX à l'intérieur : React ne rend pas les éléments de façon synchrone, donc
// un try/catch qui engloberait aussi le rendu ne rattraperait pas réellement
// les erreurs de rendu (voir react-hooks/error-boundaries).
async function chargerDonnees() {
  try {
    const donnees = await getQualiteDonnees();
    return { ok: true as const, donnees };
  } catch (erreur) {
    return {
      ok: false as const,
      message: erreur instanceof Error ? erreur.message : "Erreur inconnue.",
    };
  }
}

export default async function PageQualiteDonnees() {
  const resultat = await chargerDonnees();
  if (!resultat.ok) {
    return <ErreurChargement message={`Données indisponibles : ${resultat.message}`} />;
  }

  const donnees: QualiteDonneesReponse = resultat.donnees;
  const statuts = new Map<string, number>();
  for (const entree of donnees.registre_sources) {
    statuts.set(entree.statut, (statuts.get(entree.statut) ?? 0) + 1);
  }
  const sourcesAvecLimites = donnees.registre_sources.filter((e) => e.limites.length > 0);

  return (
    <div className="space-y-8">
        <div>
          <h1 className="text-xl font-semibold">Qualité et traçabilité des données</h1>
          <p className="mt-1 text-sm text-neutral-500">
            Contrôles déterministes, sans LLM : cohérence structurelle du corpus (ONTO-4) et
            provenance de chaque information (§4 du sujet).
          </p>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <MetricTile label="Constats" value={donnees.incoherences.length} />
          <MetricTile label="Données non collectées" value={donnees.donnees_manquantes.length} />
          <MetricTile label="Contradictions réelles" value={donnees.contradictions.length} />
        </div>

        <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900 dark:border-blue-900/40 dark:bg-blue-950/40 dark:text-blue-200">
          <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <p>
            <strong>La distinction est portée par l&apos;outil lui-même</strong>, pas par cette
            interface. Une donnée pas encore collectée (DATA-1) n&apos;est pas un défaut de
            fiabilité du corpus.
          </p>
        </div>

        <section>
          <h2 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Contradictions à traiter
          </h2>
          {donnees.contradictions.length > 0 ? (
            <div className="mt-2">
              <DataTable
                colonnes={colonnesDe(donnees.contradictions)}
                lignes={donnees.contradictions}
              />
            </div>
          ) : (
            <p className="mt-2 flex items-center gap-2 text-sm text-emerald-700 dark:text-emerald-400">
              <CheckCircle2 className="size-4" aria-hidden="true" />
              Aucune contradiction structurelle détectée dans le corpus.
            </p>
          )}
        </section>

        {donnees.donnees_manquantes.length > 0 && (
          <details className="rounded-lg border border-neutral-200 px-3 py-2 text-sm dark:border-neutral-800">
            <summary className="cursor-pointer">
              Données non encore collectées ({donnees.donnees_manquantes.length})
            </summary>
            <div className="mt-2">
              <DataTable
                colonnes={colonnesDe(donnees.donnees_manquantes)}
                lignes={donnees.donnees_manquantes}
              />
            </div>
          </details>
        )}

        <section>
          <h2 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Registre des sources (§4)
          </h2>
          {donnees.registre_sources.length === 0 ? (
            <p className="mt-2 text-sm text-amber-600 dark:text-amber-400">
              Registre des sources vide.
            </p>
          ) : (
            <>
              <div
                className="mt-3 grid gap-3"
                style={{ gridTemplateColumns: `repeat(${Math.max(statuts.size, 1)}, minmax(0, 1fr))` }}
              >
                {Array.from(statuts.entries())
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([statut, nombre]) => (
                    <MetricTile key={statut} label={statut[0].toUpperCase() + statut.slice(1)} value={nombre} />
                  ))}
              </div>
              <div className="mt-3">
                <DataTable
                  colonnes={["id", "titre", "statut", "consultée le", "limites connues", "url"]}
                  lignes={donnees.registre_sources.map((e) => ({
                    id: e.id,
                    titre: e.titre,
                    statut: e.statut,
                    "consultée le": e.date_consultation,
                    "limites connues": e.limites.length,
                    url: e.url,
                  }))}
                />
              </div>
            </>
          )}
        </section>

        {donnees.references_orphelines.length > 0 ? (
          <p className="flex items-start gap-2 text-sm text-red-700 dark:text-red-400">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            Références de source orphelines : {donnees.references_orphelines.join(", ")}
          </p>
        ) : (
          <p className="flex items-center gap-2 text-sm text-emerald-700 dark:text-emerald-400">
            <CheckCircle2 className="size-4" aria-hidden="true" />
            Toute donnée du corpus qui déclare une source pointe vers une entrée réelle du
            registre.
          </p>
        )}

        {sourcesAvecLimites.length > 0 && (
          <details className="rounded-lg border border-neutral-200 px-3 py-2 text-sm dark:border-neutral-800">
            <summary className="cursor-pointer">
              Limites déclarées par source — à lire avant toute citation
            </summary>
            <div className="mt-2 space-y-3">
              {sourcesAvecLimites.map((e) => (
                <div key={e.id}>
                  <p className="font-medium">
                    {e.id} ({e.statut})
                  </p>
                  <ul className="list-disc pl-5">
                    {e.limites.map((limite) => (
                      <li key={limite}>{limite}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </details>
        )}
    </div>
  );
}
