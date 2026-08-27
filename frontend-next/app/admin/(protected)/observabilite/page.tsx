import { ErreurChargement } from "@/components/admin/ErreurChargement";
import { LimiteSelector } from "@/components/admin/LimiteSelector";
import { ObservabiliteView } from "@/components/admin/ObservabiliteView";
import { getTraces } from "@/lib/admin-api";

async function chargerDonnees(limite: number) {
  try {
    const traces = await getTraces(limite);
    return { ok: true as const, traces };
  } catch (erreur) {
    return {
      ok: false as const,
      message: erreur instanceof Error ? erreur.message : "Erreur inconnue.",
    };
  }
}

export default async function PageObservabilite({
  searchParams,
}: {
  searchParams: Promise<{ limite?: string }>;
}) {
  const { limite: limiteBrute } = await searchParams;
  const limite = Math.min(200, Math.max(5, Number(limiteBrute) || 30));
  const resultat = await chargerDonnees(limite);

  if (!resultat.ok) {
    return <ErreurChargement message={`Traces indisponibles : ${resultat.message}`} />;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Observabilité</h1>
          <p className="mt-1 text-sm text-neutral-500">
            Traces du pipeline (§15 du sujet) : question, contexte, décision, outils, latence —
            telles qu&apos;écrites par le backend, sans retraitement.
          </p>
        </div>
        <LimiteSelector limite={limite} />
      </div>
      <ObservabiliteView traces={resultat.traces} />
    </div>
  );
}
