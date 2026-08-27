import { ErreurChargement } from "@/components/admin/ErreurChargement";
import { MesuresView } from "@/components/admin/MesuresView";
import { getMesures } from "@/lib/admin-api";

async function chargerDonnees() {
  try {
    const mesures = await getMesures();
    return { ok: true as const, mesures };
  } catch (erreur) {
    return {
      ok: false as const,
      message: erreur instanceof Error ? erreur.message : "Erreur inconnue.",
    };
  }
}

export default async function PageMesures() {
  const resultat = await chargerDonnees();
  if (!resultat.ok) {
    return <ErreurChargement message={`Mesures indisponibles : ${resultat.message}`} />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Mesures et évaluations</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Le sujet exige des résultats mesurés, pas l&apos;affirmation que le système fonctionne.
          Chaque chiffre vient d&apos;un artefact du dépôt, reproductible par la commande
          indiquée.
        </p>
      </div>
      <MesuresView
        ml={resultat.mesures.ml}
        rag={resultat.mesures.rag}
        systeme={resultat.mesures.systeme}
      />
    </div>
  );
}
