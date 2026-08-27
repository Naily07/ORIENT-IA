import { ActionsBreakdownChart } from "@/components/admin/ActionsBreakdownChart";
import { ErreurChargement } from "@/components/admin/ErreurChargement";
import { MetricTile } from "@/components/admin/MetricTile";
import { TrendChart } from "@/components/admin/TrendChart";
import { getTableauDeBord, getTendances } from "@/lib/admin-api";
import { getSante } from "@/lib/api-client";

async function chargerDonnees() {
  try {
    const [sante, tableauDeBord, tendances] = await Promise.all([
      getSante(),
      getTableauDeBord(),
      getTendances("jour"),
    ]);
    return { ok: true as const, sante, tableauDeBord, tendances };
  } catch (erreur) {
    return {
      ok: false as const,
      message: erreur instanceof Error ? erreur.message : "Erreur inconnue.",
    };
  }
}

export default async function PageTableauDeBord() {
  const donnees = await chargerDonnees();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold">Tableau de bord</h1>
        <p className="mt-1 text-sm text-neutral-500">
          ORIENT&apos;IA constitue un outil d&apos;aide à l&apos;orientation. Ses recommandations
          ne remplacent ni l&apos;avis d&apos;un conseiller pédagogique ni une décision officielle
          d&apos;admission.
        </p>
      </div>

      {!donnees.ok ? (
        <ErreurChargement
          message={`API injoignable — les indicateurs ci-dessous sont indisponibles (${donnees.message}).`}
        />
      ) : (
        <>
          <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Mentions" value={donnees.sante.corpus.mentions} />
            <MetricTile label="Parcours" value={donnees.sante.corpus.parcours} />
            <MetricTile
              label="Clé LLM"
              value={donnees.sante.cle_llm_configuree ? "configurée" : "absente"}
            />
            <MetricTile label="Modèle" value={donnees.sante.modele} />
          </section>

          <section>
            <h2 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
              Configuration de recherche documentaire
            </h2>
            <p className="mt-1 text-xs text-neutral-500">
              Valeurs calibrées sur le corpus ISPM (RAG-5) — voir la page Mesures pour le balayage
              complet.
            </p>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricTile
                label="Seuil de pertinence"
                value={donnees.tableauDeBord.configuration.rag_seuil_pertinence}
              />
              <MetricTile label="k (passages)" value={donnees.tableauDeBord.configuration.rag_k} />
              <MetricTile
                label="Itérations agent max"
                value={donnees.tableauDeBord.configuration.agent_max_iterations}
              />
              <MetricTile
                label="Seuil de confiance"
                value={donnees.tableauDeBord.configuration.orchestrateur_seuil_confiance}
              />
            </div>
          </section>

          <section>
            <h2 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
              État d&apos;avancement des données
            </h2>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricTile
                label="Matières"
                value={donnees.tableauDeBord.etat_avancement_donnees.matieres}
              />
              <MetricTile
                label="Compétences"
                value={donnees.tableauDeBord.etat_avancement_donnees.competences}
              />
              <MetricTile
                label="Métiers"
                value={donnees.tableauDeBord.etat_avancement_donnees.metiers}
              />
              <MetricTile
                label="Prérequis"
                value={donnees.tableauDeBord.etat_avancement_donnees.prerequis}
              />
            </div>
          </section>

          <section>
            <h2 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
              Évolution
            </h2>
            {donnees.tendances.seaux.length === 0 ? (
              <p className="mt-2 text-sm text-neutral-500">
                Aucune trace enregistrée — posez d&apos;abord une question côté candidat.
              </p>
            ) : (
              <div className="mt-3 grid gap-6 lg:grid-cols-2">
                <div>
                  <p className="mb-1 text-xs text-neutral-500">Volume et latence par jour</p>
                  <TrendChart seaux={donnees.tendances.seaux} />
                </div>
                <div>
                  <p className="mb-1 text-xs text-neutral-500">Répartition des actions par jour</p>
                  <ActionsBreakdownChart seaux={donnees.tendances.seaux} />
                </div>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
