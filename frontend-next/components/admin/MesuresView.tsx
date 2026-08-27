"use client";

import { useState } from "react";

import { ArtefactAbsent } from "@/components/admin/ArtefactAbsent";
import { DataTable } from "@/components/admin/DataTable";
import { MetricTile } from "@/components/admin/MetricTile";
import { decimal, pourcentage, versListe, versNombre, versObjet, versTexte } from "@/lib/mesures-lecture";
import type { ArtefactMesure } from "@/lib/types";

const ONGLETS = ["Machine Learning", "Recherche documentaire", "Système de bout en bout"] as const;

/** Port de `back_office._section_ml` — le modèle **servi**, pas la baseline
 * brute (§8 : afficher l'un pour l'autre reproduirait l'écart évalué≠servi). */
function SectionML({ artefact }: { artefact: ArtefactMesure }) {
  if (!artefact.disponible) {
    return <ArtefactAbsent nom="L'évaluation ML" commande={artefact.commande ?? ""} />;
  }
  const resultats = versObjet(artefact.donnees);
  const baseline = versObjet(
    resultats.modele_de_production_calibre ?? resultats.baseline_regression_logistique,
  );
  const apport = versObjet(resultats.apport_de_la_calibration);
  const calibration = versObjet(baseline.calibration);
  const production = versObjet(resultats.chemin_de_production);
  const classement = versObjet(production.classement);
  const stabilite = versObjet(production.stabilite_des_recommandations);
  const matrice = versObjet(baseline.matrice_confusion);
  const labels = versListe(matrice.labels).map((l) => versTexte(l));
  const lignesMatrice = versListe(matrice.matrice) as number[][];
  const ece = calibration.ece;
  const ecart = calibration.ecart_signe_confiance_moins_exactitude;

  return (
    <div className="space-y-6">
      {versTexte(resultats.avertissement) && (
        <p className="text-xs text-neutral-500">{versTexte(resultats.avertissement)}</p>
      )}
      {apport.ece_avant !== undefined && (
        <p className="text-xs text-neutral-500">
          Calibration isotonique : ECE {decimal(apport.ece_avant)} →{" "}
          <strong>{decimal(apport.ece_apres)}</strong>. {versTexte(apport.lecture)}
        </p>
      )}

      <div>
        <p className="mb-2 text-sm font-medium">Modèle de production — régression logistique calibrée</p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricTile label="Exactitude" value={pourcentage(baseline.exactitude, 1)} />
          <MetricTile label="F1 macro" value={decimal(baseline.f1_macro)} />
          <MetricTile label="MRR" value={decimal(baseline.mrr)} />
          <MetricTile label="NDCG@3" value={decimal(baseline.ndcg_3)} />
        </div>
      </div>

      <div>
        <p className="mb-2 text-sm font-medium">Calibration — la confiance affichée est-elle juste ?</p>
        <div className="grid grid-cols-3 gap-3">
          <MetricTile label="ECE" value={ece !== undefined ? decimal(ece) : "—"} />
          <MetricTile label="Score de Brier" value={decimal(calibration.score_de_brier)} />
          <MetricTile label="PR-AUC macro" value={decimal(baseline.pr_auc_macro)} />
        </div>
        {typeof ece === "number" && ece > 0.05 && typeof ecart === "number" && (
          <p className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/40 dark:text-amber-200">
            {ecart > 0 ? (
              <>
                <strong>Sur-confiance de {pourcentage(Math.abs(ecart))}</strong> : le modèle
                annonce en moyenne une certitude supérieure à sa réussite réelle — le sens
                dangereux.
              </>
            ) : (
              <>
                <strong>Sous-confiance de {pourcentage(Math.abs(ecart))}</strong> : le modèle
                réussit plus souvent qu&apos;il ne l&apos;annonce.
              </>
            )}
          </p>
        )}
        {typeof ece === "number" && ece <= 0.05 && (
          <p className="mt-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-900 dark:border-emerald-900/40 dark:bg-emerald-950/40 dark:text-emerald-200">
            Calibration correcte (ECE {decimal(ece)}) : le score affiché correspond, en moyenne, à
            la fréquence de réussite observée.
          </p>
        )}
      </div>

      {production.lecture !== undefined && (
        <div>
          <p className="mb-1 text-sm font-medium">Chemin réellement servi (analyser_profil, §8)</p>
          <p className="mb-2 text-xs text-neutral-500">{versTexte(production.lecture)}</p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Top-1" value={pourcentage(classement.top_1, 1)} />
            <MetricTile label="Top-3" value={pourcentage(classement.top_3, 1)} />
            <MetricTile label="MRR" value={decimal(classement.mrr)} />
            <MetricTile
              label="Profils jugés inexploitables"
              value={versNombre(classement.profils_juges_inexploitables)}
            />
          </div>

          {versNombre(stabilite.profils_compares) > 0 && (
            <div className="mt-3">
              <p className="mb-2 text-sm font-medium">
                Stabilité des recommandations (§7) — retrait d&apos;un trait déclaré
              </p>
              <div className="grid grid-cols-3 gap-3">
                <MetricTile label="Parcours de tête inchangé" value={pourcentage(stabilite.top_1_inchange)} />
                <MetricTile
                  label="Sélection présentée inchangée"
                  value={pourcentage(stabilite.selection_presentee_inchangee)}
                />
                <MetricTile
                  label="Top-3 fixe inchangé (référence)"
                  value={pourcentage(stabilite.top_3_fixe_inchange)}
                />
              </div>
              {versNombre(stabilite.selection_presentee_inchangee, 1) < 0.85 && (
                <p className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/40 dark:text-amber-200">
                  La sélection présentée change pour{" "}
                  {pourcentage(1 - versNombre(stabilite.selection_presentee_inchangee))} des
                  profils au retrait d&apos;un seul trait déclaré : à ne pas présenter comme un
                  classement arrêté.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {labels.length > 0 && lignesMatrice.length > 0 && (
        <details className="rounded-lg border border-neutral-200 px-3 py-2 text-sm dark:border-neutral-800">
          <summary className="cursor-pointer">Matrice de confusion (baseline)</summary>
          <div className="mt-2">
            <DataTable
              colonnes={["vraie classe", ...labels]}
              lignes={lignesMatrice.map((ligne, i) => ({
                "vraie classe": labels[i],
                ...Object.fromEntries(labels.map((predite, j) => [predite, ligne[j]])),
              }))}
            />
          </div>
        </details>
      )}
    </div>
  );
}

/** Port de `back_office._section_rag`. */
function SectionRAG({ artefact }: { artefact: ArtefactMesure }) {
  if (!artefact.disponible) {
    return <ArtefactAbsent nom="La calibration RAG" commande={artefact.commande ?? ""} />;
  }
  const resultats = versObjet(artefact.donnees);
  const meilleur = versObjet(resultats.meilleur_compromis);
  const mesures = versListe(resultats.mesures) as Record<string, unknown>[];

  return (
    <div className="space-y-4">
      <div>
        <p className="mb-2 text-sm font-medium">Balayage mesuré du seuil de pertinence et de k (RAG-5)</p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricTile label="Meilleur compromis — seuil" value={versTexte(String(meilleur.seuil ?? "?"))} />
          <MetricTile label="Meilleur compromis — k" value={versTexte(String(meilleur.k ?? "?"))} />
          <MetricTile label="Rappel" value={pourcentage(meilleur.rappel)} />
          <MetricTile label="Silence hors corpus" value={pourcentage(meilleur.silence_correct_hors_corpus)} />
        </div>
        <p className="mt-2 text-xs text-neutral-500">
          « Silence hors corpus » : part des questions sans réponse dans le corpus pour lesquelles
          le RAG ne renvoie rien — un succès, pas un échec (§9).
        </p>
      </div>

      {mesures.length > 0 && (
        <div>
          <p className="mb-1 text-sm font-medium">Balayage complet</p>
          <DataTable colonnes={Object.keys(mesures[0])} lignes={mesures} />
        </div>
      )}
      {versTexte(resultats.limite) && (
        <p className="text-xs text-neutral-500">{versTexte(resultats.limite)}</p>
      )}
    </div>
  );
}

/** Port de `back_office._section_systeme`. */
function SectionSysteme({ artefact }: { artefact: ArtefactMesure }) {
  if (!artefact.disponible) {
    return (
      <ArtefactAbsent nom="L'évaluation système (32 cas)" commande={artefact.commande ?? ""} />
    );
  }
  const resultats = versObjet(artefact.donnees);
  const reussis = versNombre(resultats.reussis);
  const total = versNombre(resultats.total);
  const latence = versObjet(resultats.latence_ms);
  const parCategorie = versObjet(resultats.taux_par_categorie);
  const details = versListe(resultats.resultats_detailles);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <MetricTile label="Cas réussis" value={`${reussis}/${total}`} />
        <MetricTile label="Taux" value={total ? pourcentage(reussis / total) : "—"} />
        <MetricTile label="Latence moyenne" value={`${versTexte(String(latence.moyenne ?? "?"))} ms`} />
      </div>

      <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/40 dark:text-amber-200">
        <strong>À remesurer avant la remise.</strong> Ces chiffres datent d&apos;avant la
        calibration du seuil RAG (RAG-5) : ils ne correspondent plus à la configuration livrée.
      </p>

      {Object.keys(parCategorie).length > 0 && (
        <div>
          <p className="mb-1 text-sm font-medium">Résultat par catégorie du §13</p>
          <DataTable
            colonnes={["catégorie", "résultat"]}
            lignes={Object.entries(parCategorie).map(([categorie, resultat]) => ({
              catégorie: categorie,
              résultat: resultat,
            }))}
          />
        </div>
      )}

      {details.length > 0 && (
        <details className="rounded-lg border border-neutral-200 px-3 py-2 text-sm dark:border-neutral-800">
          <summary className="cursor-pointer">Résultats détaillés</summary>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-neutral-100 p-3 text-xs dark:bg-neutral-900">
            {JSON.stringify(details, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

export function MesuresView({
  ml,
  rag,
  systeme,
}: {
  ml: ArtefactMesure;
  rag: ArtefactMesure;
  systeme: ArtefactMesure;
}) {
  const [onglet, setOnglet] = useState<(typeof ONGLETS)[number]>(ONGLETS[0]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-1 border-b border-neutral-200 dark:border-neutral-800">
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

      {onglet === "Machine Learning" && <SectionML artefact={ml} />}
      {onglet === "Recherche documentaire" && <SectionRAG artefact={rag} />}
      {onglet === "Système de bout en bout" && <SectionSysteme artefact={systeme} />}
    </div>
  );
}
