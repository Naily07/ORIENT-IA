import {
  AlertTriangle,
  BookOpen,
  Circle,
  GraduationCap,
  HelpCircle,
  Landmark,
  Target,
  type LucideIcon,
} from "lucide-react";

import { libelleAction, type TonaliteAction } from "@/lib/actions-labels";
import { formaterScore } from "@/lib/format-score";
import { admissibiliteAVerifier, scoreEstCreux, type Marqueurs } from "@/lib/markers";
import type { OrientationReponse, RecommandationParcours } from "@/lib/types";

const TONALITE_CLASSES: Record<TonaliteAction, string> = {
  info: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200",
  succes: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200",
  attention: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
  violet: "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-200",
};

// Ensemble fermé (correspond exactement aux `icone` de `lib/actions-labels.ts`) :
// une table statique plutôt qu'une résolution dynamique dans tout l'espace de
// noms `lucide-react`, pour que chaque composant d'icône reste déclaré une
// fois pour toutes au niveau module.
const ICONES_ACTION: Record<string, LucideIcon> = {
  BookOpen,
  Target,
  HelpCircle,
  GraduationCap,
  Landmark,
  Circle,
};

function CarteParcours({
  candidat,
  rang,
  marqueurs,
}: {
  candidat: RecommandationParcours;
  rang: number;
  marqueurs: Marqueurs;
}) {
  const justification = candidat.justification || "";
  const creux = scoreEstCreux(justification, marqueurs);
  const ecarte = admissibiliteAVerifier(justification, marqueurs);

  return (
    <div className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
      <p className="font-medium">{candidat.parcours}</p>
      {creux ? (
        <p className="text-xs text-neutral-500">score non significatif</p>
      ) : (
        <p className="text-2xl font-semibold tabular-nums">
          {formaterScore(candidat.score_adequation)}
        </p>
      )}
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {rang === 1 && !creux && !ecarte && (
          <span className="inline-flex items-center rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200">
            Meilleure adéquation
          </span>
        )}
        {ecarte && (
          <span className="inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-200">
            Admissibilité à vérifier
          </span>
        )}
      </div>
      <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-300">{justification || "—"}</p>
    </div>
  );
}

/** Port fidèle de `front_office._afficher_decision`/`_carte_parcours`. */
export function DecisionCard({
  reponse,
  marqueurs,
}: {
  reponse: OrientationReponse;
  marqueurs: Marqueurs;
}) {
  const { decision } = reponse;
  const { libelle, icone, tonalite } = libelleAction(decision.action);
  const Icone = ICONES_ACTION[icone] ?? Circle;

  const parcours = decision.parcours_recommandes ?? [];
  const manquantes = decision.informations_manquantes ?? [];
  const sources = decision.sources ?? [];
  const outils = Array.from(new Set(decision.outils_utilises ?? []));

  return (
    <div className="space-y-4">
      <span
        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${TONALITE_CLASSES[tonalite]}`}
      >
        <Icone className="size-3.5" aria-hidden="true" />
        {libelle}
      </span>

      {decision.resume && <p className="text-sm">{decision.resume}</p>}

      {decision.incertitude_declaree && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/40 dark:text-amber-200">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <p>
            <strong>L&apos;assistant n&apos;est pas certain de cette réponse.</strong> Les
            informations dont il dispose ne suffisent pas à conclure — prenez-la comme une piste à
            confirmer, pas comme un conseil arrêté.
          </p>
        </div>
      )}

      {parcours.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Parcours suggérés
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {parcours.slice(0, 5).map((candidat, index) => (
              <CarteParcours
                key={`${candidat.parcours}-${index}`}
                candidat={candidat}
                rang={index + 1}
                marqueurs={marqueurs}
              />
            ))}
          </div>
        </div>
      )}

      {manquantes.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Ce qui aiderait à mieux vous répondre
          </h3>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm">
            {manquantes.map((element) => (
              <li key={element}>{element}</li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
          Pourquoi cette réponse
        </h3>
        <p className="mt-1 text-sm text-neutral-700 dark:text-neutral-300">
          {decision.explication || "—"}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Documents cités
          </h3>
          {sources.length > 0 ? (
            <ul className="mt-1 space-y-0.5 text-sm">
              {sources.map((source) => (
                <li key={source}>
                  <code className="text-xs">{source}</code>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-xs text-neutral-500">
              Aucun document du corpus n&apos;a été cité pour cette réponse.
            </p>
          )}
        </div>
        <div>
          <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Ce que l&apos;assistant a consulté
          </h3>
          {outils.length > 0 ? (
            <ul className="mt-1 space-y-0.5 text-sm">
              {outils.map((outil) => (
                <li key={outil}>
                  <code className="text-xs">{outil}</code>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-xs text-neutral-500">Aucun outil appelé.</p>
          )}
        </div>
      </div>

      <p className="text-xs text-neutral-500">
        Confiance déclarée : {Math.round((decision.confiance ?? 0) * 100)} % · trace{" "}
        <code>{reponse.trace_id}</code>
      </p>

      <details className="text-xs">
        <summary className="cursor-pointer text-neutral-500">
          Réponse brute (JSON) — pour le jury
        </summary>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-neutral-100 p-3 dark:bg-neutral-900">
          {JSON.stringify(reponse, null, 2)}
        </pre>
      </details>
    </div>
  );
}
