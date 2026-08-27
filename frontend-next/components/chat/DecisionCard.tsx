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
import { profilRenseigne } from "@/lib/profil";
import type { OrientationReponse, RecommandationParcours } from "@/lib/types";

const TONALITE_CLASSES: Record<TonaliteAction, string> = {
  info: "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200",
  succes: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200",
  attention: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
  violet: "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-200",
};

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

/**
 * Affiche la réponse de l'assistant façon conversation : le texte rédigé
 * (`decision.reponse`) en premier, les scores de parcours quand il y en a, et
 * toute la traçabilité (résumé technique, explication, sources, outils,
 * confiance, JSON brut) repliée dans un `<details>` pour le jury.
 */
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

  // Le modèle a été consulté mais n'a produit aucun classement à montrer : le
  // profil déclaré ne portait pas assez de traits reconnus pour qu'un score
  // d'adéquation dise quoi que ce soit de ce candidat (backend :
  // `_masquer_classement_non_informatif`). Plutôt qu'un podium vide ou, pire,
  // un « TEE 7 % » trompeur, on invite à compléter le profil.
  const modeleConsulteSansClassement =
    parcours.length === 0 && outils.includes("analyser_profil_ml");
  // `reponse.profil` peut manquer sur une réponse mise en cache avant l'ajout du
  // champ : on ne montre alors pas le variant « profil vide ».
  const profilVide = reponse.profil ? !profilRenseigne(reponse.profil) : false;

  // `decision.reponse` est rempli par l'agent dans la quasi-totalité des cas ;
  // repli sur le résumé si jamais il manque.
  const texte = (decision.reponse || decision.resume || "").trim();
  const paragraphes = texte.split(/\n{2,}/).filter(Boolean);

  return (
    <div className="space-y-4">
      <div className="space-y-3 text-sm leading-relaxed text-neutral-800 dark:text-neutral-100">
        {paragraphes.length > 0 ? (
          paragraphes.map((p, i) => (
            <p key={i} className="whitespace-pre-wrap">
              {p}
            </p>
          ))
        ) : (
          <p>—</p>
        )}
      </div>

      {/* Bandeau de prudence réservé aux réponses qui *conseillent* : sur une
          question factuelle ou une demande de précisions, la nuance est déjà
          portée par le texte, un bandeau en plus n'est que du bruit. */}
      {decision.incertitude_declaree &&
        (decision.action === "recommandation" || decision.action === "escalade_conseiller") && (
          <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/40 dark:text-amber-200">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            <p>
              À confirmer avec un conseiller : les informations disponibles ne suffisent pas à en
              faire un conseil arrêté.
            </p>
          </div>
        )}

      {modeleConsulteSansClassement && (
        <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900 dark:border-blue-900/40 dark:bg-blue-950/40 dark:text-blue-200">
          <HelpCircle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <p>
            {profilVide
              ? "Le modèle n'a pas pu calculer de score : renseignez « Mon profil » (matières, compétences, série du bac) pour une recommandation chiffrée."
              : "Le modèle n'a pas assez d'éléments reconnus pour un score fiable : complétez « Mon profil » pour affiner la recommandation."}
          </p>
        </div>
      )}

      {parcours.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
            Scores du modèle
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {parcours.slice(0, 4).map((candidat, index) => (
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

      {/* `informations_manquantes` n'est pas affiché comme une liste : la
          question à l'utilisateur vit dans `decision.reponse`, en langage
          naturel. La liste reste dans la traçabilité ci-dessous, pour le jury. */}

      <details className="group rounded-lg border border-neutral-200 text-sm dark:border-neutral-800">
        <summary className="flex cursor-pointer items-center gap-1.5 px-3 py-2 text-xs font-medium text-neutral-500 select-none">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 ${TONALITE_CLASSES[tonalite]}`}
          >
            <Icone className="size-3.5" aria-hidden="true" />
            {libelle}
          </span>
          <span className="ml-auto group-open:hidden">Voir la traçabilité</span>
          <span className="ml-auto hidden group-open:inline">Masquer la traçabilité</span>
        </summary>

        <div className="space-y-4 border-t border-neutral-200 px-3 py-3 dark:border-neutral-800">
          {decision.resume && (
            <div>
              <h4 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
                Demande comprise
              </h4>
              <p className="mt-1 text-neutral-700 dark:text-neutral-300">{decision.resume}</p>
            </div>
          )}

          <div>
            <h4 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
              Ce qui fonde la réponse
            </h4>
            <p className="mt-1 whitespace-pre-wrap text-neutral-700 dark:text-neutral-300">
              {decision.explication || "—"}
            </p>
          </div>

          {manquantes.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
                Informations manquantes (suivi interne)
              </h4>
              <ul className="mt-1 list-disc space-y-0.5 pl-5 text-neutral-700 dark:text-neutral-300">
                {manquantes.map((element) => (
                  <li key={element}>{element}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <h4 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
                Documents cités
              </h4>
              {sources.length > 0 ? (
                <ul className="mt-1 space-y-0.5">
                  {sources.map((source) => (
                    <li key={source}>
                      <code className="text-xs">{source}</code>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-1 text-xs text-neutral-500">Aucun document du corpus cité.</p>
              )}
            </div>
            <div>
              <h4 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase">
                Outils consultés
              </h4>
              {outils.length > 0 ? (
                <ul className="mt-1 space-y-0.5">
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
            <summary className="cursor-pointer text-neutral-500">Réponse brute (JSON)</summary>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-neutral-100 p-3 dark:bg-neutral-900">
              {JSON.stringify(reponse, null, 2)}
            </pre>
          </details>
        </div>
      </details>
    </div>
  );
}
