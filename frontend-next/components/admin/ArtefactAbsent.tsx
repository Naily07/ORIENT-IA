import { Info } from "lucide-react";

/** Port de `noyau.artefact_absent()` : dit quelle commande produit
 * l'artefact manquant plutôt que d'afficher un vide muet. */
export function ArtefactAbsent({ nom, commande }: { nom: string; commande: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900 dark:border-blue-900/40 dark:bg-blue-950/40 dark:text-blue-200">
      <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <div>
        <p>
          <strong>{nom}</strong> n&apos;a pas encore été généré.
        </p>
        <p className="mt-1">Le produire avec :</p>
        <pre className="mt-1 overflow-x-auto rounded-lg bg-white/60 px-2 py-1 text-xs dark:bg-black/30">
          <code>{commande}</code>
        </pre>
      </div>
    </div>
  );
}
