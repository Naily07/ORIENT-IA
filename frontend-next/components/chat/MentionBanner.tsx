import { Info } from "lucide-react";

/**
 * Mention exigée au §16 du sujet, affichée **inconditionnellement** — jamais
 * conditionnée à un `/health` réussi. Défaut déjà corrigé côté Streamlit
 * (`noyau.afficher_mention_obligatoire`) : l'API éteinte est précisément
 * l'état où cette mention doit rester visible, pas disparaître.
 */
export function MentionBanner({ mention }: { mention: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900 dark:border-blue-900/40 dark:bg-blue-950/40 dark:text-blue-200">
      <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <p className="font-medium">{mention}</p>
    </div>
  );
}
