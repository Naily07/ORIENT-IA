"use client";

import { Lock } from "lucide-react";
import { useActionState } from "react";

import { connexionAdmin, type EtatConnexion } from "@/app/admin/login/actions";

const ETAT_INITIAL: EtatConnexion = {};

export default function PageConnexionAdmin() {
  const [etat, action, enCours] = useActionState(connexionAdmin, ETAT_INITIAL);

  return (
    <div className="flex min-h-dvh items-center justify-center bg-neutral-50 px-4 dark:bg-neutral-950">
      <form
        action={action}
        className="w-full max-w-sm space-y-4 rounded-2xl border border-neutral-200 bg-white p-6 dark:border-neutral-800 dark:bg-neutral-900"
      >
        <div className="flex items-center gap-2">
          <Lock className="size-5" aria-hidden="true" />
          <h1 className="text-lg font-semibold">Espace d&apos;administration</h1>
        </div>
        <p className="text-sm text-neutral-500">
          Cet espace est réservé à l&apos;équipe. Saisissez le code d&apos;accès.
        </p>

        <label className="block">
          <span className="text-xs font-medium text-neutral-500">Code d&apos;accès</span>
          <input
            name="code"
            type="password"
            autoFocus
            className="mt-1 w-full rounded-lg border border-neutral-300 px-3 py-2 text-sm focus:ring-2 focus:ring-neutral-900 focus:outline-none dark:border-neutral-700 dark:bg-neutral-950 dark:focus:ring-white"
          />
        </label>

        {etat.erreur && <p className="text-sm text-red-600 dark:text-red-400">{etat.erreur}</p>}

        <button
          type="submit"
          disabled={enCours}
          className="w-full rounded-lg bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition-transform duration-150 ease-out-strong active:scale-[0.98] disabled:opacity-50 disabled:active:scale-100 dark:bg-white dark:text-neutral-900"
        >
          Entrer
        </button>
      </form>
    </div>
  );
}
