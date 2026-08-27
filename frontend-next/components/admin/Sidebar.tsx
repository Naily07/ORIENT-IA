"use client";

import {
  Activity,
  Info,
  LayoutDashboard,
  LineChart,
  LogOut,
  Network,
  ShieldCheck,
  Wrench,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { deconnexionAdmin } from "@/app/admin/login/actions";

const LIENS = [
  { href: "/admin/tableau-de-bord", label: "Tableau de bord", icone: LayoutDashboard },
  { href: "/admin/observabilite", label: "Observabilité", icone: Activity },
  { href: "/admin/qualite-donnees", label: "Qualité des données", icone: ShieldCheck },
  { href: "/admin/corpus", label: "Corpus et graphe", icone: Network },
  { href: "/admin/mesures", label: "Mesures", icone: LineChart },
] as const;

export function Sidebar({ accesOuvert }: { accesOuvert: boolean }) {
  const chemin = usePathname();

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-neutral-200 p-4 dark:border-neutral-800">
      <div className="mb-6 flex items-center gap-2">
        <Wrench className="size-4" aria-hidden="true" />
        <div>
          <p className="text-sm font-semibold">ORIENT&apos;IA</p>
          <p className="text-xs text-neutral-500">Administration</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1">
        {LIENS.map(({ href, label, icone: Icone }) => {
          const actif = chemin === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors duration-150 ${
                actif
                  ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                  : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-900"
              }`}
            >
              <Icone className="size-4" aria-hidden="true" />
              {label}
            </Link>
          );
        })}
      </nav>

      {accesOuvert ? (
        <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/40 dark:text-amber-200">
          <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
          <span>Espace admin ouvert (aucun ORIENTIA_ADMIN_CODE défini).</span>
        </div>
      ) : (
        <form action={deconnexionAdmin}>
          <button
            type="submit"
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-neutral-600 transition-[transform,background-color] duration-150 ease-out-strong hover:bg-neutral-100 active:scale-[0.98] dark:text-neutral-300 dark:hover:bg-neutral-900"
          >
            <LogOut className="size-4" aria-hidden="true" />
            Se déconnecter
          </button>
        </form>
      )}
    </aside>
  );
}
