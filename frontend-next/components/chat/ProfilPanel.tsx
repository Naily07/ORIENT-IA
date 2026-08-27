"use client";

import { X } from "lucide-react";
import { useState, type KeyboardEvent } from "react";

import { useConversation } from "@/components/chat/ConversationProvider";
import type { ProfilCandidat } from "@/lib/types";

function ChipListField({
  label,
  placeholder,
  values,
  onChange,
}: {
  label: string;
  placeholder: string;
  values: string[];
  onChange: (valeurs: string[]) => void;
}) {
  const [brouillon, setBrouillon] = useState("");

  function ajouter() {
    const valeur = brouillon.trim();
    setBrouillon("");
    if (!valeur || values.includes(valeur)) return;
    onChange([...values, valeur]);
  }

  function gererTouche(evenement: KeyboardEvent<HTMLInputElement>) {
    if (evenement.key === "Enter" || evenement.key === ",") {
      evenement.preventDefault();
      ajouter();
    }
  }

  return (
    <div>
      <span className="text-xs font-medium text-neutral-500">{label}</span>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 rounded-lg border border-neutral-300 px-2 py-1.5 dark:border-neutral-700">
        {values.map((valeur) => (
          <span
            key={valeur}
            className="inline-flex items-center gap-1 rounded-full bg-neutral-100 px-2 py-1 text-xs dark:bg-neutral-800"
          >
            {valeur}
            <button
              type="button"
              onClick={() => onChange(values.filter((v) => v !== valeur))}
              aria-label={`Retirer ${valeur}`}
              className="rounded-full transition-transform duration-150 ease-out-strong active:scale-90"
            >
              <X className="size-3" />
            </button>
          </span>
        ))}
        <input
          value={brouillon}
          onChange={(evenement) => setBrouillon(evenement.target.value)}
          onKeyDown={gererTouche}
          onBlur={ajouter}
          placeholder={values.length === 0 ? placeholder : ""}
          className="min-w-[6rem] flex-1 bg-transparent text-xs outline-none"
        />
      </div>
    </div>
  );
}

/** Panneau « Mon profil » : équivalent conversationnel du formulaire à 6
 * champs de `front_office._formulaire_profil`.
 *
 * Deux sources le remplissent, toutes deux déclaratives :
 * - l'édition directe de l'utilisateur ici ;
 * - l'extraction de ce qu'il a **explicitement déclaré** dans le chat
 *   (« j'aime les maths », « je suis en bac D »), fusionnée côté backend
 *   (`extraction_profil.py`) et renvoyée dans `reponse.profil`.
 *
 * Ni l'une ni l'autre n'infère un trait à partir du ton ou du style d'écriture,
 * et aucun attribut sensible n'y entre (SEC-4, §16). L'utilisateur garde la
 * main : tout ce qui apparaît ici est modifiable ou supprimable. */
export function ProfilPanel() {
  const { profil, mettreAJourProfil } = useConversation();

  function definir<K extends keyof ProfilCandidat>(champ: K, valeur: ProfilCandidat[K]) {
    mettreAJourProfil({ ...profil, [champ]: valeur });
  }

  return (
    <div className="space-y-4 rounded-xl border border-neutral-200 p-4 text-sm dark:border-neutral-800">
      <div>
        <h2 className="font-medium">Mon profil</h2>
        <p className="mt-0.5 text-xs text-neutral-500">
          Tout est facultatif et se complète aussi depuis la conversation. Écrivez comme
          vous le diriez ; l&apos;assistant vous dira ce qu&apos;il n&apos;a pas su rattacher.
        </p>
      </div>

      <ChipListField
        label="Matières préférées"
        placeholder="maths, info…"
        values={profil.matieres_preferees}
        onChange={(v) => definir("matieres_preferees", v)}
      />
      <ChipListField
        label="Compétences"
        placeholder="Python, dessin technique…"
        values={profil.competences_declarees}
        onChange={(v) => definir("competences_declarees", v)}
      />
      <ChipListField
        label="Centres d'intérêt"
        placeholder="IA, robotique, nature…"
        values={profil.centres_interet}
        onChange={(v) => definir("centres_interet", v)}
      />
      <ChipListField
        label="Métiers ou domaines visés"
        placeholder="développement logiciel…"
        values={profil.preferences_professionnelles}
        onChange={(v) => definir("preferences_professionnelles", v)}
      />

      <label className="block">
        <span className="text-xs font-medium text-neutral-500">
          Environnement de travail souhaité
        </span>
        <input
          value={profil.environnement_travail_recherche ?? ""}
          onChange={(evenement) =>
            definir("environnement_travail_recherche", evenement.target.value || null)
          }
          placeholder="bureau, laboratoire, terrain"
          className="mt-1 w-full rounded-lg border border-neutral-300 bg-transparent px-2 py-1.5 text-xs dark:border-neutral-700"
        />
      </label>

      <label className="block">
        <span className="text-xs font-medium text-neutral-500">Série du baccalauréat</span>
        <input
          value={profil.serie_bac ?? ""}
          onChange={(evenement) => definir("serie_bac", evenement.target.value || null)}
          placeholder="C, D, S, A2…"
          className="mt-1 w-full rounded-lg border border-neutral-300 bg-transparent px-2 py-1.5 text-xs dark:border-neutral-700"
        />
        <span className="mt-1 block text-[11px] text-neutral-400">
          Sert à vérifier les conditions d&apos;admission. Sans elle, l&apos;assistant ne peut pas
          dire si un parcours vous est accessible.
        </span>
      </label>
    </div>
  );
}
