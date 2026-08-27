function formaterCellule(valeur: unknown): string {
  if (valeur === null || valeur === undefined) return "—";
  if (Array.isArray(valeur)) return valeur.join(", ");
  if (typeof valeur === "object") return JSON.stringify(valeur);
  return String(valeur);
}

export function DataTable({
  colonnes,
  lignes,
}: {
  colonnes: string[];
  lignes: Record<string, unknown>[];
}) {
  if (lignes.length === 0) {
    return <p className="text-sm text-neutral-500">Aucune donnée.</p>;
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-neutral-200 dark:border-neutral-800">
      <table className="w-full text-left text-sm">
        <thead className="bg-neutral-50 dark:bg-neutral-900">
          <tr>
            {colonnes.map((colonne) => (
              <th key={colonne} className="px-3 py-2 font-medium text-neutral-500">
                {colonne}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {lignes.map((ligne, index) => (
            <tr key={index} className="border-t border-neutral-100 dark:border-neutral-800">
              {colonnes.map((colonne) => (
                <td key={colonne} className="px-3 py-2 align-top">
                  {formaterCellule(ligne[colonne])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
