/**
 * Accesseurs sûrs pour lire les artefacts JSON de `/admin/mesures` — leur
 * forme est celle produite par `backend/tests/eval_*.py`, pas un schéma
 * Pydantic garanti, d'où ces conversions défensives plutôt que des accès
 * directs qui planteraient sur un champ absent ou renommé.
 */
export function versObjet(valeur: unknown): Record<string, unknown> {
  return valeur && typeof valeur === "object" && !Array.isArray(valeur)
    ? (valeur as Record<string, unknown>)
    : {};
}

export function versNombre(valeur: unknown, defaut = 0): number {
  return typeof valeur === "number" ? valeur : defaut;
}

export function versTexte(valeur: unknown, defaut = ""): string {
  return typeof valeur === "string" ? valeur : defaut;
}

export function versListe(valeur: unknown): unknown[] {
  return Array.isArray(valeur) ? valeur : [];
}

export function pourcentage(valeur: unknown, decimales = 0): string {
  const n = versNombre(valeur, NaN);
  return Number.isNaN(n) ? "—" : `${(n * 100).toFixed(decimales)} %`;
}

export function decimal(valeur: unknown, decimales = 3): string {
  const n = versNombre(valeur, NaN);
  return Number.isNaN(n) ? "—" : n.toFixed(decimales);
}
