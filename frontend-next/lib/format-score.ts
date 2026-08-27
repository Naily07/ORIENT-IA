/**
 * Miroir de `front_office.py::_formater_score()`.
 *
 * Le modèle est borné pour ne jamais produire exactement 1 (voir
 * `ml.entrainement.ModeleBorne`), mais un arrondi naïf à l'entier ramènerait
 * 0,9953 à « 100 % » — rétablissant à l'affichage la certitude absolue que le
 * modèle s'interdit. Un assistant d'orientation ne peut pas annoncer à
 * quelqu'un que son avenir est certain à 100 %.
 */
export function formaterScore(score: number): string {
  if (score >= 0.995) return "> 99 %";
  if (score <= 0.005) return "< 1 %";
  return `${Math.round(score * 100)} %`;
}
