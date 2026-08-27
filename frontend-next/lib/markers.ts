/**
 * Miroir de `front_office.py::_marqueurs()` — détecte, dans la justification
 * d'un parcours recommandé, les marqueurs internes que le backend insère pour
 * signaler un score creux ou une admissibilité à vérifier.
 *
 * Source unique réelle : `GET /health` (`marqueur_regle_admission`,
 * `avertissement_non_exploitable`). Les constantes ci-dessous ne sont qu'un
 * repli si l'API est injoignable — même mécanisme que la mention obligatoire
 * (voir `components/chat/MentionBanner.tsx`), pas une source de vérité.
 */

export const AVERTISSEMENT_NON_EXPLOITABLE_REPLI =
  "Score non informatif : le profil déclaré n'a pas pu être rattaché au " +
  "vocabulaire du modèle. Cette valeur reflète la distribution générale des " +
  "parcours, pas ce candidat, et ne doit pas fonder une recommandation.";

export const MARQUEUR_REGLE_ADMISSION_REPLI = "[Règle d'admission]";

export interface Marqueurs {
  avertissementNonExploitable: string;
  marqueurRegleAdmission: string;
}

export const MARQUEURS_REPLI: Marqueurs = {
  avertissementNonExploitable: AVERTISSEMENT_NON_EXPLOITABLE_REPLI,
  marqueurRegleAdmission: MARQUEUR_REGLE_ADMISSION_REPLI,
};

/** Score numériquement présent mais ne portant aucune information sur ce
 * candidat — ne doit jamais être affiché comme un score fiable. */
export function scoreEstCreux(justification: string, marqueurs: Marqueurs): boolean {
  return justification.includes(marqueurs.avertissementNonExploitable);
}

/** Parcours rétrogradé par les règles d'admission : accessibilité à
 * confirmer avant de le présenter comme un choix ouvert. */
export function admissibiliteAVerifier(justification: string, marqueurs: Marqueurs): boolean {
  return justification.includes(marqueurs.marqueurRegleAdmission);
}
