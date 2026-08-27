import type { ProfilCandidat } from "@/lib/types";

/**
 * Fusionne le profil renvoyé par le backend (déjà complété depuis le message,
 * voir `backend/src/extraction_profil.py`) avec le profil courant du client.
 *
 * Le backend a fusionné sur le profil tel qu'il était à l'envoi de la requête ;
 * `courant` peut avoir changé entre-temps si l'utilisateur a édité le panneau
 * « Mon profil » pendant que la réponse arrivait. On réunit donc les deux, en
 * donnant la priorité au client pour les valeurs qu'il a saisies explicitement.
 */
export function fusionnerProfils(
  courant: ProfilCandidat,
  renvoye: ProfilCandidat,
): ProfilCandidat {
  return {
    matieres_preferees: unionListe(courant.matieres_preferees, renvoye.matieres_preferees),
    competences_declarees: unionListe(
      courant.competences_declarees,
      renvoye.competences_declarees,
    ),
    centres_interet: unionListe(courant.centres_interet, renvoye.centres_interet),
    activites_projets: unionListe(courant.activites_projets, renvoye.activites_projets),
    preferences_professionnelles: unionListe(
      courant.preferences_professionnelles,
      renvoye.preferences_professionnelles,
    ),
    environnement_travail_recherche:
      courant.environnement_travail_recherche ?? renvoye.environnement_travail_recherche,
    serie_bac: courant.serie_bac ?? renvoye.serie_bac,
    resultats_scolaires: { ...renvoye.resultats_scolaires, ...courant.resultats_scolaires },
    informations_manquantes: renvoye.informations_manquantes ?? courant.informations_manquantes,
  };
}

/** Union ordonnée, dédoublonnée à la casse et aux espaces près — les entrées du
 *  client d'abord, leur graphie conservée. */
function unionListe(prioritaire: string[], complement: string[]): string[] {
  const resultat = [...prioritaire];
  const connus = new Set(prioritaire.map((v) => v.trim().toLowerCase()));
  for (const terme of complement) {
    const cle = terme.trim().toLowerCase();
    if (cle && !connus.has(cle)) {
      resultat.push(terme);
      connus.add(cle);
    }
  }
  return resultat;
}

/** Vrai si au moins un champ du profil porte une information. */
export function profilRenseigne(profil: ProfilCandidat): boolean {
  return (
    profil.matieres_preferees.length > 0 ||
    profil.competences_declarees.length > 0 ||
    profil.centres_interet.length > 0 ||
    profil.activites_projets.length > 0 ||
    profil.preferences_professionnelles.length > 0 ||
    profil.environnement_travail_recherche !== null ||
    profil.serie_bac !== null ||
    Object.keys(profil.resultats_scolaires).length > 0
  );
}
