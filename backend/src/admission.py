"""Règle d'admissibilité : une série de baccalauréat satisfait-elle un prérequis ?

Règle **déterministe**, extraite ici pour être partagée sans duplication entre :

- `tools.verifier_prerequis()` — l'outil que l'agent appelle explicitement ;
- `ml.hybride` — le filtrage d'admissibilité appliqué à toute recommandation ML.

Un module dédié plutôt qu'une fonction dans l'un des deux : `tools.py` importe
déjà `src.ml.outils`, donc `src/ml/` ne peut pas importer `tools.py` en retour
sans créer un cycle. La règle vit donc en amont des deux.
"""

import re

# Formulation générique des prérequis ouverts à toutes les séries, telle qu'elle
# apparaît dans le corpus collecté (voir `backend/data/prerequis.json`).
SERIES_TOUTE = "toute série"

# Tête de phrase que les candidats ajoutent devant leur série (« bac D »,
# « Baccalauréat série C ») et que le formulaire n'impose pas de retirer. Sans
# ce nettoyage, « bac D » ne correspond à aucun prérequis rédigé « série C, D, S »
# et le parcours est rétrogradé à tort — mesuré, un « bac D » scientifique voyait
# toutes ses formations techniques passer derrière le tourisme.
_PREFIXE_SERIE = re.compile(
    r"^\s*(?:bac(?:calaur[ée]at)?|s[ée]ries?|en|fili[èe]re)\b[\s:.-]*", re.IGNORECASE
)


def serie_bac_nettoyee(serie: str) -> str:
    """Retire les têtes de phrase (« bac », « série »…) répétées devant la série.

    Partagée avec `extraction_profil.fusionner_profils` : la série est stockée et
    affichée sous sa forme courte (« D »), et confrontée aux prérequis sous cette
    même forme."""
    precedent = None
    courant = serie.strip()
    while courant and courant != precedent:
        precedent = courant
        courant = _PREFIXE_SERIE.sub("", courant).strip()
    return courant or serie.strip()


def serie_satisfait_prerequis(serie_declaree: str | None, descriptions: list[str]) -> bool | None:
    """La série déclarée satisfait-elle au moins un des prérequis listés ?

    Retourne `None` quand la question ne peut pas être tranchée — série non
    déclarée, ou aucun prérequis connu pour ce parcours. Un `None` n'est pas un
    « non » : c'est une information manquante, que l'appelant doit signaler
    plutôt que de trancher à la place du candidat.

    `.strip()` avant le test de vérité, pas seulement dans la regex : une saisie
    composée uniquement d'espaces passerait le test, puis produirait un motif
    vide dont `\\b\\b` correspond à n'importe quel texte — et la fonction
    confirmerait à tort l'admissibilité.
    """
    if not descriptions:
        return None

    serie = (serie_declaree or "").strip()
    if not serie:
        return None
    # « bac D », « Série D » → « D » : le prérequis est rédigé avec la seule
    # lettre/mention de série, jamais avec la tête de phrase.
    serie = serie_bac_nettoyee(serie)
    if not serie:
        return None

    # Frontière de mot obligatoire : une correspondance par simple sous-chaîne
    # ferait matcher une lettre isolée (« L ») contre une lettre au milieu d'un
    # mot de la phrase (« baccaLauréat ») — trouvé en testant « série L »
    # contre « Baccalauréat série C, D, S ».
    motif_serie = re.compile(rf"\b{re.escape(serie)}\b", re.IGNORECASE)
    return any(
        SERIES_TOUTE in description.lower() or motif_serie.search(description)
        for description in descriptions
    )
