"""Modèle hybride : apprentissage statistique **et** règles d'admission (§6 du sujet).

Le sujet cite explicitement le « modèle hybride combinant apprentissage
statistique et règles » parmi les approches valorisées. Ce module en est la
mise en œuvre concrète, et corrige un défaut mesuré du modèle seul.

**Le défaut.** Le modèle ML ne voit jamais la série de baccalauréat : elle
n'est pas dans l'espace de features (`ml.features.noms_features()`). Il peut
donc classer en tête un parcours auquel le candidat n'est pas admissible.
Vérifié sur le corpus réel — un profil « Bac A, intéressé par l'informatique »
obtenait ses **quatre** premières recommandations parmi des parcours exigeant
un Bac C, D, S ou technique industrielle :

    IGGLIA 54 %  — Baccalauréat série C, D, S, ou série techniques industrielles
    ESIIA  11 %  — idem
    ISAIA   4 %  — idem
    EMII    4 %  — idem

Recommander une formation dans laquelle quelqu'un ne peut pas s'inscrire est
l'erreur la plus coûteuse pour un assistant d'orientation, et va frontalement
contre l'exigence de recommandation « prudente » (§2).

**La règle appliquée.** Les prérequis viennent du graphe de connaissances
(`graphe.prerequis_du_parcours`, relation `necessite` — ONTO-3), et la
compatibilité est tranchée par `src.admission.serie_satisfait_prerequis()`,
partagée avec l'outil `verifier_prerequis` pour que les deux ne divergent
jamais.

**Trois choix de conception délibérés :**

1. **Rétrograder, pas masquer.** Un parcours inadmissible reste visible, avec
   son score et une justification qui dit pourquoi il est écarté. Cacher
   l'information priverait le candidat d'un élément de décision (une
   équivalence, une passerelle, une réorientation restent possibles) — et le
   sujet distingue justement conseil pédagogique et décision administrative
   (§16) : ce module conseille, il ne prononce pas une inadmissibilité
   définitive.
2. **Ne rien faire quand on ne sait pas.** Série non déclarée ou prérequis
   inconnus (`serie_satisfait_prerequis` retourne `None`) : le classement est
   laissé strictement intact. L'incertitude ne doit pas se transformer en
   pénalité silencieuse.
3. **Les scores ne sont jamais modifiés.** Seul l'ordre change. Un score
   affiché reste celui que le modèle a réellement produit, condition pour que
   la distinction « résultat du modèle / règle pédagogique » du §6 reste
   lisible.
"""

from dataclasses import dataclass

import networkx as nx

from src.admission import serie_satisfait_prerequis
from src.graphe import prerequis_du_parcours
from src.schemas import ProfilCandidat, RecommandationParcours


@dataclass
class VerdictAdmission:
    """Verdict d'admissibilité pour un parcours donné."""

    parcours: str
    admissible: bool | None
    prerequis: list[str]

    @property
    def inadmissible(self) -> bool:
        """Vrai uniquement en cas de verdict négatif *établi* — `None`
        (indéterminable) n'est jamais traité comme un refus."""
        return self.admissible is False


def evaluer_admissibilite(
    graphe: nx.DiGraph, parcours_id: str, serie_bac: str | None
) -> VerdictAdmission:
    """Confronte la série déclarée aux prérequis du parcours dans le graphe."""
    prerequis = prerequis_du_parcours(graphe, parcours_id)
    return VerdictAdmission(
        parcours=parcours_id,
        admissible=serie_satisfait_prerequis(serie_bac, prerequis),
        prerequis=prerequis,
    )


# Marqueur repris tel quel par l'interface pour signaler un parcours
# retrograde : une constante partagee plutot qu'une chaine devinee des deux cotes.
MARQUEUR_REGLE_ADMISSION = "[Règle d'admission]"


def _justification_annotee(
    candidat: RecommandationParcours, verdict: VerdictAdmission
) -> str:
    prerequis = verdict.prerequis[0] if verdict.prerequis else "prérequis non précisés"
    return (
        f"{candidat.justification} {MARQUEUR_REGLE_ADMISSION} Ce parcours est classé après les "
        f"parcours accessibles : la série de baccalauréat déclarée ne correspond pas "
        f"aux prérequis connus ({prerequis}). Une équivalence ou une passerelle relève "
        "de l'administration de l'ISPM."
    )


def appliquer_regles_admission(
    candidats: list[RecommandationParcours],
    profil: ProfilCandidat,
    graphe: nx.DiGraph | None,
) -> list[RecommandationParcours]:
    """Rétrograde les parcours inadmissibles sous les parcours accessibles.

    L'ordre relatif issu du modèle est préservé à l'intérieur de chaque groupe :
    seule la frontière admissible/inadmissible est introduite, jamais un
    reclassement arbitraire.

    Retourne la liste inchangée si le graphe est absent ou si la série n'est pas
    déclarée — l'enrichissement symbolique est un bonus, jamais une condition
    pour répondre (même principe que `tools.expliquer_recommandation`).
    """
    if graphe is None or not (profil.serie_bac or "").strip():
        return candidats

    accessibles: list[RecommandationParcours] = []
    ecartes: list[RecommandationParcours] = []

    for candidat in candidats:
        verdict = evaluer_admissibilite(graphe, candidat.parcours, profil.serie_bac)
        if verdict.inadmissible:
            ecartes.append(
                candidat.model_copy(
                    update={"justification": _justification_annotee(candidat, verdict)}
                )
            )
        else:
            accessibles.append(candidat)

    return accessibles + ecartes
