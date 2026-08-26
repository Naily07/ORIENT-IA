"""Schéma d'entités et de relations de l'ontologie ORIENT'IA (§12 du sujet).

Ce module fixe le **vocabulaire contrôlé** des entités et des relations que
l'extension symbolique du sujet autorise. Il ne construit pas de graphe : la
construction (peuplement depuis le corpus structuré, `src.models.CorpusFormations`)
viendra dans un ticket dédié une fois le corpus collecté (voir `BACKLOG.md`,
ONTO-2), et s'appuiera sur `relation_valide()` pour ne pas laisser une relation
mal saisie s'ajouter silencieusement au graphe.

Établir ce schéma dès maintenant — avant même d'avoir des données réelles —
permet à `verifier_prerequis()` (ONTO-3), `detecter_incoherences()` (ONTO-4)
et la construction du graphe (ONTO-2) de partager la même définition plutôt
que de la redécouvrir chacun à leur façon.
"""

from typing import Literal, NamedTuple, get_args

TypeEntite = Literal[
    "Etudiant",
    "Formation",
    "Mention",
    "Parcours",
    "Matiere",
    "Competence",
    "Prerequis",
    "Metier",
    "CentreInteret",
]

TypeRelation = Literal[
    "enseigne",
    "developpe",
    "prepareA",
    "necessite",
    "possede",
    "prefere",
    "estRequisePour",
]

ENTITES: tuple[str, ...] = get_args(TypeEntite)
RELATIONS: tuple[str, ...] = get_args(TypeRelation)


class SchemaRelation(NamedTuple):
    """Un triplet (source, relation, cible) autorisé par le schéma."""

    source: TypeEntite
    relation: TypeRelation
    cible: TypeEntite


# Relations explicitement listées au §12 du sujet (Fig. 2). Un parcours peut
# avoir plusieurs prérequis/compétences/matières : ces relations sont
# many-to-many, portées par des listes d'identifiants côté
# `src.models.Parcours`, pas par ce module qui ne fait que définir ce qui est
# structurellement permis.
SCHEMA_RELATIONS: tuple[SchemaRelation, ...] = (
    SchemaRelation("Parcours", "enseigne", "Matiere"),
    SchemaRelation("Parcours", "developpe", "Competence"),
    SchemaRelation("Parcours", "prepareA", "Metier"),
    SchemaRelation("Parcours", "necessite", "Prerequis"),
    SchemaRelation("Etudiant", "possede", "Competence"),
    SchemaRelation("Etudiant", "prefere", "Matiere"),
    SchemaRelation("Competence", "estRequisePour", "Metier"),
)


def relation_valide(source_type: str, relation: str, cible_type: str) -> bool:
    """Vrai si le triplet (source, relation, cible) fait partie du schéma.

    Garde-fou déterministe pour la construction du graphe (ONTO-2) et la
    détection d'incohérences (ONTO-4) : une relation hors-schéma signale une
    erreur de saisie dans le corpus, pas une variante à accepter en silence.
    """
    return (source_type, relation, cible_type) in SCHEMA_RELATIONS


def relations_depuis(type_entite: str) -> tuple[SchemaRelation, ...]:
    """Les relations du schéma dont `type_entite` est la source.

    Utile pour explorer ce qu'il est possible de demander à partir d'une
    entité donnée (ex. depuis un Parcours : quelles relations peut-on suivre ?).
    """
    return tuple(r for r in SCHEMA_RELATIONS if r.source == type_entite)
