"""Graphe de connaissances : construction et requêtes (§12 du sujet, ONTO-2 à ONTO-5).

Peuple un graphe orienté (NetworkX) à partir du corpus structuré
(`src.models.CorpusFormations`), en respectant strictement le vocabulaire
déclaratif d'`ontologie.py` (ONTO-1) : chaque arête ajoutée est un triplet
(source, relation, cible) que `relation_valide()` autorise, jamais une
relation devinée.

Le graphe se construit avec ce qui est disponible dans le corpus, pas avec ce
qui serait plausible : un identifiant référencé par un `Parcours`
(`matieres`, `competences`, `prerequis`, `debouches`) mais absent des listes
correspondantes du corpus n'aboutit pas à une arête fantôme — c'est une
référence orpheline, du ressort de `detecter_incoherences()` (ONTO-4), pas de
la construction du graphe elle-même.

Sur le corpus réel actuel (16 parcours ISPM, DATA-1), seules les mentions,
parcours et prérequis sont collectés : matières, compétences et débouchés
métiers restent à documenter, et ne doivent pas être devinés (voir
`DOCS/bloc-donnees-guide.md`, §3.1 : « ne pas les inventer »). Le graphe issu
du corpus réel reste donc partiel pour l'instant — les fonctions ci-dessous
sont pleinement fonctionnelles dès que ces fichiers existeront, voir
`backend/tests/test_graphe.py` pour une démonstration sur un corpus jouet
entièrement peuplé, et `backend/tests/eval_ontologie.py` (ONTO-6) pour la
preuve mesurée de ce que le graphe apporte une fois ces données disponibles.

NetworkX plutôt que RDFLib (voir BACKLOG.md, ONTO-2) : le domaine est un
graphe de propriétés simple, quelques types de nœuds et sept relations
fixes, sans besoin d'inférence OWL/SPARQL — RDFLib ajouterait de la
complexité sans bénéfice pour la durée du hackathon.
"""

from __future__ import annotations

import networkx as nx

from src.models import CorpusFormations
from src.ontologie import TypeEntite, TypeRelation, relation_valide


def id_noeud(type_entite: TypeEntite, identifiant: str) -> str:
    """Identifiant de nœud unique, préfixé par type pour éviter toute collision
    entre deux entités de types différents qui partageraient un id."""
    return f"{type_entite}:{identifiant}"


def type_et_id(noeud: str) -> tuple[str, str]:
    """Inverse de `id_noeud` : redécompose un identifiant de nœud."""
    type_entite, _, identifiant = noeud.partition(":")
    return type_entite, identifiant


def _ajouter_entites(
    graphe: nx.DiGraph, entites, type_entite: TypeEntite, champ_nom: str
) -> None:
    for entite in entites:
        graphe.add_node(
            id_noeud(type_entite, entite.id), type=type_entite, nom=getattr(entite, champ_nom)
        )


def _relier(
    graphe: nx.DiGraph,
    type_source: TypeEntite,
    id_source: str,
    relation: TypeRelation,
    type_cible: TypeEntite,
    ids_cibles: list[str],
    ids_connues: set[str],
) -> None:
    if not relation_valide(type_source, relation, type_cible):
        # Erreur de programmation (appel interne à ce module hors schéma), pas
        # une donnée du corpus à signaler : ontologie.py fait foi, ce module
        # ne doit jamais s'en écarter silencieusement.
        raise ValueError(f"Relation hors schéma : {type_source} {relation} {type_cible}")
    noeud_source = id_noeud(type_source, id_source)
    for id_cible in ids_cibles:
        if id_cible not in ids_connues:
            continue  # référence orpheline : voir detecter_incoherences (ONTO-4)
        graphe.add_edge(noeud_source, id_noeud(type_cible, id_cible), relation=relation)


def construire_graphe(corpus: CorpusFormations) -> nx.DiGraph:
    """Peuple le graphe de connaissances depuis le corpus structuré (ONTO-2)."""
    graphe = nx.DiGraph()

    _ajouter_entites(graphe, corpus.mentions, "Mention", "nom")
    _ajouter_entites(graphe, corpus.parcours, "Parcours", "nom")
    _ajouter_entites(graphe, corpus.matieres, "Matiere", "nom")
    _ajouter_entites(graphe, corpus.competences, "Competence", "nom")
    _ajouter_entites(graphe, corpus.prerequis, "Prerequis", "description")
    _ajouter_entites(graphe, corpus.metiers, "Metier", "nom")

    ids_mentions = {m.id for m in corpus.mentions}
    ids_parcours = {p.id for p in corpus.parcours}
    ids_matieres = {m.id for m in corpus.matieres}
    ids_competences = {c.id for c in corpus.competences}
    ids_prerequis = {p.id for p in corpus.prerequis}
    ids_metiers = {m.id for m in corpus.metiers}

    for parcours in corpus.parcours:
        _relier(
            graphe, "Parcours", parcours.id, "enseigne", "Matiere", parcours.matieres, ids_matieres
        )
        _relier(
            graphe,
            "Parcours",
            parcours.id,
            "developpe",
            "Competence",
            parcours.competences,
            ids_competences,
        )
        _relier(
            graphe,
            "Parcours",
            parcours.id,
            "necessite",
            "Prerequis",
            parcours.prerequis,
            ids_prerequis,
        )
        _relier(
            graphe, "Parcours", parcours.id, "prepareA", "Metier", parcours.debouches, ids_metiers
        )
        # `mention_id` est un identifiant seul, pas une liste : emballé ici pour
        # passer par le même chemin de validation que les autres relations.
        _relier(
            graphe,
            "Parcours",
            parcours.id,
            "appartientA",
            "Mention",
            [parcours.mention_id],
            ids_mentions,
        )
        _relier(
            graphe,
            "Parcours",
            parcours.id,
            "passerelleVers",
            "Parcours",
            parcours.passerelles,
            ids_parcours,
        )

    for competence in corpus.competences:
        _relier(
            graphe,
            "Competence",
            competence.id,
            "estRequisePour",
            "Metier",
            competence.metiers_requis,
            ids_metiers,
        )

    return graphe


def prerequis_du_parcours(graphe: nx.DiGraph, parcours_id: str) -> list[str]:
    """Descriptions des prérequis d'admission d'un parcours (ONTO-3).

    Requête déterministe du graphe (relation `necessite`) — pas d'appel LLM.
    Liste vide si le parcours est inconnu du graphe ou n'a aucun prérequis
    relié (y compris une référence orpheline, silencieusement absente du
    graphe depuis `construire_graphe`)."""
    noeud = id_noeud("Parcours", parcours_id)
    if noeud not in graphe:
        return []
    return [
        graphe.nodes[cible]["nom"]
        for _, cible, donnees in graphe.out_edges(noeud, data=True)
        if donnees.get("relation") == "necessite"
    ]


def chemin_competence_parcours_metier(graphe: nx.DiGraph, parcours_id: str) -> list[dict]:
    """Chemins Compétence → Parcours → Métier partant d'un parcours (ONTO-5).

    Pour chaque compétence que ce parcours développe (`developpe`) et qui est
    à son tour explicitement requise pour un métier (`estRequisePour`),
    retourne le triplet et le chemin de nœuds correspondant — de quoi enrichir
    `expliquer_recommandation` d'un raisonnement multiétape traçable plutôt
    que du seul score du modèle ML.

    Une liste vide signifie qu'aucun chemin de ce type n'est démontrable avec
    les données actuellement disponibles (compétences ou métiers pas encore
    collectés pour ce parcours, voir BACKLOG.md DATA-1) — pas l'absence d'un
    lien réel entre ce parcours et un métier.
    """
    noeud_parcours = id_noeud("Parcours", parcours_id)
    if noeud_parcours not in graphe:
        return []

    chemins = []
    for _, noeud_competence, donnees_pc in graphe.out_edges(noeud_parcours, data=True):
        if donnees_pc.get("relation") != "developpe":
            continue
        for _, noeud_metier, donnees_cm in graphe.out_edges(noeud_competence, data=True):
            if donnees_cm.get("relation") != "estRequisePour":
                continue
            chemins.append(
                {
                    "parcours": parcours_id,
                    "competence": graphe.nodes[noeud_competence]["nom"],
                    "metier": graphe.nodes[noeud_metier]["nom"],
                    "chemin": [noeud_parcours, noeud_competence, noeud_metier],
                }
            )
    return chemins


def _references_orphelines(corpus: CorpusFormations) -> list[dict]:
    """Tout identifiant référencé par une entité mais absent du corpus.

    Couvre **tous** les champs de référence du corpus structuré, pas seulement
    les listes de `Parcours` : `Parcours.mention_id` et `Parcours.passerelles`,
    ainsi que `Competence.metiers_requis`, sont des références au même titre.
    Une omission ici est silencieuse de bout en bout — `construire_graphe`
    ignore l'arête et plus rien ne signale le problème — d'où la table
    explicite ci-dessous plutôt qu'une énumération partielle.
    """
    ids_connues: dict[str, set[str]] = {
        "Mention": {m.id for m in corpus.mentions},
        "Parcours": {p.id for p in corpus.parcours},
        "Matiere": {m.id for m in corpus.matieres},
        "Competence": {c.id for c in corpus.competences},
        "Prerequis": {p.id for p in corpus.prerequis},
        "Metier": {m.id for m in corpus.metiers},
    }
    # (entités porteuses, type de l'entité porteuse, champ, type cible, le
    # champ est-il une liste ?)
    sources: tuple[tuple[list, TypeEntite, str, TypeEntite, bool], ...] = (
        (corpus.parcours, "Parcours", "matieres", "Matiere", True),
        (corpus.parcours, "Parcours", "competences", "Competence", True),
        (corpus.parcours, "Parcours", "prerequis", "Prerequis", True),
        (corpus.parcours, "Parcours", "debouches", "Metier", True),
        (corpus.parcours, "Parcours", "passerelles", "Parcours", True),
        (corpus.parcours, "Parcours", "mention_id", "Mention", False),
        (corpus.competences, "Competence", "metiers_requis", "Metier", True),
    )

    incoherences = []
    for entites, type_source, champ, type_cible, est_liste in sources:
        for entite in entites:
            valeur = getattr(entite, champ)
            references = valeur if est_liste else ([valeur] if valeur is not None else [])
            for id_reference in references:
                if id_reference in ids_connues[type_cible]:
                    continue
                incoherences.append(
                    {
                        "type": "reference_orpheline",
                        "entite": entite.id,
                        "type_entite": type_source,
                        # Conservé pour compatibilité : les parcours restent le
                        # cas de loin le plus fréquent et l'agent lit ce champ.
                        "parcours": entite.id if type_source == "Parcours" else None,
                        "champ": champ,
                        "id_reference": id_reference,
                        "message": (
                            f"{type_source} {entite.id} référence {type_cible} "
                            f"« {id_reference} », absent du corpus structuré."
                        ),
                    }
                )
    return incoherences


def _parcours_sans_debouche(corpus: CorpusFormations) -> list[dict]:
    """Parcours dont aucun débouché n'est renseigné (exemple cité au §12).

    Formulé comme une **donnée manquante**, pas comme une contradiction du
    corpus : sur l'état actuel de DATA-1 aucun débouché n'a encore de source
    fiable, donc ce contrôle remonte les 16 parcours de l'ISPM. Le message le
    dit explicitement pour qu'un agent qui le relaie ne transforme pas un
    chantier de collecte en défaut de fiabilité — même formulation que
    `tools.identifier_debouches`, qui rapporte le même état sous
    `information_manquante`.
    """
    return [
        {
            "type": "parcours_sans_debouche",
            "parcours": parcours.id,
            "donnee_manquante": True,
            "message": (
                f"Aucun débouché professionnel n'est encore renseigné pour le parcours "
                f"{parcours.id} : information non collectée à ce stade du projet "
                f"(voir BACKLOG.md, DATA-1), et non une contradiction du corpus."
            ),
        }
        for parcours in corpus.parcours
        if not parcours.debouches
    ]


def _competences_requises_sans_chemin_verifiable(graphe: nx.DiGraph) -> list[dict]:
    """Compétence explicitement requise pour un métier (`estRequisePour`) mais
    pour laquelle aucun chemin fiable ne permet de l'acquérir : soit aucun
    parcours connu ne la développe (compétence inaccessible), soit **aucun**
    des parcours qui la développent n'a de prérequis d'admission connu
    (impossible de vérifier qu'un candidat peut seulement y accéder).

    Le « aucun » de la seconde catégorie est strict, et c'est ce qui rend le
    constat vrai : dès qu'un seul parcours développeur a des prérequis connus,
    l'accès à la compétence *est* vérifiable par cette voie, et signaler une
    incohérence reviendrait à inventer un défaut de corpus — que l'agent
    relaierait ensuite à un étudiant comme un fait.
    """
    incoherences = []
    for noeud, donnees_noeud in graphe.nodes(data=True):
        if donnees_noeud.get("type") != "Competence":
            continue
        _, id_competence = type_et_id(noeud)

        metiers = [
            graphe.nodes[cible]["nom"]
            for _, cible, donnees in graphe.out_edges(noeud, data=True)
            if donnees.get("relation") == "estRequisePour"
        ]
        if not metiers:
            continue

        parcours_developpeurs = [
            type_et_id(source)[1]
            for source, _, donnees in graphe.in_edges(noeud, data=True)
            if donnees.get("relation") == "developpe"
        ]
        if not parcours_developpeurs:
            incoherences.append(
                {
                    "type": "competence_requise_sans_parcours",
                    "competence": id_competence,
                    "metiers": metiers,
                    "message": (
                        f"La compétence « {donnees_noeud['nom']} » est requise pour "
                        f"{', '.join(metiers)} mais n'est développée par aucun "
                        f"parcours connu."
                    ),
                }
            )
            continue

        sans_prerequis = [
            pid for pid in parcours_developpeurs if not prerequis_du_parcours(graphe, pid)
        ]
        if len(sans_prerequis) == len(parcours_developpeurs):
            incoherences.append(
                {
                    "type": "competence_requise_sans_prerequis_verifiable",
                    "competence": id_competence,
                    "metiers": metiers,
                    "parcours_sans_prerequis": sans_prerequis,
                    "message": (
                        f"La compétence « {donnees_noeud['nom']} », requise pour "
                        f"{', '.join(metiers)}, ne s'acquiert que via "
                        f"{', '.join(sans_prerequis)}, dont aucun prérequis "
                        f"d'admission n'est connu : impossible de vérifier si un "
                        f"candidat peut y accéder."
                    ),
                }
            )
    return incoherences


def detecter_incoherences(corpus: CorpusFormations, graphe: nx.DiGraph) -> list[dict]:
    """Détection déterministe d'incohérences structurelles du corpus (ONTO-4).

    Trois catégories, chacune avec un `type` stable et un `message` lisible :
    référence orpheline (identifiant utilisé par un parcours mais absent du
    corpus), parcours sans débouché renseigné (exemple cité au §12 du sujet),
    compétence requise pour un métier mais inaccessible ou dont l'accès n'est
    pas vérifiable (autre exemple cité au §12).

    Ne remplace pas une relecture humaine du corpus, mais rend visible d'un
    coup ce qu'une lecture manuelle des fichiers JSON manquerait facilement —
    voir `backend/tests/eval_ontologie.py` (ONTO-6) pour la mesure de cet
    apport sur le corpus réel."""
    return (
        _references_orphelines(corpus)
        + _parcours_sans_debouche(corpus)
        + _competences_requises_sans_chemin_verifiable(graphe)
    )
