"""Corpus jouet partagé par les tests et par l'évaluation de l'ontologie.

Trois consommateurs construisaient auparavant leur propre variante quasi
identique (`test_graphe.py`, `test_tools.py`, `eval_ontologie.py`). Un ajout
au schéma du corpus — comme `Competence.metiers_requis` — devait alors être
répercuté trois fois, et la copie de l'évaluation, qu'aucun test n'assertit,
aurait dérivé en silence : la preuve ONTO-6 aurait cessé de correspondre à ce
que les tests valident.

Deux variantes explicites plutôt qu'un seul corpus paramétrable :
- `corpus_coherent()` — tout est relié et résolvable, aucune incohérence
  attendue. C'est le corpus « tel que DATA-1 le vise à terme », utilisé pour
  démontrer les requêtes de graphe (ONTO-3, ONTO-5).
- `corpus_avec_incoherences()` — le précédent, augmenté de cas défectueux
  choisis pour exercer chaque catégorie d'ONTO-4.
"""

from src.models import (
    Competence,
    CorpusFormations,
    Matiere,
    Mention,
    Metier,
    Parcours,
    Prerequis,
)


def corpus_coherent() -> CorpusFormations:
    """Corpus jouet entièrement peuplé et sans défaut.

    IGGLIA développe la programmation, qui est requise pour Développeur
    logiciel : le chemin Compétence → Parcours → Métier d'ONTO-5 y est
    démontrable. TEH n'a pas de débouché renseigné (état réel de DATA-1 sur le
    corpus ISPM), donc `detecter_incoherences` y remonte cette seule
    donnée manquante.
    """
    return CorpusFormations(
        mentions=[
            Mention(id="MENTION-INFO", nom="Informatique et Télécommunications", niveau="Licence"),
            Mention(id="MENTION-TOURISME", nom="Tourisme", niveau="Licence"),
        ],
        parcours=[
            Parcours(
                id="IGGLIA",
                nom="Informatique de Gestion, Génie Logiciel et IA",
                mention_id="MENTION-INFO",
                matieres=["MAT-INFO"],
                competences=["COMP-PROG"],
                prerequis=["PREREQ-SCIENTIFIQUE"],
                debouches=["METIER-DEV"],
                source_id="FORM-IGGLIA-JOUET",
            ),
            Parcours(
                id="TEH",
                nom="Tourisme et Hôtellerie",
                mention_id="MENTION-TOURISME",
                prerequis=["PREREQ-TOUTE-SERIE"],
            ),
        ],
        matieres=[Matiere(id="MAT-INFO", nom="informatique")],
        competences=[
            Competence(id="COMP-PROG", nom="programmation", metiers_requis=["METIER-DEV"])
        ],
        prerequis=[
            Prerequis(id="PREREQ-SCIENTIFIQUE", description="Baccalauréat série C, D, S"),
            Prerequis(id="PREREQ-TOUTE-SERIE", description="Baccalauréat toute série"),
        ],
        metiers=[Metier(id="METIER-DEV", nom="Développeur logiciel")],
    )


def corpus_avec_incoherences() -> CorpusFormations:
    """`corpus_coherent()` augmenté d'un défaut par catégorie d'ONTO-4.

    - `TEH.prerequis` pointe vers un prérequis inexistant → référence orpheline
      sur un `Parcours` ;
    - `COMP-ISOLEE` est requise pour un métier mais aucun parcours ne la
      développe → compétence inaccessible ;
    - `CAA` développe `COMP-NEGO` sans déclarer de prérequis, et c'est le seul
      parcours à la développer → accès non vérifiable ;
    - `COMP-MIXTE` est développée par IGGLIA (avec prérequis) **et** par CAA
      (sans) : elle ne doit **pas** être signalée, l'accès restant vérifiable
      via IGGLIA. C'est le cas de non-régression du bug d'inversion any/all ;
    - `COMP-ORPHELINE.metiers_requis` pointe vers un métier inexistant →
      référence orpheline sur une `Competence`.
    """
    corpus = corpus_coherent()

    teh = next(p for p in corpus.parcours if p.id == "TEH")
    teh.prerequis = ["PREREQ-INEXISTANT"]

    igglia = next(p for p in corpus.parcours if p.id == "IGGLIA")
    igglia.competences = ["COMP-PROG", "COMP-MIXTE"]

    corpus.parcours.append(
        Parcours(
            id="CAA",
            nom="Commerce et Administration des Affaires",
            mention_id="MENTION-AFFAIRES",  # mention inexistante : orpheline aussi
            competences=["COMP-NEGO", "COMP-MIXTE"],
        )
    )
    corpus.competences.extend(
        [
            Competence(id="COMP-NEGO", nom="négociation", metiers_requis=["METIER-COMMERCIAL"]),
            Competence(id="COMP-MIXTE", nom="gestion de projet", metiers_requis=["METIER-DEV"]),
            Competence(
                id="COMP-ISOLEE", nom="compétence sans parcours", metiers_requis=["METIER-DEV"]
            ),
            Competence(
                id="COMP-ORPHELINE", nom="compétence orpheline", metiers_requis=["METIER-X"]
            ),
        ]
    )
    corpus.metiers.append(Metier(id="METIER-COMMERCIAL", nom="Commercial"))
    return corpus
