"""Robustesse de la sortie structurée d'un appel LLM.

`generer_avec_retry()` : quand un LLM produit une sortie non conforme au
schéma attendu, on ne l'accepte jamais telle quelle et on ne crashe pas non
plus — on régénère une fois avec le message d'erreur de validation intégré au
prompt (« ta réponse précédente ne respectait pas le schéma : ... »), puis on
abandonne proprement si l'échec persiste.

Repris tel quel d'un hackathon ISPM précédent : mécanisme générique, sans
dépendance à un schéma de décision particulier. La réponse de repli
« toujours valide » à produire quand toutes les tentatives échouent dépend en
revanche du schéma de décision d'ORIENT'IA (`RecommandationDecision`, encore à
définir) — elle viendra avec l'orchestrateur, pas ici.
"""

from pydantic import BaseModel, ValidationError

from src.config import config
from src.llm_client import SchemaNonConforme

# `prompt_fn` peut retourner un objet déjà validé par `llm_call(...,
# response_schema=...)` (qui lève `SchemaNonConforme`, pas `ValidationError`,
# quand le modèle ne respecte pas le schéma — voir llm_client.py) ou un dict
# brut à valider ici même. On réagit aux deux : toute autre exception (réseau,
# quota) doit continuer à remonter immédiatement, ce n'est pas un problème de
# schéma qu'une régénération pourrait corriger.
_ERREURS_SCHEMA = (ValidationError, SchemaNonConforme)


def generer_avec_retry(
    prompt_fn,
    schema: type[BaseModel],
    max_essais: int | None = None,
) -> BaseModel:
    """Génère une sortie conforme au schéma, avec une régénération sur échec.

    `prompt_fn` est le fournisseur de la génération : il reçoit
    `erreur_precedente` (None au 1er essai, l'erreur de schéma précédente
    ensuite) et retourne soit l'objet déjà validé, soit un dict brut — dans
    les deux cas, c'est lui qui sait intégrer l'erreur dans le prompt de
    régénération. Plusieurs étapes du pipeline (extraction de profil, RAG,
    recommandation) peuvent ainsi partager la même stratégie sans la
    dupliquer.

    Lève `RuntimeError` si toutes les tentatives échouent : à l'appelant de
    dégrader (ex. escalade vers un conseiller pédagogique).
    """
    essais = max_essais if max_essais is not None else config.llm_max_essais_validation
    derniere_erreur: Exception | None = None

    for _ in range(essais):
        try:
            brut = prompt_fn(erreur_precedente=derniere_erreur)
            return schema.model_validate(brut)
        except _ERREURS_SCHEMA as e:
            derniere_erreur = e

    raise RuntimeError(
        f"Échec de génération conforme à {schema.__name__} "
        f"après {essais} essais. "
        f"Dernière erreur de validation : {derniere_erreur}"
    ) from derniere_erreur
