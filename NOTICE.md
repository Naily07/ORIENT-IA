# Travaux tiers réutilisés

ORIENT'IA réutilise des données et du code provenant de projets tiers. Cette
page les recense avec leur licence et leur titulaire, comme ces licences
l'exigent.

La provenance est également portée **dans les données elles-mêmes** :
`backend/data/registre_sources.json` décrit chaque source, et chaque
enregistrement dérivé d'un tiers porte le `source_id` correspondant. Le
contrôle `src.sources.verifier_provenance()` échoue si une donnée référence
une source absente du registre.

---

## source externe

- **Dépôt** : source externe
- **Licence** : MIT
- **Notice de copyright** : `MIT License — Copyright (c) 2026 source externe`
- **Identifiant interne** : `SRC-EXTERNE`

Équipe tierce ayant traité le même sujet ISPM. Deux éléments réutilisés :

1. **Débouchés professionnels** — 66 métiers rattachés aux 16 parcours
   (`backend/data/metiers.json`, `backend/data/parcours.json`). Comble un
   manque de notre collecte : le site officiel de l'ISPM ne publie pas ces
   listes.
2. **Réponses d'enquête** — 86 réponses anonymisées, transformées vers notre
   schéma par `backend/src/enquete_import.py`
   (`backend/data/enquete/reponses_anonymisees.json`).

**Statut : externe.** Ces données ne sont pas confirmées par l'ISPM et ne
doivent jamais être présentées comme officielles. Un recoupement de leurs
prérequis d'admission avec notre source officielle (`ispm-edu.com`) donne
14 concordances sur 16 ; en cas de divergence, notre source officielle fait
foi. Le détail figure dans le registre des sources.

---

## X-project-ISPM/EXAM-S2

- **Dépôt** : https://github.com/X-project-ISPM/EXAM-S2
- **Licence** : MIT
- **Notice de copyright** : `MIT License — Copyright (c) 2026 X-project-ISPM`

Rendu d'un hackathon ISPM précédent par la même organisation. Plusieurs
modules d'infrastructure domaine-agnostiques en sont adaptés : client LLM,
observabilité, moteur RAG, garde-fous anti-injection, sortie structurée.
L'analyse de ce qui a été repris, adapté ou écarté figure en tête de
[`BACKLOG.md`](BACKLOG.md).
