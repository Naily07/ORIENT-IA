# Travaux tiers réutilisés

ORIENT'IA réutilise du code provenant d'un projet tiers. Cette page le recense
avec sa licence, comme celle-ci l'exige.

La provenance des **données** est portée séparément, dans les données
elles-mêmes : `backend/data/registre_sources.json` décrit chaque source, et
chaque enregistrement porte le `source_id` correspondant. Le contrôle
`src.sources.verifier_provenance()` échoue si une donnée référence une source
absente du registre.

---

## X-project-ISPM/EXAM-S2

- **Dépôt** : https://github.com/X-project-ISPM/EXAM-S2
- **Licence** : MIT — notice de copyright dans le fichier
  [`LICENSE`](https://github.com/X-project-ISPM/EXAM-S2/blob/main/LICENSE) du
  dépôt d'origine, qui fait foi

Rendu d'un hackathon ISPM précédent par la même organisation. Plusieurs
modules d'infrastructure domaine-agnostiques en sont adaptés : client LLM,
observabilité, moteur RAG, garde-fous anti-injection, sortie structurée.
L'analyse de ce qui a été repris, adapté ou écarté figure en tête de
[`BACKLOG.md`](BACKLOG.md).
