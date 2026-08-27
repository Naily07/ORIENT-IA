"""Préchauffage de l'image de déploiement — exécuté au *build*, jamais au runtime.

Le premier démarrage de l'API paie deux coûts que ce script déplace dans la
couche d'image Docker, une fois pour toutes :

1. **Téléchargement du modèle d'embedding ONNX** (~80 Mo, `all-MiniLM-L6-v2`).
   ChromaDB le récupère depuis S3 au premier embedding et le met en cache sous
   `~/.cache/chroma/onnx_models/`. Sur un conteneur éphémère (Railway), ce cache
   est vide à chaque démarrage → retéléchargé à chaque boot.
2. **Ingestion du corpus RAG** : découpage + embedding de toutes les fiches,
   puis écriture de l'index Chroma dans `backend/chroma_db/`.

En lançant l'ingestion pendant le build, l'index **et** le cache du modèle sont
cuits dans l'image. Au runtime, `rag.index_a_jour()` retrouve l'empreinte du
corpus déjà indexée et `api.lifespan()` saute complètement la ré-ingestion —
le conteneur démarre à froid avec tout déjà sur le disque.

N'a besoin d'aucun secret : l'ingestion n'appelle pas le LLM, seulement le
modèle d'embedding local.

Lancement (build) : `cd backend && python -m scripts.prechauffer_deploiement`
"""

from __future__ import annotations

from src.models import charger_corpus_rag
from src.rag import empreinte_corpus, index_a_jour, ingerer, nombre_de_fragments


def main() -> None:
    documents = charger_corpus_rag()
    if not documents:
        # Cas anormal en déploiement (le corpus est versionné), mais on ne fait
        # pas échouer le build pour autant : l'API démarrera, RAG vide, et
        # l'ingestion sera simplement retentée au runtime.
        print("[prechauffage] aucun document à indexer — corpus absent ?")
        return

    if index_a_jour(documents):
        print(f"[prechauffage] index déjà à jour ({nombre_de_fragments()} fragments)")
        return

    nombre = ingerer(documents)
    print(
        f"[prechauffage] {nombre} fragments indexés depuis {len(documents)} documents "
        f"(empreinte {empreinte_corpus(documents)[:12]}…)"
    )


if __name__ == "__main__":
    main()
