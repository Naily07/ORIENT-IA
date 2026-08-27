# Image de déploiement de l'API ORIENT'IA (Railway).
#
# Objectif : un conteneur qui démarre à froid sans rien télécharger ni
# ré-indexer. Le modèle d'embedding ONNX et l'index RAG sont construits pendant
# le build et cuits dans l'image ; `api.lifespan()` n'a plus qu'à charger le
# corpus structuré (quelques ms) avant d'accepter les requêtes.
#
# Le frontend (`frontend/` Streamlit, `frontend-next/`) se déploie séparément.

FROM python:3.12-slim

# libgomp1 : OpenMP, requis par onnxruntime sur l'image slim.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/backend

WORKDIR /app/backend

# 1. Dépendances de l'API seule — couche cachée tant que le fichier ne bouge pas.
COPY backend/requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# 2. Modèle d'embedding ONNX (~80 Mo) cuit dans l'image. Couche indépendante du
#    code applicatif : elle n'est réinvalidée que si les dépendances changent.
#    Le .tar.gz est retiré après extraction — ChromaDB ne re-télécharge pas tant
#    que le dossier `onnx/` extrait est complet (voir _download_model_if_not_exists).
RUN python -c "from chromadb.utils import embedding_functions as ef; ef.ONNXMiniLM_L6_V2()(['prechauffage'])" \
 && rm -f /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

# 3. Code applicatif + données versionnées (corpus, registre de sources, jeux ML).
COPY backend/ .

# 4. Index RAG cuit dans l'image. `prechauffer_deploiement` écrit
#    backend/chroma_db/ + son empreinte ; au runtime `rag.index_a_jour()` la
#    reconnaît et `api.lifespan()` saute l'ingestion.
RUN python -m scripts.prechauffer_deploiement

# Railway injecte $PORT ; 8000 en repli pour un `docker run` local.
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
