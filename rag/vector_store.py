"""
Thin wrapper around Chroma so the rest of the codebase depends on our own
interface, not directly on a specific vector DB. Swapping to pgvector/
Pinecone/Weaviate later only requires changes here.
"""

import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config.settings import settings

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection_name: str | None = None):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        # Sentence-transformers embedding function runs locally (no API
        # cost/latency) -- important for a system that may process PHI-adjacent
        # data and shouldn't send it to a third-party embedding API.
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="pritamdeka/S-PubMedBert-MS-MARCO"  # domain-tuned biomedical embeddings
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.vector_collection,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]
    ) -> None:
        """Upserts documents in batches to avoid memory spikes on large corpora."""
        batch_size = 256
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        log.info(
            "Upserted %d documents into collection '%s'", len(ids), self.collection.name
        )

    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Dense similarity search. Returns list of {id, text, metadata, score}."""
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        out = []
        for doc, meta, dist, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            # Chroma returns cosine *distance*; convert to a similarity score
            out.append({"id": doc_id, "text": doc, "metadata": meta, "score": 1 - dist})
        return out

    def count(self) -> int:
        return self.collection.count()
