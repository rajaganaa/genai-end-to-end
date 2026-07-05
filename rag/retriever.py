"""
Hybrid retrieval: dense (embedding) search + BM25 lexical search, fused and
then reranked with a cross-encoder. Medical queries need lexical precision
(exact drug names, ICD-10 codes, dosage units) that pure dense retrieval
sometimes misses, so we combine both rather than relying on embeddings alone.
"""

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import settings
from rag.vector_store import VectorStore
from rag.parent_child_store import ParentDocumentStore
from rag.hyde import HyDEGenerator
from rag.query_decomposition import QueryDecomposer
from rag.self_rag import SelfRAGReflector
from monitoring.metrics import RETRIEVAL_LATENCY, RETRIEVAL_RELEVANCE_SCORE
from monitoring.tracing import get_tracer

log = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    text: str
    source: str
    score: float


class MedicalRetriever:
    def __init__(self):
        self.store = VectorStore()
        # Cross-encoder reranker: much more accurate than bi-encoder
        # similarity alone, at the cost of being too slow to run over the
        # whole corpus -- so we only rerank the top-k candidates.
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._bm25_index = None
        self._bm25_corpus_ids: list[str] = []
        self._bm25_corpus_texts: list[str] = []

        # Advanced RAG components (each independently toggleable via settings)
        self.parent_store = ParentDocumentStore()
        self.hyde = HyDEGenerator()
        self.decomposer = QueryDecomposer()
        self.reflector = SelfRAGReflector()

    def _lazy_build_bm25(self):
        """BM25 needs the full corpus tokenized in memory; build once and
        cache. For very large corpora, replace with an Elasticsearch/OpenSearch
        BM25 index instead of doing this in-process."""
        if self._bm25_index is not None:
            return
        all_docs = self.store.collection.get()
        self._bm25_corpus_ids = all_docs["ids"]
        self._bm25_corpus_texts = all_docs["documents"]
        tokenized = [doc.lower().split() for doc in self._bm25_corpus_texts]
        if not tokenized:
            self._bm25_index = None
            log.warning("BM25 index not built: vector store is empty (0 documents)")
            return
        self._bm25_index = BM25Okapi(tokenized)
        log.info("Built BM25 index over %d documents", len(tokenized))

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        self._lazy_build_bm25()
        if self._bm25_index is None:
            return []
        scores = self._bm25_index.get_scores(query.lower().split())
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            {
                "id": self._bm25_corpus_ids[i],
                "text": self._bm25_corpus_texts[i],
                "score": scores[i],
            }
            for i in ranked_idx
        ]

    def retrieve(self, query: str) -> list[RetrievedPassage]:
        """Returns reranked, deduplicated passages above the relevance floor.
        Empty list signals "insufficient evidence" to the caller -- callers
        must NOT let the LLM answer unsupported in that case."""
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "rag.retrieve"
        ) as span, RETRIEVAL_LATENCY.time():
            span.set_attribute("rag.query_length", len(query))

            # HyDE: search with a hypothetical answer passage instead of the
            # raw question, to close the question/answer phrasing gap.
            # Note: we search with the hypothetical text but rerank against
            # the *original* query, since reranking benefits from the user's
            # actual intent/phrasing rather than the LLM's guess at an answer.
            search_text = self.hyde.embed_query(query)
            span.set_attribute("rag.hyde_used", search_text != query)

            dense_hits = self.store.query(search_text, top_k=settings.retrieval_top_k)
            lexical_hits = self._bm25_search(query, top_k=settings.retrieval_top_k)

            # Merge candidate pools by id, dense hits take priority on collision
            merged: dict[str, dict] = {h["id"]: h for h in lexical_hits}
            merged.update({h["id"]: h for h in dense_hits})
            candidates = list(merged.values())
            span.set_attribute("rag.candidate_count", len(candidates))

            if not candidates:
                RETRIEVAL_RELEVANCE_SCORE.observe(0.0)
                return []

            # Cross-encoder reranking: scores (query, passage) pairs jointly,
            # which is more accurate than comparing independent embeddings.
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)

            scored = list(zip(candidates, rerank_scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            RETRIEVAL_RELEVANCE_SCORE.observe(float(scored[0][1]) if scored else 0.0)

            results = []
            seen_parent_ids = set()
            for candidate, score in scored[: settings.rerank_top_k]:
                if score < settings.min_relevance_score:
                    continue
                metadata = candidate.get("metadata", {})
                parent_id = metadata.get("parent_id")

                # Parent-child expansion: return the full parent chunk (more
                # context) rather than just the small child chunk that
                # matched, so the LLM sees surrounding caveats/details.
                # Skip if we've already included this parent from a
                # different matching child, to avoid duplicate context.
                if parent_id and parent_id not in seen_parent_ids:
                    parent_text = self.parent_store.get_parent(parent_id)
                    if parent_text:
                        seen_parent_ids.add(parent_id)
                        results.append(
                            RetrievedPassage(
                                text=parent_text,
                                source=metadata.get("source_file", candidate["id"]),
                                score=float(score),
                            )
                        )
                        continue

                # Fallback: no parent mapping available (e.g. legacy
                # flat-chunked data) -- use the matched text directly.
                results.append(
                    RetrievedPassage(
                        text=candidate["text"],
                        source=metadata.get("source_file", candidate["id"]),
                        score=float(score),
                    )
                )
            span.set_attribute("rag.passages_returned", len(results))
            return results

    def retrieve_multi_hop(
        self, query: str
    ) -> tuple[list[RetrievedPassage], "SelfRAGResult"]:
        """Entry point that combines query decomposition + retrieval +
        self-RAG grading. This is what agents/tools.py::rag_search should
        call for full "advanced RAG" behavior; retrieve() alone remains
        available for callers that just want raw hybrid retrieval.
        """

        sub_queries = self.decomposer.decompose(query)
        log.debug(
            "Decomposed %r into %d sub-question(s): %s",
            query,
            len(sub_queries),
            sub_queries,
        )

        all_passages: list[RetrievedPassage] = []
        seen_texts = set()
        for sub_q in sub_queries:
            for passage in self.retrieve(sub_q):
                if passage.text not in seen_texts:
                    seen_texts.add(passage.text)
                    all_passages.append(passage)

        # Cap total context sent to the LLM even if decomposition fanned out
        all_passages.sort(key=lambda p: p.score, reverse=True)
        all_passages = all_passages[: settings.rerank_top_k * 2]

        passages_text = "\n\n".join(f"[{p.source}] {p.text}" for p in all_passages)
        grading = self.reflector.grade(query, passages_text)

        return all_passages, grading
