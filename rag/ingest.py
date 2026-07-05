"""
Chunks and embeds source documents (clinical guidelines, drug formularies,
de-identified notes) into the vector store used by the RAG tool, using a
**parent-child chunking strategy**:

  - "Parent" chunks (~2048 tokens): large enough to preserve full context
    around a clinical fact (e.g. a dosage AND its adjacent contraindication
    caveat), but too large/diffuse to embed precisely.
  - "Child" chunks (~512 tokens): small slices *within* each parent, small
    enough for precise embedding similarity, tagged with a `parent_id`.

We embed and search over children (precision), but at answer time we
resolve each matched child back to its full parent text (context) via
ParentDocumentStore -- see rag/retriever.py's `_expand_to_parent`.

Usage:
    python rag/ingest.py --source data/raw --collection medical_kb
"""

import argparse
import hashlib
import logging
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from rag.vector_store import VectorStore
from rag.parent_child_store import ParentDocumentStore
from config.settings import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_source_files(source_dir: Path) -> list[tuple[str, str]]:
    """Returns list of (filename, raw_text). Extend this to add PDF/HTML
    parsers (e.g. via pypdf or llama_index readers) as needed."""
    docs = []
    for path in source_dir.rglob("*.txt"):
        docs.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
    for path in source_dir.rglob("*.md"):
        docs.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
    return docs


def chunk_id(source: str, idx: int, text: str) -> str:
    """Deterministic ID so re-running ingestion upserts instead of duplicating."""
    h = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{source}-{idx}-{h}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="directory of source docs")
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Two splitters: coarse for parents, fine for children within each parent.
    parent_splitter = SentenceSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_size // 8,
    )
    child_splitter = SentenceSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_size // 8,
    )

    store = VectorStore(collection_name=args.collection)
    parent_store = ParentDocumentStore()

    docs = load_source_files(source_dir)
    log.info("Found %d source documents in %s", len(docs), source_dir)

    child_ids, child_texts, child_metadatas = [], [], []
    parents_batch: dict[str, str] = {}

    for filename, raw_text in docs:
        parent_chunks = parent_splitter.split_text(raw_text)
        for p_idx, parent_text in enumerate(parent_chunks):
            if len(parent_text.strip()) < 20:
                continue
            parent_id = chunk_id(filename, p_idx, parent_text)
            parents_batch[parent_id] = parent_text

            child_chunks = child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_chunks):
                if len(child_text.strip()) < 20:
                    continue
                cid = f"{parent_id}-child-{c_idx}"
                child_ids.append(cid)
                child_texts.append(child_text)
                child_metadatas.append(
                    {
                        "source_file": filename,
                        "parent_id": parent_id,
                        "child_index": c_idx,
                    }
                )

    if not child_ids:
        log.warning("No chunks produced -- check source directory contents.")
        return

    # Parents are looked up by id at query time, never embedded/searched
    # directly -- only children go into the vector store.
    parent_store.add_parents_batch(parents_batch)
    store.add_documents(ids=child_ids, texts=child_texts, metadatas=child_metadatas)

    log.info(
        "Ingestion complete. %d parent chunks, %d child chunks embedded.",
        len(parents_batch),
        store.count(),
    )


if __name__ == "__main__":
    main()
