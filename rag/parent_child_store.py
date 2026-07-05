"""
Parent-child chunking support: we embed and search over small "child"
chunks (precise semantic match), but return the larger "parent" chunk as
context to the LLM (enough surrounding text to avoid answering from a
fragment that's missing a crucial adjacent sentence -- e.g. a dosage
caveat two sentences after the dosage itself).

This module is the parent-id -> parent-text lookup store. Backed by a
simple JSON file for this reference implementation; swap for Redis/
DynamoDB/a Postgres table in production once corpus size warrants it.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PARENT_STORE_PATH = Path("chroma_data/parent_store.json")


class ParentDocumentStore:
    def __init__(self, path: Path = DEFAULT_PARENT_STORE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data))

    def add_parent(self, parent_id: str, text: str) -> None:
        self._data[parent_id] = text
        self._save()

    def add_parents_batch(self, parents: dict[str, str]) -> None:
        self._data.update(parents)
        self._save()

    def get_parent(self, parent_id: str) -> str | None:
        return self._data.get(parent_id)
