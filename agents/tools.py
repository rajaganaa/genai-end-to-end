"""
LangChain-compatible tools used by the agent. Each tool has a narrow,
auditable responsibility -- this matters in a safety-critical domain where
we want to be able to point at exactly which component produced a given
claim (retrieval vs. a deterministic lookup vs. the LLM's own phrasing).
"""

import csv
import logging
from pathlib import Path

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.prompts import (
    EMERGENCY_PATTERNS,
    EMERGENCY_RESPONSE,
    INSUFFICIENT_EVIDENCE_MSG,
)
from rag.retriever import MedicalRetriever
from monitoring.metrics import TOOL_CALL_COUNT, INSUFFICIENT_EVIDENCE_COUNT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emergency Triage Tool -- deterministic, runs first, can short-circuit
# ---------------------------------------------------------------------------
class EmergencyTriageInput(BaseModel):
    symptom_description: str = Field(..., description="Raw patient/user symptom text")


def check_emergency(symptom_description: str) -> str:
    """Rule-based (not LLM-based) emergency detector. Uses regex patterns
    (agents/prompts.py::EMERGENCY_PATTERNS) so it tolerates natural
    phrasing variation, not just exact literal substrings."""
    TOOL_CALL_COUNT.labels(tool_name="EmergencyTriageCheck").inc()
    for pattern in EMERGENCY_PATTERNS:
        if pattern.search(symptom_description):
            log.warning("Emergency pattern matched: '%s'", pattern.pattern)
            return EMERGENCY_RESPONSE
    return "NO_EMERGENCY_DETECTED"


emergency_triage_tool = StructuredTool.from_function(
    func=check_emergency,
    name="EmergencyTriageCheck",
    description=(
        "Call this FIRST on every user message to check for emergency "
        "symptom patterns before doing anything else. Returns "
        "'NO_EMERGENCY_DETECTED' if safe to proceed, otherwise returns the "
        "emergency response to relay to the user verbatim."
    ),
    args_schema=EmergencyTriageInput,
)


# ---------------------------------------------------------------------------
# RAG Tool -- retrieval over the medical knowledge base
# ---------------------------------------------------------------------------
class RAGQueryInput(BaseModel):
    query: str = Field(
        ..., description="Clinical question to search the knowledge base for"
    )


_retriever = MedicalRetriever()  # module-level singleton, index built lazily


def rag_search(query: str) -> str:
    """Full advanced-RAG path: query decomposition (multi-hop) -> hybrid
    retrieval with HyDE -> parent-child context expansion -> self-RAG
    confidence grading. If the reflector judges the retrieved evidence
    insufficient to actually answer the question (not just "topically
    related"), we return the insufficient-evidence message rather than
    letting the agent guess."""
    TOOL_CALL_COUNT.labels(tool_name="MedicalKnowledgeSearch").inc()
    passages, grading = _retriever.retrieve_multi_hop(query)

    if not passages or not grading.should_answer:
        INSUFFICIENT_EVIDENCE_COUNT.inc()
        log.info(
            "Self-RAG deemed evidence insufficient (confidence=%.2f, reason=%r)",
            grading.confidence,
            grading.reason,
        )
        return INSUFFICIENT_EVIDENCE_MSG

    formatted = "\n\n".join(
        f"[Source: {p.source} | relevance={p.score:.2f}]\n{p.text}" for p in passages
    )
    return formatted


rag_tool = StructuredTool.from_function(
    func=rag_search,
    name="MedicalKnowledgeSearch",
    description=(
        "Search the medical knowledge base (clinical guidelines, formulary, "
        "reference material) for passages relevant to a clinical question. "
        "Always call this before answering any factual medical question, "
        "and cite the returned sources in your answer."
    ),
    args_schema=RAGQueryInput,
)


# ---------------------------------------------------------------------------
# Drug Interaction Tool -- deterministic lookup, not LLM-generated
# ---------------------------------------------------------------------------
class DrugInteractionInput(BaseModel):
    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")


class DrugInteractionDB:
    """Loads a simple CSV lookup table. In production this should be backed
    by a licensed drug-interaction database (e.g. First Databank, Micromedex)
    via API, not a static CSV -- this is a swappable placeholder."""

    def __init__(self, csv_path: str = "data/raw/drug_interactions.csv"):
        self.table: dict[tuple[str, str], str] = {}
        path = Path(csv_path)
        if not path.exists():
            log.warning("Drug interaction DB not found at %s", csv_path)
            return
        with open(path) as f:
            for row in csv.DictReader(f):
                key = tuple(sorted([row["drug_a"].lower(), row["drug_b"].lower()]))
                self.table[key] = row["interaction"]

    def lookup(self, drug_a: str, drug_b: str) -> str | None:
        key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
        return self.table.get(key)


_drug_db = DrugInteractionDB()


def drug_interaction_lookup(drug_a: str, drug_b: str) -> str:
    TOOL_CALL_COUNT.labels(tool_name="DrugInteractionLookup").inc()
    result = _drug_db.lookup(drug_a, drug_b)
    if result is None:
        return (
            f"No documented interaction found between {drug_a} and {drug_b} "
            "in the local database. This does NOT rule out an interaction -- "
            "verify with a pharmacist or a licensed drug-interaction database."
        )
    return f"Interaction between {drug_a} and {drug_b}: {result}"


drug_interaction_tool = StructuredTool.from_function(
    func=drug_interaction_lookup,
    name="DrugInteractionLookup",
    description=(
        "Deterministic lookup for known interactions between two drugs. "
        "Always use this tool's output verbatim rather than reasoning about "
        "drug interactions yourself."
    ),
    args_schema=DrugInteractionInput,
)


ALL_TOOLS = [emergency_triage_tool, rag_tool, drug_interaction_tool]
