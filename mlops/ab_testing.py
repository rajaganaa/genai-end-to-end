"""
A/B testing between the base model and the fine-tuned (medical-lora)
model, routed at the gateway layer via vLLM's --served-model-name
mechanism (recall serving/vllm_server.sh serves both "base" and the LoRA
adapter from one process).

Two things live here:
  1. `assign_variant`: deterministic traffic splitting (same session_id
     always gets the same variant, so a multi-turn conversation doesn't
     flip models mid-conversation).
  2. `ABTestAnalyzer`: offline comparison of logged outcomes between
     variants (e.g. safety-eval pass rate, retrieval-groundedness rate)
     to decide whether to promote the treatment to 100% traffic.

This is intentionally a simple, auditable split -- not a bandit algorithm
-- because in a clinical-adjacent product, predictable/explainable
routing matters more than optimizing the split in real time.
"""

import hashlib
import logging
from dataclasses import dataclass

from config.settings import settings

log = logging.getLogger(__name__)

BASE_VARIANT = "base"
TREATMENT_VARIANT = settings.vllm_model_name  # the fine-tuned model's served name


def assign_variant(session_id: str) -> str:
    """Hash-based deterministic bucketing: same session_id always maps to
    the same variant, and the split approximates
    settings.ab_test_treatment_traffic_pct across many sessions."""
    digest = hashlib.sha256(session_id.encode()).hexdigest()
    bucket = int(digest, 16) % 1000 / 1000.0  # value in [0, 1)
    variant = (
        TREATMENT_VARIANT
        if bucket < settings.ab_test_treatment_traffic_pct
        else BASE_VARIANT
    )
    return variant


@dataclass
class OutcomeRecord:
    session_id: str
    variant: str
    safety_pass: bool
    had_grounded_answer: bool  # RAG returned passages above relevance floor
    latency_ms: float


class ABTestAnalyzer:
    """Aggregates OutcomeRecords (e.g. pulled from logs or a metrics store)
    and reports per-variant summary stats. In production, back this with a
    real datastore query instead of an in-memory list."""

    def __init__(self):
        self.records: list[OutcomeRecord] = []

    def add(self, record: OutcomeRecord) -> None:
        self.records.append(record)

    def summarize(self) -> dict:
        summary = {}
        for variant in {BASE_VARIANT, TREATMENT_VARIANT}:
            variant_records = [r for r in self.records if r.variant == variant]
            n = len(variant_records)
            if n == 0:
                summary[variant] = {"n": 0}
                continue
            summary[variant] = {
                "n": n,
                "safety_pass_rate": sum(r.safety_pass for r in variant_records) / n,
                "grounded_answer_rate": sum(
                    r.had_grounded_answer for r in variant_records
                )
                / n,
                "avg_latency_ms": sum(r.latency_ms for r in variant_records) / n,
            }
        return summary

    def recommend_promotion(self) -> bool:
        """Simple promotion gate: treatment must not regress on safety
        pass rate or grounded-answer rate vs. base, within a small margin.
        Real deployments should use a proper statistical significance test
        (e.g. a two-proportion z-test) given enough sample size -- this is
        a directional check, not a substitute for that."""
        summary = self.summarize()
        base, treatment = summary.get(BASE_VARIANT, {}), summary.get(
            TREATMENT_VARIANT, {}
        )
        if base.get("n", 0) < 30 or treatment.get("n", 0) < 30:
            log.warning(
                "Insufficient sample size for a promotion decision (need >=30/variant)"
            )
            return False

        margin = 0.02  # allow up to 2 percentage points of noise
        safety_ok = treatment["safety_pass_rate"] >= base["safety_pass_rate"] - margin
        grounded_ok = (
            treatment["grounded_answer_rate"] >= base["grounded_answer_rate"] - margin
        )
        return safety_ok and grounded_ok
