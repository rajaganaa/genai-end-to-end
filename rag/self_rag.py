"""
Self-RAG-inspired reflection layer: after retrieval, the LLM critiques
whether the retrieved passages actually support answering the question
before we let it generate a final answer. This catches the case where
retrieval returns passages that are topically related but don't actually
contain the needed fact (which a pure relevance-score floor can miss --
a passage can be highly "relevant" in embedding space while still not
answering the specific question asked).

Reference: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and
Critique through Self-Reflection" (2023) -- this is a lightweight,
prompt-based approximation of that idea rather than a model trained with
the paper's special reflection tokens.
"""

import logging
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from config.settings import settings

log = logging.getLogger(__name__)

REFLECTION_PROMPT = """You are grading whether retrieved passages are \
sufficient to answer a clinical question. Respond with a single line in \
the exact format:
CONFIDENCE: <a number from 0.0 to 1.0>
REASON: <one short sentence>

A high confidence (>0.7) means the passages directly and specifically \
answer the question. A low confidence (<0.4) means the passages are only \
tangentially related or missing key specifics (e.g. dosage, contraindication,
patient population) needed to actually answer it.

Question: {query}

Retrieved passages:
{passages}

Your grading:"""


@dataclass
class SelfRAGResult:
    confidence: float
    reason: str
    should_answer: bool


class SelfRAGReflector:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model_name,
            temperature=0.0,
            max_tokens=100,
            timeout=20,
        )

    def grade(self, query: str, passages_text: str) -> SelfRAGResult:
        if not settings.enable_self_rag:
            # Feature disabled -- treat everything as sufficiently confident
            return SelfRAGResult(
                confidence=1.0, reason="self-RAG disabled", should_answer=True
            )

        if not passages_text.strip():
            return SelfRAGResult(
                confidence=0.0, reason="no passages retrieved", should_answer=False
            )

        try:
            response = self.llm.invoke(
                REFLECTION_PROMPT.format(query=query, passages=passages_text)
            )
            text = response.content.strip()
            confidence = 0.0
            reason = "unparsed"
            for line in text.splitlines():
                if line.upper().startswith("CONFIDENCE:"):
                    confidence = float(line.split(":", 1)[1].strip())
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            should_answer = confidence >= settings.self_rag_confidence_floor
            log.debug("Self-RAG grading: confidence=%.2f reason=%r", confidence, reason)
            return SelfRAGResult(
                confidence=confidence, reason=reason, should_answer=should_answer
            )
        except Exception:
            log.exception(
                "Self-RAG grading failed -- defaulting to cautious 'insufficient'"
            )
            # Fail closed: if we can't verify sufficiency, don't let the
            # agent answer as if it were sufficient.
            return SelfRAGResult(
                confidence=0.0, reason="grading error", should_answer=False
            )
