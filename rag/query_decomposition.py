"""
Query decomposition: breaks a multi-hop clinical question ("Is drug X safe
for a patient with condition Y who is also taking drug Z?") into simpler
sub-questions that can each be answered by a single retrieval pass, then
lets the agent synthesize across the sub-answers.

Without this, a single dense/hybrid retrieval call over a compound
question tends to retrieve passages that are each only partially relevant,
because the embedding of the compound question is a blurry average of its
parts.
"""

import json
import logging

from langchain_openai import ChatOpenAI

from config.settings import settings

log = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """Break the following clinical question into 2-4 \
simpler, independently-answerable sub-questions if -- and only if -- it \
genuinely requires combining multiple pieces of information (e.g. a drug \
interaction check, a condition + medication interplay, or a multi-step \
reasoning chain). If the question is already simple/single-hop, return it \
as the only item.

Respond ONLY with a JSON array of strings, no other text.

Question: {query}

JSON array:"""


class QueryDecomposer:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model_name,
            temperature=0.0,  # deterministic decomposition
            max_tokens=300,
            timeout=20,
        )

    def decompose(self, query: str) -> list[str]:
        if not settings.enable_query_decomposition:
            return [query]

        try:
            response = self.llm.invoke(DECOMPOSITION_PROMPT.format(query=query))
            raw = response.content.strip()
            # Strip accidental markdown code fences if the model adds them
            raw = (
                raw.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            sub_questions = json.loads(raw)
            if not isinstance(sub_questions, list) or not sub_questions:
                raise ValueError("Decomposition did not return a non-empty list")
            return [str(q) for q in sub_questions][
                :4
            ]  # hard cap: avoid runaway fan-out
        except Exception:
            log.warning(
                "Query decomposition failed/malformed, falling back to single query",
                exc_info=True,
            )
            return [query]
