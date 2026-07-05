"""
HyDE (Hypothetical Document Embeddings): instead of embedding the raw user
query, we ask the LLM to first write a hypothetical answer/passage, then
embed *that* and search with it. This closes the "question vs. answer"
phrasing gap -- clinical questions are often phrased very differently from
how the answer appears in a guideline document, and a hypothetical answer
tends to be lexically/semantically closer to the real passage we want to
retrieve than the bare question is.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (2022).
"""

import logging

from langchain_openai import ChatOpenAI

from config.settings import settings

log = logging.getLogger(__name__)

HYDE_PROMPT_TEMPLATE = """You are a medical reference writer. Write a short, \
factual passage (3-5 sentences) that would plausibly answer the following \
clinical question, as it might appear in a clinical guideline document. \
Do not hedge or add disclaimers -- just write the hypothetical reference \
passage itself.

Question: {query}

Hypothetical passage:"""


class HyDEGenerator:
    def __init__(self):
        # A cheaper/faster call than the main agent LLM is fine here --
        # this is a retrieval-quality booster, not a user-facing answer,
        # so it doesn't need the fine-tuned adapter's clinical tone.
        self.llm = ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model_name,  # base (non-LoRA) model is enough for this step
            temperature=0.3,
            max_tokens=200,
            timeout=20,
        )

    def generate_hypothetical_passage(self, query: str) -> str:
        try:
            response = self.llm.invoke(HYDE_PROMPT_TEMPLATE.format(query=query))
            return response.content.strip()
        except Exception:
            log.exception("HyDE generation failed, falling back to raw query")
            return query  # graceful degradation: just use the original query

    def embed_query(self, query: str) -> str:
        """Returns the text to actually embed/search with -- either the
        hypothetical passage (if HyDE is enabled) or the raw query."""
        if not settings.enable_hyde:
            return query
        hypothetical = self.generate_hypothetical_passage(query)
        log.debug("HyDE hypothetical passage for %r: %r", query, hypothetical)
        return hypothetical
