"""
Builds the LangChain AgentExecutor that ties the fine-tuned LLM (served via
vLLM's OpenAI-compatible endpoint) together with the tools in tools.py.
"""
import logging
import re

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from agents.prompts import AGENT_SYSTEM_PROMPT, EMERGENCY_RESPONSE
from agents.tools import ALL_TOOLS, check_emergency
from config.settings import settings

log = logging.getLogger(__name__)

_SOURCE_TAG_PATTERN = re.compile(r"\[Source:\s*([^\|\]]+?)(?:\s*\||\])")

# Matches an entire "[Source: ...]" tag (including any trailing
# "| relevance=..." suffix) for stripping, as opposed to _SOURCE_TAG_PATTERN
# above which only captures the name for extraction.
_FULL_CITATION_TAG_PATTERN = re.compile(r"\s*\[Source:[^\]]*\]")


def build_llm() -> ChatOpenAI:
    """vLLM exposes an OpenAI-compatible API, so we can use LangChain's
    standard OpenAI client pointed at our own endpoint -- no vLLM-specific
    LangChain integration needed."""
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model_name,
        temperature=0.2,   # low temperature: we want consistent, cite-backed answers
        max_tokens=800,
        timeout=60,
    )


def build_agent_executor() -> AgentExecutor:
    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=6,          # cap tool-call loops to avoid runaway cost/latency
        max_execution_time=45,     # seconds, guards against a hanging tool call
        handle_parsing_errors=True,  # don't crash the request on a malformed tool call
        return_intermediate_steps=True,  # needed to extract sources_used below
    )


def _extract_sources(intermediate_steps: list) -> list[str]:
    """Pulls distinct source names out of every tool observation in this
    run (MedicalKnowledgeSearch emits "[Source: ...]" tags)."""
    sources: list[str] = []
    for _action, observation in intermediate_steps:
        if not isinstance(observation, str):
            continue
        for match in _SOURCE_TAG_PATTERN.findall(observation):
            name = match.strip()
            if name and name not in sources:
                sources.append(name)
    return sources


def _sanitize_citations(output: str, valid_sources: list[str]) -> str:
    """Deterministic guard against fabricated citations: strips any
    "[Source: ...]" tag in the LLM's final answer that doesn't match one
    of the real document names a tool actually returned this turn.

    Why this exists as code, not just a prompt instruction: the system
    prompt tells the model to cite real document names and never a tool's
    own name, but prompt-following is a *soft* constraint. We observed the
    model append "[Source: MedicalKnowledgeSearch]" -- a tool name, not a
    document -- to an "insufficient evidence" answer, which looks
    misleadingly credible. This function is the same "hard gate over soft
    prompt" pattern already used for emergency detection (see
    agents/tools.py::check_emergency): don't trust the model to always
    comply, verify the claim against ground truth and strip it if it
    doesn't match.

    Any tag that doesn't correspond to a real returned source is removed
    entirely rather than left in some rewritten form, since we have no
    reliable way to know what the model "meant" -- removing is the safe
    default (better to look uncited than falsely cited)."""
    def _replace(match: re.Match) -> str:
        tag_text = match.group(0)
        if any(src.lower() in tag_text.lower() for src in valid_sources):
            return tag_text  # matches a real source this turn -- keep it
        log.warning("Stripped fabricated/unverifiable citation tag: %r", tag_text.strip())
        return ""

    return _FULL_CITATION_TAG_PATTERN.sub(_replace, output).rstrip()


class MedicalAssistant:
    """Application-facing wrapper: adds the hard-coded emergency
    short-circuit *before* the agent (and therefore the LLM) is ever invoked."""

    def __init__(self):
        self.executor = build_agent_executor()

    def respond(self, user_input: str, chat_history: list | None = None) -> dict:
        # Hard safety gate -- runs regardless of what the LLM would have said.
        emergency_check = check_emergency(user_input)
        if emergency_check != "NO_EMERGENCY_DETECTED":
            return {"output": emergency_check, "emergency": True, "tool_calls": [], "sources_used": []}

        try:
            result = self.executor.invoke(
                {"input": user_input, "chat_history": chat_history or []}
            )
            sources = _extract_sources(result.get("intermediate_steps", []))
            clean_output = _sanitize_citations(result["output"], sources)
            return {"output": clean_output, "emergency": False, "sources_used": sources}
        except Exception:
            log.exception("Agent execution failed")
            return {
                "output": (
                    "I'm unable to process this request right now. "
                    "Please try again or consult a clinician directly."
                ),
                "emergency": False,
                "error": True,
                "sources_used": [],
            }
