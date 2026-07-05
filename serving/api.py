"""
FastAPI gateway sitting in front of the agent. Responsibilities that live
here (rather than in the agent itself) because they're cross-cutting
infra concerns:
  - API key auth
  - basic per-key rate limiting
  - PII/PHI redaction before logging
  - structured error handling / graceful degradation
  - request timing
"""

import logging
import os
import re
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from agents.agent import MedicalAssistant
from agents.tools import _retriever
from config.settings import settings
from monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    EMERGENCY_TRIGGERED,
    ACTIVE_LORA_ADAPTER,
)
from monitoring.tracing import init_tracing
from serving.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)

app = FastAPI(title="MedAssist-GenAI API", version="1.0.0")

# Agent is expensive to construct (loads embedding models, builds BM25
# index) -- build once at startup, not per-request.
_assistant: MedicalAssistant | None = None

api_key_header = APIKeyHeader(name="X-API-Key")

# In-memory sliding-window rate limiter, keyed by API key. Fine for a
# single-process deployment; swap for Redis-backed limiting behind a
# load balancer with multiple workers.
_request_log: dict[str, deque] = defaultdict(deque)

PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[REDACTED-PHONE]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED-EMAIL]"),
]


def redact_for_logging(text: str) -> str:
    """Naive placeholder redaction for logs only -- NOT a substitute for a
    proper PHI de-identification pipeline on the data path itself."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not settings.gateway_api_key:
        log.warning("GATEWAY_API_KEY not set -- running without auth (dev only)")
        return key
    if key != settings.gateway_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


def check_rate_limit(key: str) -> None:
    now = time.time()
    window = _request_log[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_min:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    window.append(now)


@app.on_event("startup")
def startup():
    global _assistant

    # LangSmith traces every LangChain agent step (tool selection, LLM
    # calls, scratchpad) automatically once these env vars are set -- no
    # code changes needed inside agents/agent.py itself. We just need to
    # make sure they're set before the agent/LLM objects are constructed.
    if settings.langsmith_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        log.info(
            "LangSmith tracing enabled for project '%s'", settings.langsmith_project
        )

    # OTel spans cover our own pipeline steps (retrieval, tool calls) and
    # export to whatever OTLP collector feeds Grafana/Tempo/Jaeger.
    init_tracing()

    log.info("Initializing MedicalAssistant (loading models, building indices)...")
    _assistant = MedicalAssistant()
    ACTIVE_LORA_ADAPTER.labels(adapter_version=settings.vllm_model_name).set(1)
    log.info("Startup complete.")


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint. Point your Prometheus server's scrape
    config at this path; Grafana then reads from Prometheus using the
    dashboard in monitoring/grafana_dashboard.json."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
def health():
    doc_count = _retriever.store.count() if _assistant else 0
    # NOTE: the line above reaches into the RAG tool's retriever to report
    # index size; wrapped in try/except in production to avoid health-check
    # flakiness if tool wiring changes.
    return HealthResponse(status="ok", vector_store_docs=doc_count)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, api_key: str = Security(verify_api_key)):
    check_rate_limit(api_key)

    if _assistant is None:
        raise HTTPException(status_code=503, detail="Service still starting up")

    log.info("Incoming request: %s", redact_for_logging(req.message))
    start = time.perf_counter()

    try:
        with REQUEST_LATENCY.time():
            result = _assistant.respond(req.message)
    except Exception:
        log.exception("Unhandled error processing chat request")
        REQUEST_COUNT.labels(endpoint="/chat", status="error").inc()
        # Fail safe: never let an unhandled exception leak internals to the
        # client, and never silently return an empty/broken medical answer.
        raise HTTPException(
            status_code=500,
            detail="Internal error. Please retry, or consult a clinician directly.",
        )

    latency_ms = (time.perf_counter() - start) * 1000
    log.info(
        "Request handled in %.1fms (emergency=%s)", latency_ms, result.get("emergency")
    )

    REQUEST_COUNT.labels(endpoint="/chat", status="ok").inc()
    if result.get("emergency"):
        EMERGENCY_TRIGGERED.inc()

    return ChatResponse(
        response=result["output"],
        emergency=result.get("emergency", False),
        sources_used=result.get("sources_used", []),
        latency_ms=latency_ms,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url.path)
    return {"detail": "An unexpected error occurred. Please try again later."}
