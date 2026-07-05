"""
OpenTelemetry instrumentation for the RAG and agent pipeline.

Design note: we use OTel spans for our *own* pipeline steps (retrieval,
reranking, tool calls) and LangSmith for *agent-internal* tracing (the
LLM's reasoning/tool-selection loop), because LangSmith understands
LangChain's agent scratchpad natively while OTel gives us vendor-neutral
spans we can export to any backend (Jaeger, Tempo, Honeycomb, etc.) via
Grafana/Prometheus stack. Together they cover both "what did the agent
decide" and "where did the latency/errors happen."
"""
import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config.settings import settings

log = logging.getLogger(__name__)

_tracer: trace.Tracer | None = None


def init_tracing(service_name: str = "medassist-genai") -> trace.Tracer:
    """Idempotent tracer setup -- safe to call multiple times (e.g. from
    both api.py startup and a standalone script)."""
    global _tracer
    if _tracer is not None:
        return _tracer

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service_name)
    log.info("OpenTelemetry tracing initialized, exporting to %s", settings.otel_exporter_endpoint)
    return _tracer


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        return init_tracing()
    return _tracer


# --- Convenience context managers for the spans we care about most ---

def traced_retrieval(query: str):
    """Usage: with traced_retrieval(query) as span: ... span.set_attribute(...)"""
    tracer = get_tracer()
    span_cm = tracer.start_as_current_span("rag.retrieve")
    return span_cm, query


def traced_llm_call(model_name: str):
    tracer = get_tracer()
    return tracer.start_as_current_span(
        "llm.generate", attributes={"llm.model": model_name}
    )


def traced_tool_call(tool_name: str):
    tracer = get_tracer()
    return tracer.start_as_current_span(
        "agent.tool_call", attributes={"tool.name": tool_name}
    )
