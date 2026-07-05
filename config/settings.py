"""
Centralized configuration, loaded once and imported everywhere else.
Keeping this in one place avoids scattered os.getenv() calls and makes
it obvious what the system depends on at a glance.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- vLLM / model serving ---
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    vllm_model_name: str = os.getenv("VLLM_MODEL_NAME", "medical-lora-13b")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local-dev-key")

    # --- Vector store ---
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    vector_collection: str = os.getenv("VECTOR_COLLECTION", "medical_kb")

    # --- Gateway ---
    gateway_api_key: str = os.getenv("GATEWAY_API_KEY", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    rate_limit_per_min: int = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

    # --- RAG tuning ---
    retrieval_top_k: int = 8          # candidates pulled before reranking
    rerank_top_k: int = 3             # final passages sent to the LLM
    min_relevance_score: float = 0.35  # below this -> "insufficient evidence"

    # --- Safety ---
    emergency_keywords_path: str = "agents/emergency_keywords.txt"

    # --- Observability ---
    langsmith_tracing: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_project: str = os.getenv("LANGCHAIN_PROJECT", "medassist-genai")
    otel_exporter_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # --- MLOps ---
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "medassist-qlora-sft")
    model_registry_name: str = os.getenv("MODEL_REGISTRY_NAME", "medassist-medical-lora")
    ab_test_treatment_traffic_pct: float = float(os.getenv("AB_TREATMENT_PCT", "0.5"))
    retrain_trigger_score_drop: float = float(os.getenv("RETRAIN_TRIGGER_DROP", "0.05"))

    # --- Advanced RAG ---
    enable_hyde: bool = os.getenv("ENABLE_HYDE", "true").lower() == "true"
    enable_query_decomposition: bool = os.getenv("ENABLE_QUERY_DECOMP", "true").lower() == "true"
    enable_self_rag: bool = os.getenv("ENABLE_SELF_RAG", "true").lower() == "true"
    self_rag_confidence_floor: float = float(os.getenv("SELF_RAG_CONFIDENCE_FLOOR", "0.4"))
    parent_chunk_size: int = 2048   # tokens, the "parent" context window
    child_chunk_size: int = 512     # tokens, what actually gets embedded/matched


settings = Settings()
