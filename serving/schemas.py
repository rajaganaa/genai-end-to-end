from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="For multi-turn conversations")


class ChatResponse(BaseModel):
    response: str
    emergency: bool = False
    sources_used: list[str] = []
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    vector_store_docs: int
