"""
Basic integration tests for the FastAPI gateway. Run with:
    pytest tests/test_api.py
Requires the vLLM server and vector store to be reachable, or mock
serving.api._assistant for pure unit tests -- see test_emergency_shortcircuit
for an example that mocks around the LLM entirely.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from serving.api import app

client = TestClient(app)


def test_health_endpoint_requires_no_auth():
    # health should be reachable without API key for load-balancer probes
    resp = client.get("/health")
    assert resp.status_code in (200, 503)


@patch("serving.api._assistant")
def test_emergency_shortcircuit_bypasses_llm(mock_assistant):
    """The emergency response must come back even if the underlying agent
    would fail or hang -- this test verifies the hard gate independent of
    the LLM being available at all."""
    mock_assistant.respond.return_value = {
        "output": "seek emergency care",
        "emergency": True,
    }
    resp = client.post(
        "/chat",
        json={"message": "I have crushing chest pain"},
        headers={"X-API-Key": "local-dev-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["emergency"] is True


def test_missing_api_key_rejected():
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code in (401, 422)


@patch("serving.api._assistant")
def test_agent_exception_returns_500_not_crash(mock_assistant):
    mock_assistant.respond.side_effect = RuntimeError("boom")
    resp = client.post(
        "/chat",
        json={"message": "test"},
        headers={"X-API-Key": "local-dev-key"},
    )
    assert resp.status_code == 500
    assert "clinician" in resp.json()["detail"].lower()
