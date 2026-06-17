"""
Unit tests for the FastAPI endpoints.

All LLM / ChromaDB calls are mocked — these tests verify request validation,
response shape, and error handling without external dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch the RAG chain before the app module is imported so the startup
# lifespan doesn't attempt a real ChromaDB connection.
@pytest.fixture(scope="module")
def client() -> TestClient:
    with (
        patch("pipeline.rag_chain._vectorstore"),
        patch("pipeline.rag_chain.build_chain"),
    ):
        from app.api.main import app

        with TestClient(app) as c:
            yield c


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_ok(client: TestClient) -> None:
    with patch("app.api.main.build_chain"):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "chain_ready" in body
    assert body["version"] == "1.0.0"


# ── /tickers ─────────────────────────────────────────────────────────────────

def test_list_tickers(client: TestClient) -> None:
    resp = client.get("/tickers")
    assert resp.status_code == 200
    tickers = resp.json()["tickers"]
    assert len(tickers) == 10
    for expected in ("AAPL", "MSFT", "NVDA", "TSLA"):
        assert expected in tickers


# ── /query validation ─────────────────────────────────────────────────────────

def test_query_too_short(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "Hi"})
    assert resp.status_code == 422


def test_query_invalid_ticker_filter(client: TestClient) -> None:
    resp = client.post(
        "/query",
        json={"question": "Tell me about this stock.", "ticker_filter": ["FAKE"]},
    )
    assert resp.status_code == 422
    assert "Unsupported ticker" in resp.json()["detail"][0]["msg"]


def test_query_ticker_filter_case_insensitive(client: TestClient) -> None:
    """Lowercase tickers should be normalised to uppercase."""
    mock_result = {
        "answer": "AAPL is bullish.",
        "sources": [
            {
                "ticker": "AAPL",
                "trade_date": "2024-06-01",
                "trend_signal": "bullish",
                "volatility_bucket": "low",
                "snippet": "Close: $185.00",
            }
        ],
    }
    with patch("app.api.main.rag_query", return_value=mock_result):
        resp = client.post(
            "/query",
            json={"question": "What is AAPL doing?", "ticker_filter": ["aapl"]},
        )
    assert resp.status_code == 200


# ── /query success ────────────────────────────────────────────────────────────

def test_query_returns_expected_shape(client: TestClient) -> None:
    mock_result = {
        "answer": "NVDA is in a bullish trend with strong momentum.",
        "sources": [
            {
                "ticker": "NVDA",
                "trade_date": "2024-06-15",
                "trend_signal": "bullish",
                "volatility_bucket": "high",
                "snippet": "Ticker: NVDA | Date: 2024-06-15 | Close: $1200.00",
            }
        ],
    }
    with patch("app.api.main.rag_query", return_value=mock_result):
        resp = client.post("/query", json={"question": "What is the trend for NVDA?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == mock_result["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["ticker"] == "NVDA"
    assert isinstance(body["latency_ms"], float)


# ── /query error handling ─────────────────────────────────────────────────────

def test_query_rag_failure_returns_503(client: TestClient) -> None:
    with patch("app.api.main.rag_query", side_effect=RuntimeError("ChromaDB down")):
        resp = client.post("/query", json={"question": "What is AAPL doing today?"})
    assert resp.status_code == 503
