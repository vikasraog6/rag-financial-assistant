"""
Unit tests for the RAG chain module.

All external calls (ChromaDB, OpenAI) are mocked to keep tests fast and
hermetic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Reset lru_cache between tests to prevent cross-test state leakage."""
    from pipeline import rag_chain

    rag_chain._vectorstore.cache_clear()
    rag_chain.build_chain.cache_clear()
    yield
    rag_chain._vectorstore.cache_clear()
    rag_chain.build_chain.cache_clear()


# ── _vectorstore ─────────────────────────────────────────────────────────────

def test_vectorstore_is_cached() -> None:
    """_vectorstore() should return the same object on repeated calls."""
    mock_vs = MagicMock()
    with (
        patch("pipeline.rag_chain.chromadb.HttpClient"),
        patch("pipeline.rag_chain.Chroma", return_value=mock_vs),
        patch("pipeline.rag_chain.OpenAIEmbeddings"),
    ):
        from pipeline.rag_chain import _vectorstore

        vs1 = _vectorstore()
        vs2 = _vectorstore()
        assert vs1 is vs2


# ── build_chain ───────────────────────────────────────────────────────────────

def test_build_chain_returns_runnable() -> None:
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()

    with (
        patch("pipeline.rag_chain._vectorstore", return_value=mock_vs),
        patch("pipeline.rag_chain.ChatOpenAI"),
        patch("pipeline.rag_chain.create_retrieval_chain") as mock_crc,
        patch("pipeline.rag_chain.create_stuff_documents_chain"),
    ):
        from pipeline.rag_chain import build_chain

        chain = build_chain()
        assert chain is mock_crc.return_value


# ── query ─────────────────────────────────────────────────────────────────────

def _make_mock_doc(ticker: str = "AAPL", date: str = "2024-06-01") -> MagicMock:
    doc = MagicMock()
    doc.page_content = f"Ticker: {ticker} | Date: {date} | Close: $185.00"
    doc.metadata = {
        "ticker": ticker,
        "trade_date": date,
        "trend_signal": "bullish",
        "volatility_bucket": "low",
    }
    return doc


def test_query_no_filter_uses_cached_chain() -> None:
    """Without a ticker_filter, query() should reuse the cached chain."""
    mock_doc = _make_mock_doc()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "AAPL is bullish.",
        "context": [mock_doc],
    }

    with patch("pipeline.rag_chain.build_chain", return_value=mock_chain):
        from pipeline.rag_chain import query

        result = query("What is AAPL doing?")

    assert result["answer"] == "AAPL is bullish."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["ticker"] == "AAPL"


def test_query_with_filter_builds_fresh_chain() -> None:
    """With a ticker_filter, query() should construct a fresh retriever with the filter."""
    mock_doc = _make_mock_doc("NVDA", "2024-06-15")
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "NVDA is bullish.",
        "context": [mock_doc],
    }
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()

    with (
        patch("pipeline.rag_chain._vectorstore", return_value=mock_vs),
        patch("pipeline.rag_chain.create_retrieval_chain", return_value=mock_chain),
        patch("pipeline.rag_chain.create_stuff_documents_chain"),
        patch("pipeline.rag_chain.ChatOpenAI"),
    ):
        from pipeline.rag_chain import query

        result = query("What is NVDA doing?", ticker_filter=["NVDA"])

    mock_vs.as_retriever.assert_called_once()
    call_kwargs = mock_vs.as_retriever.call_args.kwargs["search_kwargs"]
    assert call_kwargs["filter"] == {"ticker": "NVDA"}
    assert result["answer"] == "NVDA is bullish."


def test_query_source_snippet_truncated() -> None:
    """Snippets in sources should be at most 250 characters."""
    long_content = "x" * 500
    mock_doc = MagicMock()
    mock_doc.page_content = long_content
    mock_doc.metadata = {
        "ticker": "MSFT",
        "trade_date": "2024-06-01",
        "trend_signal": "neutral",
        "volatility_bucket": "medium",
    }
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"answer": "MSFT is neutral.", "context": [mock_doc]}

    with patch("pipeline.rag_chain.build_chain", return_value=mock_chain):
        from pipeline.rag_chain import query

        result = query("Tell me about MSFT.")

    assert len(result["sources"][0]["snippet"]) == 250
