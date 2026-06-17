"""
RAG chain: ChromaDB retrieval + GPT-4o-mini generation.

The chain is lazily initialised on first use and cached for the process
lifetime so FastAPI workers don't rebuild it on every request.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "stock_data"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

SYSTEM_PROMPT = """\
You are a senior financial analyst assistant specialising in US equity markets.
Answer ONLY from the stock data in the context below. Always cite specific
figures (prices, dates, percentages, ticker symbols). If the data is
insufficient to answer confidently, say so — never speculate.

{context}
"""


@lru_cache(maxsize=1)
def _vectorstore() -> Chroma:
    """Lazy-init the ChromaDB vectorstore, cached for process lifetime."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    logger.info("Connecting to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
    )


@lru_cache(maxsize=1)
def build_chain() -> Runnable:
    """
    Construct the retrieval-augmented generation chain.

    Uses Maximum Marginal Relevance (MMR) retrieval for result diversity:
    fetches `fetch_k` candidates and re-ranks to `k` by balancing relevance
    against redundancy.
    """
    vs = _vectorstore()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_K * 3},
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    logger.info(
        "RAG chain ready — model=%s embed=%s k=%d", CHAT_MODEL, EMBED_MODEL, RETRIEVAL_K
    )
    return chain


def _make_filter(ticker_filter: list[str]) -> dict[str, Any]:
    """Build a ChromaDB $in metadata filter from a list of tickers."""
    if len(ticker_filter) == 1:
        return {"ticker": ticker_filter[0]}
    return {"ticker": {"$in": ticker_filter}}


def query(
    question: str,
    ticker_filter: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run a RAG query against the financial vector store.

    Args:
        question:      Natural language question from the user.
        ticker_filter: Restrict context retrieval to these tickers.

    Returns:
        ``{"answer": str, "sources": list[dict]}``
    """
    if ticker_filter:
        # Build a fresh retriever with the metadata filter applied at query time.
        vs = _vectorstore()
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K * 3,
                "filter": _make_filter(ticker_filter),
            },
        )
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{input}")]
        )
        chain = create_retrieval_chain(
            retriever, create_stuff_documents_chain(llm, prompt)
        )
    else:
        chain = build_chain()

    result = chain.invoke({"input": question})

    sources = [
        {
            "ticker": doc.metadata.get("ticker", ""),
            "trade_date": doc.metadata.get("trade_date", ""),
            "trend_signal": doc.metadata.get("trend_signal", ""),
            "volatility_bucket": doc.metadata.get("volatility_bucket", ""),
            "snippet": doc.page_content[:250],
        }
        for doc in result.get("context", [])
    ]

    logger.info(
        "Query complete — sources=%d ticker_filter=%s question=%r",
        len(sources), ticker_filter, question[:80],
    )
    return {"answer": result["answer"], "sources": sources}
