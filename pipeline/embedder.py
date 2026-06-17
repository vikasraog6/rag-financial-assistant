"""
Embedding pipeline: BigQuery → ChromaDB

Reads the fct_stock_summary mart, formats each row as a richly-structured
text document, embeds it with OpenAI text-embedding-3-small, and upserts
the vectors into a ChromaDB persistent collection.

Usage:
    # Full load
    python -m pipeline.embedder

    # Incremental (only rows newer than a given date)
    python -m pipeline.embedder --since 2024-06-01

    # Dry run (fetch + format, skip the upsert)
    python -m pipeline.embedder --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from datetime import date
from typing import Iterator

import chromadb
from google.cloud import bigquery
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "stock_data"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

GCP_PROJECT = os.environ["GCP_PROJECT"]
BQ_DATASET = os.getenv("BQ_DATASET", "marts")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))


def _doc_id(ticker: str, trade_date: str) -> str:
    """Stable SHA-1 ID so repeated upserts are idempotent."""
    return hashlib.sha1(f"{ticker}:{trade_date}".encode()).hexdigest()


def _row_to_document(row: bigquery.Row) -> Document:
    """
    Serialise a BigQuery result row into a LangChain Document.

    The page_content is written as structured prose so the embedding captures
    semantic meaning; metadata fields are kept flat for retrieval filtering.
    """
    content = (
        f"Ticker: {row.ticker} | Date: {row.trade_date}\n"
        f"Open: ${row.open_price:.2f} | High: ${row.high_price:.2f} | "
        f"Low: ${row.low_price:.2f} | Close: ${row.close_price:.2f}\n"
        f"Volume: {row.volume:,} | 30d Avg Volume: {row.avg_volume_30d:,.0f} | "
        f"Volume vs Avg: {row.volume_ratio:.2f}x\n"
        f"Daily Return: {row.daily_return_pct:.2%} | "
        f"7d Return: {row.return_7d_pct:.2%} | "
        f"30d Return: {row.return_30d_pct:.2%}\n"
        f"MA(7): ${row.ma_7d:.2f} | MA(30): ${row.ma_30d:.2f} | MA(90): ${row.ma_90d:.2f}\n"
        f"Trend Signal: {row.trend_signal} | "
        f"30d Volatility: {row.volatility_bucket} ({row.volatility_30d:.4f})\n"
        f"Intraday Range: ${row.intraday_range:.2f} ({row.intraday_range_pct:.2%})"
    )

    return Document(
        page_content=content,
        metadata={
            "id": _doc_id(row.ticker, str(row.trade_date)),
            "ticker": row.ticker,
            "trade_date": str(row.trade_date),
            "trend_signal": row.trend_signal,
            "volatility_bucket": row.volatility_bucket,
            "is_latest": bool(row.is_latest),
        },
    )


def _fetch_rows(
    client: bigquery.Client,
    since: date | None = None,
) -> Iterator[bigquery.Row]:
    """Stream rows from fct_stock_summary; add a WHERE clause for incremental runs."""
    where = f"WHERE trade_date >= '{since}'" if since else ""
    sql = f"""
        SELECT
            ticker, trade_date,
            open_price, high_price, low_price, close_price,
            volume, avg_volume_30d, volume_ratio,
            daily_return_pct, return_7d_pct, return_30d_pct,
            ma_7d, ma_30d, ma_90d,
            trend_signal, volatility_30d, volatility_bucket,
            intraday_range, intraday_range_pct, is_latest
        FROM `{GCP_PROJECT}.{BQ_DATASET}.fct_stock_summary`
        {where}
        ORDER BY ticker, trade_date
    """
    logger.info("Querying BigQuery (since=%s)…", since)
    yield from client.query(sql).result()


def _batched(it: Iterator, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks from an iterator."""
    batch: list = []
    for item in it:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def run(since: date | None = None, dry_run: bool = False) -> int:
    """
    Execute the embedding pipeline.

    Args:
        since:   Earliest trade_date to embed. None = full reload.
        dry_run: Fetch and format documents but skip ChromaDB writes.

    Returns:
        Total number of documents processed.
    """
    bq_client = bigquery.Client(project=GCP_PROJECT)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    total = 0
    for batch_rows in _batched(_fetch_rows(bq_client, since), BATCH_SIZE):
        docs = [_row_to_document(r) for r in batch_rows]
        ids = [d.metadata["id"] for d in docs]

        if not dry_run:
            vectorstore.add_documents(documents=docs, ids=ids)

        total += len(docs)
        logger.info(
            "Processed batch of %d (running total: %d, dry_run=%s)",
            len(docs), total, dry_run,
        )

    logger.info("Embedding run complete — %d total documents.", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Embed BigQuery stock data into ChromaDB.")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Only embed rows on or after this date (incremental mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and format documents without writing to ChromaDB.",
    )
    args = parser.parse_args()
    run(since=args.since, dry_run=args.dry_run)
