"""
Airflow DAG: ingest_financial_data

Downloads daily OHLCV data for 10 US equities from yfinance and lands each
ticker as a date-partitioned JSON object in S3. A manifest file is written
last so downstream consumers can detect a complete load.

Schedule: 10 PM UTC Mon–Fri (4–5 hours after US market close for data availability).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import boto3
import yfinance as yf
from airflow.decorators import dag, task
from airflow.models import Variable
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "JNJ",
]

S3_BUCKET: str = Variable.get("S3_BUCKET", default_var="rag-financial-data")
S3_PREFIX: str = "raw/stock_prices"

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": True,
    "email_on_retry": False,
}


@dag(
    dag_id="ingest_financial_data",
    description="Fetch daily OHLCV stock prices from yfinance and land to S3.",
    schedule="0 22 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["finance", "ingestion", "s3", "yfinance"],
    max_active_runs=1,
    doc_md=__doc__,
)
def ingest_financial_data() -> None:
    """DAG factory — Airflow discovers this via the returned DAG object."""

    @task()
    def fetch_stock_data(ticker: str, **context: Any) -> dict[str, Any]:
        """
        Download one trading day of OHLCV data for a single ticker.

        Returns a serialisable dict so XCom can pass it to the upload task.
        An empty DataFrame (non-trading day) results in rows=0 and no S3 write.
        """
        execution_date: datetime = context["data_interval_end"]
        trade_date = execution_date.strftime("%Y-%m-%d")
        logger.info("Fetching %s for %s", ticker, trade_date)

        stock = yf.Ticker(ticker)
        df = stock.history(start=trade_date, end=trade_date, interval="1d")

        if df.empty:
            logger.warning(
                "No data for %s on %s — likely a non-trading day or data lag.",
                ticker, trade_date,
            )
            return {"ticker": ticker, "date": trade_date, "rows": 0}

        df = df.reset_index()
        df["ticker"] = ticker
        df["ingested_at"] = datetime.utcnow().isoformat()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        return {
            "ticker": ticker,
            "date": trade_date,
            "rows": len(df),
            "data": df.to_json(orient="records", date_format="iso"),
        }

    @task()
    def upload_to_s3(payload: dict[str, Any]) -> str:
        """
        Write one ticker's payload to S3 under a Hive-style partition prefix.

        Key pattern: raw/stock_prices/date=YYYY-MM-DD/ticker=XXX/data.json
        Returns the full S3 URI, or 'skipped' when rows=0.
        """
        if payload.get("rows", 0) == 0:
            logger.info("Skipping S3 upload for %s — no rows.", payload["ticker"])
            return "skipped"

        s3 = boto3.client("s3")
        key = (
            f"{S3_PREFIX}"
            f"/date={payload['date']}"
            f"/ticker={payload['ticker']}"
            f"/data.json"
        )

        try:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=payload["data"].encode(),
                ContentType="application/json",
                ServerSideEncryption="AES256",
                Metadata={
                    "ticker": payload["ticker"],
                    "trade_date": payload["date"],
                    "row_count": str(payload["rows"]),
                },
            )
        except ClientError as exc:
            logger.error("S3 upload failed for %s: %s", payload["ticker"], exc)
            raise

        uri = f"s3://{S3_BUCKET}/{key}"
        logger.info("Uploaded %s (%d rows)", uri, payload["rows"])
        return uri

    @task()
    def write_manifest(s3_uris: list[str], **context: Any) -> None:
        """
        Write a _manifest.json that lists every successfully uploaded object
        for the day. Downstream pipelines poll this file to confirm load completeness.
        """
        execution_date: datetime = context["data_interval_end"]
        trade_date = execution_date.strftime("%Y-%m-%d")

        successful = [u for u in s3_uris if u != "skipped"]
        manifest = {
            "schema_version": "1.0",
            "date": trade_date,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "ticker_count": len(successful),
            "paths": successful,
        }

        s3 = boto3.client("s3")
        key = f"{S3_PREFIX}/date={trade_date}/_manifest.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(manifest, indent=2).encode(),
            ContentType="application/json",
        )
        logger.info(
            "Manifest written: s3://%s/%s (%d tickers)", S3_BUCKET, key, len(successful)
        )

    # Dynamic task mapping — one fetch + upload pair per ticker, run in parallel.
    payloads = fetch_stock_data.expand(ticker=TICKERS)
    uris = upload_to_s3.expand(payload=payloads)
    write_manifest(uris)


ingest_financial_data()
