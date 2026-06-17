"""
Streamlit chat UI for the RAG Financial Data Assistant.

Communicates with the FastAPI backend at API_BASE_URL (default: localhost:8000).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

SUPPORTED_TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "JNJ",
]

EXAMPLE_QUESTIONS: list[str] = [
    "What is the current trend signal for NVDA?",
    "Compare the 30-day returns of AAPL and MSFT.",
    "Which stocks have the highest 30-day volatility?",
    "Is TSLA bullish or bearish based on its moving averages?",
    "Which tickers are trading above their 90-day moving average?",
    "Show me stocks where volume is significantly above the 30-day average.",
    "What happened to META's price over the last 7 days?",
    "Which tech stocks have the lowest volatility right now?",
]


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Financial Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .source-card { background:#f0f2f6; border-radius:8px; padding:10px 14px; margin-bottom:8px; }
    .source-card b { color:#1a1a2e; }
    .badge-bullish  { background:#d4edda; color:#155724; border-radius:4px; padding:2px 6px; font-size:0.8em; }
    .badge-bearish  { background:#f8d7da; color:#721c24; border-radius:4px; padding:2px 6px; font-size:0.8em; }
    .badge-neutral  { background:#fff3cd; color:#856404; border-radius:4px; padding:2px 6px; font-size:0.8em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 RAG Financial\nAssistant")
    st.caption(
        "Powered by **GPT-4o-mini** · **ChromaDB** · **LangChain** · **dbt + BigQuery**"
    )
    st.divider()

    st.subheader("🔍 Ticker Filter")
    selected_tickers: list[str] = st.multiselect(
        "Restrict context to",
        options=SUPPORTED_TICKERS,
        default=[],
        help="Leave blank to search across all 10 tickers.",
    )

    st.divider()
    st.subheader("💡 Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state["pending_question"] = q

    st.divider()

    # API health indicator
    with st.container():
        try:
            resp = httpx.get(f"{API_BASE}/health", timeout=3.0)
            health = resp.json()
            if health.get("chain_ready"):
                st.success("API  ·  Connected", icon="✅")
            else:
                st.warning("API  ·  Degraded", icon="⚠️")
        except Exception:
            st.error("API  ·  Unreachable", icon="🔴")
            st.caption(f"Expected at `{API_BASE}`")


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None


# ── Chat history ──────────────────────────────────────────────────────────────
st.header("💬 Ask a Financial Question")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            sources: list[dict] = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source document(s)"):
                    for src in sources:
                        trend_cls = f"badge-{src.get('trend_signal', 'neutral').lower()}"
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<b>{src['ticker']}</b> &nbsp;·&nbsp; {src['trade_date']} &nbsp;"
                            f"<span class='{trend_cls}'>{src['trend_signal']}</span> &nbsp;"
                            f"<span style='font-size:0.8em;color:#555;'>"
                            f"volatility: {src['volatility_bucket']}</span><br/>"
                            f"<small>{src['snippet']}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            if latency := msg.get("latency_ms"):
                st.caption(f"_Response time: {latency} ms_")


# ── Input ─────────────────────────────────────────────────────────────────────
user_input: str | None = st.chat_input("Ask about stocks, trends, volatility…")

# Example button fills the chat input via session state
if st.session_state["pending_question"] and not user_input:
    user_input = st.session_state.pop("pending_question")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Retrieving relevant data…_")

        payload: dict[str, Any] = {"question": user_input}
        if selected_tickers:
            payload["ticker_filter"] = selected_tickers

        try:
            resp = httpx.post(f"{API_BASE}/query", json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()

            placeholder.markdown(data["answer"])

            sources = data.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source document(s)"):
                    for src in sources:
                        trend_cls = f"badge-{src.get('trend_signal', 'neutral').lower()}"
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<b>{src['ticker']}</b> &nbsp;·&nbsp; {src['trade_date']} &nbsp;"
                            f"<span class='{trend_cls}'>{src['trend_signal']}</span> &nbsp;"
                            f"<span style='font-size:0.8em;color:#555;'>"
                            f"volatility: {src['volatility_bucket']}</span><br/>"
                            f"<small>{src['snippet']}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            if latency := data.get("latency_ms"):
                st.caption(f"_Response time: {latency} ms_")

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": sources,
                    "latency_ms": data.get("latency_ms"),
                }
            )

        except httpx.HTTPStatusError as exc:
            err_msg = f"API error {exc.response.status_code}: {exc.response.text[:200]}"
            placeholder.error(err_msg)
            st.session_state["messages"].append({"role": "assistant", "content": err_msg})
        except httpx.TimeoutException:
            err_msg = "Request timed out. The RAG chain may be cold-starting — please retry."
            placeholder.warning(err_msg)
            st.session_state["messages"].append({"role": "assistant", "content": err_msg})
        except Exception as exc:
            err_msg = f"Unexpected error: {exc}"
            placeholder.error(err_msg)
            st.session_state["messages"].append({"role": "assistant", "content": err_msg})
