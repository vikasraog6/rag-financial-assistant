import logging
import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
from pipeline.rag_chain import get_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_rag_chain()
    yield

app = FastAPI(title="RAG Financial Data Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    num_docs_retrieved: int
    latency_ms: float

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_financial_data(request: QueryRequest):
    start_time = time.time()
    try:
        chain = get_rag_chain()
        result = chain.query(request.question)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            num_docs_retrieved=result["num_docs_retrieved"],
            latency_ms=round((time.time() - start_time) * 1000, 1),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail="RAG pipeline temporarily unavailable. Please retry.")

@app.get("/tickers")
async def list_tickers():
    try:
        chain = get_rag_chain()
        results = chain.vectorstore.get(include=["metadatas"])
        tickers = list({m["ticker"] for m in results["metadatas"]})
        return {"tickers": sorted(tickers), "count": len(tickers)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))