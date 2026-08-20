"""
api.py — FastAPI server exposing POST /query for the RAG engine.

Usage:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag

app = FastAPI(
    title="Financial Report Q&A Engine",
    description="Ask questions about 10-K filings via RAG.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class SourceCitation(BaseModel):
    source_file: str
    page: int | str
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    try:
        result = rag.query(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return QueryResponse(
        answer=result.answer,
        sources=[SourceCitation(**s) for s in result.sources],
    )
