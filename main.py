# ============================================================
# BHAIRAV AI — SECURE FASTAPI MAIN APPLICATION
# ============================================================

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import os
import time

from pydantic import BaseModel
from typing import List, Optional, Union

# ── Rate Limiter ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── ENV CONFIG ──────────────────────────────────────────────
API_KEY = os.getenv("APP_API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# ── Request / Response Models ───────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    include_citations: Optional[bool] = True

class Citation(BaseModel):
    id: str
    source: str
    book: str
    chapter: Optional[Union[str, int]] = None
    verse: Optional[Union[str, int]] = None
    text: str
    tier: int

class RetrievedChunk(BaseModel):
    id: str
    source: str
    book: str
    chapter: Optional[Union[str, int]] = None
    verse: Optional[Union[str, int]] = None
    hindi_summary: str
    tier: int
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    sources_used: List[str]

class HealthResponse(BaseModel):
    status: str
    faiss_vectors: int
    bm25_corpus_size: int
    metadata_entries: int
    embedding_model: str
    llm_model: str

# ── Security Helpers ────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def is_malicious(query: str) -> bool:
    patterns = [
        "ignore previous instructions",
        "system prompt",
        "jailbreak",
        "act as",
    ]
    q = query.lower()
    return any(p in q for p in patterns)

# ── App Lifecycle ───────────────────────────────────────────
from retriever import Retriever
from reranker import Reranker
from generator import Generator
from config import EMBEDDING_MODEL, GROQ_MODEL, RERANK_TOP_N

retriever = None
reranker = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, reranker, generator

    print("Starting Bhairav AI...")

    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()

    print("System ready")

    yield

    print("Shutting down Bhairav AI")

# ── FastAPI App ─────────────────────────────────────────────
app = FastAPI(
    title="Bhairav AI",
    version="2.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
    )

# ── CORS (secured) ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "running", "service": "Bhairav AI"}

@app.get("/health", response_model=HealthResponse)
async def health():
    if not retriever:
        raise HTTPException(status_code=503, detail="System not ready")

    return HealthResponse(
        status="healthy",
        faiss_vectors=retriever.faiss_index.ntotal,
        bm25_corpus_size=len(retriever.chunk_ids),
        metadata_entries=len(retriever.metadata),
        embedding_model=EMBEDDING_MODEL,
        llm_model=GROQ_MODEL,
    )

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(
    request: QueryRequest,
    _: str = Depends(verify_api_key),
):
    if not retriever or not reranker or not generator:
        raise HTTPException(status_code=503, detail="System not ready")

    query_text = request.query.strip()

    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query_text) > 500:
        raise HTTPException(status_code=400, detail="Query too long")

    if is_malicious(query_text):
        raise HTTPException(status_code=400, detail="Invalid query")

    start_time = time.time()

    # ── Retrieval ───────────────────────────────────────────
    candidates, _, _, _ = retriever.retrieve(query_text, top_k=20)

    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # ── Reranking ───────────────────────────────────────────
    top_chunks = reranker.rerank(
        query_text,
        candidates,
        top_n=request.top_k or RERANK_TOP_N,
    )

    # ── Generation ──────────────────────────────────────────
    answer, citations = generator.generate(query_text, top_chunks[:15])

    elapsed = round(time.time() - start_time, 2)

    retrieved_chunks = [
        RetrievedChunk(
            id=c["id"],
            source=c["source"],
            book=c["book"],
            chapter=c.get("chapter"),
            verse=c.get("verse"),
            hindi_summary=c["hindi_summary"],
            tier=c["tier"],
            score=round(c.get("rerank_score", c.get("score", 0.0)), 4),
        )
        for c in top_chunks
    ]

    citation_objects = [
        Citation(
            id=cit["id"],
            source=cit["source"],
            book=cit.get("book_canonical", cit.get("book", "")),
            chapter=cit.get("chapter"),
            verse=cit.get("verse"),
            text=cit["text"],
            tier=cit["tier"],
        )
        for cit in citations
    ] if request.include_citations else []

    sources_used = list({c["source"] for c in top_chunks})

    return QueryResponse(
        query=query_text,
        answer=answer,
        citations=citation_objects,
        retrieved_chunks=retrieved_chunks,
        sources_used=sources_used,
    )

# ── Local Run ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)