import logging
import sys
import time
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.rag.retriever import semantic_search
from backend.llm.openai_client import chat_completion
from backend.llm.prompt import build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

app = FastAPI(title="Solver Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's problem description or query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    type: Optional[str] = Field(default=None, pattern=r"^(algorithms|projects|datasets)$", description="Filter by document type")


class ResultItem(BaseModel):
    type: str
    name: str
    path: str
    summary: str
    similarity: float


class RecommendResponse(BaseModel):
    query: str
    results: list[ResultItem]
    answer: str
    elapsed_ms: float


# ── Routes ──────────────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    start = time.perf_counter()
    logger.info("POST /recommend query=%.80s... top_k=%d type=%s", req.query, req.top_k, req.type or "all")

    try:
        results = await semantic_search(req.query, top_k=req.top_k, doc_type=req.type)
    except Exception:
        logger.exception("Semantic search failed")
        raise HTTPException(status_code=502, detail="Embedding or database error")

    if not results:
        logger.warning("No results found for query")
        raise HTTPException(status_code=404, detail="No matching documents found")

    prompt = build_prompt(req.query, results)

    try:
        answer = await chat_completion(prompt)
    except Exception:
        logger.exception("LLM chat completion failed")
        raise HTTPException(status_code=502, detail="LLM service error")

    elapsed = (time.perf_counter() - start) * 1000
    logger.info("POST /recommend completed in %.0fms", elapsed)

    return RecommendResponse(
        query=req.query,
        results=[
            ResultItem(
                type=r["type"],
                name=r["name"],
                path=r["path"],
                summary=r["summary"],
                similarity=round(r["similarity"], 4),
            )
            for r in results
        ],
        answer=answer,
        elapsed_ms=round(elapsed, 1),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.recommend:app", host="0.0.0.0", port=8000, reload=True)
