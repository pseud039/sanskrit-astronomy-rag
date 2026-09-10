from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval import bm25_search, semantic_search  # imports from P3's file

app = FastAPI()

# Required — browsers block cross-origin requests from your Next.js
# frontend unless the API explicitly allows it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend URL before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest):
    bm25_results = bm25_search(request.query, request.top_k)
    semantic_results = semantic_search(request.query, request.top_k)

    return {
        "query": request.query,
        "bm25": bm25_results.to_dict(orient="records"),
        "semantic": semantic_results.to_dict(orient="records"),
    }