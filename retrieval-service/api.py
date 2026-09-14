import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_retrieval = {}

def _load_retrieval():
    import retrieval
    return retrieval

@asynccontextmanager
async def lifespan(app: FastAPI):
    retrieval = await asyncio.to_thread(_load_retrieval)  # <-- runs in a worker thread, event loop stays free
    _retrieval["bm25_search"] = retrieval.bm25_search
    _retrieval["semantic_search"] = retrieval.semantic_search
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    ready = "bm25_search" in _retrieval
    return {"status": "ok" if ready else "warming up"}


@app.post("/search")
def search(request: SearchRequest):
    bm25_results = _retrieval["bm25_search"](request.query, request.top_k)
    semantic_results = _retrieval["semantic_search"](request.query, request.top_k)
    return {
        "query": request.query,
        "bm25": bm25_results.to_dict(orient="records"),
        "semantic": semantic_results.to_dict(orient="records"),
    }