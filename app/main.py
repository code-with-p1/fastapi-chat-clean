from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from typing import List
from app.vector_factory import get_vector_store
from app.vectordb.reranker import rerank_results

from app.config import get_settings
from app.middleware.tracing import TracingMiddleware
from app.routes import chat, health
from app.services import redis_service

settings = get_settings()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

class SetupIndexRequest(BaseModel):
    db_provider: str         # "pinecone", "weaviate", "qdrant", "milvus"
    index_name: str          # Name of the collection/index
    dimension: int = 1536    # Default to OpenAI text-embedding-3-small dimension
    recreate: bool = False   # Set to True to drop existing collections with the same name

class IngestRequest(BaseModel):
    db_provider: str  # "pinecone", "weaviate", "qdrant"
    documents: List[str]

class QueryRequest(BaseModel):
    db_provider: str
    query: str
    top_k: int = 10
    rerank_top_n: int = 5

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═══ %s v%s starting ═══", settings.app_name, settings.app_version)
    await redis_service.init_redis()
    yield
    await redis_service.close_redis()
    logger.info("Shutdown complete ✓")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Async chat API — FastAPI + LangChain + OpenAI streaming + Redis sessions",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-Request-ID", "X-Session-ID"],
    )
    app.add_middleware(TracingMiddleware)

    app.include_router(health.router)
    app.include_router(chat.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "type": type(exc).__name__})

    return app

app = create_app()

@app.post("/api/setup_index")
async def setup_database_index(req: SetupIndexRequest):
    """
    Dynamically creates the schema, collections, and index configurations 
    tailored to the specific vector database provider.
    """
    try:
        store = get_vector_store(req.db_provider)
        result_message = store.create_index(
            index_name=req.index_name, 
            dimension=req.dimension, 
            recreate=req.recreate
        )
        return {
            "status": "success", 
            "provider": req.db_provider,
            "message": result_message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_documents(req: IngestRequest):
    try:
        store = get_vector_store(req.db_provider)
        store.ingest(req.documents)
        return {"status": "success", "message": f"Ingested {len(req.documents)} docs into {req.db_provider}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrieve")
async def retrieve_and_rerank(req: QueryRequest):
    try:
        # 1. Dynamic DB Selection & Hybrid Search
        store = get_vector_store(req.db_provider)
        initial_results = store.hybrid_search(req.query, top_k=req.top_k)
        
        # 2. Automatic Cohere Re-Ranking
        final_results = rerank_results(
            query=req.query, 
            retrieved_docs=initial_results, 
            top_n=req.rerank_top_n
        )
        
        # 3. Formulate RAG Context
        context = "\n".join([doc["text"] for doc in final_results])
        
        return {
            "query": req.query,
            "provider": req.db_provider,
            "context_ready": context,
            "metadata": final_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
